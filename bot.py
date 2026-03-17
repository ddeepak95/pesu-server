#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Voice Assessment Bot.

The example runs a simple voice AI bot that you can connect to using your
browser and speak with it. You can also deploy this bot to Pipecat Cloud.

Features:
- Voice-based formative assessment
- Per-utterance audio recording to Firebase Storage
- Voice message logging to Supabase

Required AI services:
- OpenAI (STT + LLM)
- Cartesia (Text-to-Speech)

Run the bot using::

    uv run bot.py
"""

import asyncio
import io
import os
from pydoc import text
import wave
from collections import deque
from datetime import datetime, timezone

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams

logger.info("✅ Silero VAD model loaded")

from pipecat.frames.frames import (
    LLMRunFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    LLMTextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    EndTaskFrame,
    TTSSpeakFrame,
)

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    UserTurnStoppedMessage,
    AssistantTurnStoppedMessage,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.services.cartesia.tts import CartesiaTTSService, language_to_cartesia_language
from pipecat.services.tts_service import TextAggregationMode
from pipecat.services.google.llm import GoogleLLMService, GoogleThinkingConfig
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig
from LANGUAGE_CONSTANTS import LANGUAGES
from firebase_storage import upload_audio, generate_audio_path, generate_session_audio_chunk_path
from supabase_client import (
    log_voice_message,
    update_voice_message_audio,
    update_voice_message_interrupted,
    update_voice_message_content,
    append_session_audio_chunk,
)

logger.info("✅ All components loaded successfully!")


def audio_to_wav(audio_bytes: bytes, sample_rate: int, num_channels: int) -> bytes:
    """Convert raw audio bytes to WAV format.
    
    Args:
        audio_bytes: Raw PCM audio data
        sample_rate: Sample rate in Hz
        num_channels: Number of audio channels
    
    Returns:
        WAV-formatted audio bytes
    """
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    return wav_buffer.getvalue()

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")
    body = getattr(runner_args, 'body', {})
    logger.info(f"Body: {body}")
    
    # Apply Supabase environment override from frontend (for correct dev/prod targeting)
    supabase_env = body.get("supabase_env")
    if supabase_env:
        from supabase_client import set_supabase_env
        set_supabase_env(supabase_env)
        logger.info(f"Supabase environment override applied: {supabase_env}")
    
    language_key = body.get("language", "en")
    
    # Check if this is an assessment question or a general topic
    question_prompt = body.get("question_prompt")
    rubric = body.get("rubric", [])
    assignment_id = body.get("assignment_id")
    question_order = body.get("question_order")
    
    # New: Custom prompts from frontend (already interpolated)
    # If provided, these take precedence over the hardcoded prompts
    custom_system_prompt = body.get("system_prompt")
    custom_greeting = body.get("greeting")
    
    # Session metadata for audio recording
    submission_id = body.get("submission_id")
    attempt_number = body.get("attempt_number", 1)
    
    # Separate counters: transcripts and audio events do NOT arrive in lockstep.
    # - transcript_counter: used for "how many rows did we log?"
    # - audio_counter: used for unique audio filenames
    transcript_counter = {"user": 0, "bot": 0}
    audio_counter = {"user": 0, "bot": 0}

    # Full LLM-generated assistant responses (always full, even if speech is interrupted)
    full_assistant_text_queue = deque()
    is_disconnecting = False
    disconnect_bot_row_id: str | None = None

    # ID-based correlation: maps utterance_id -> row or audio item
    # utterance_id = f"{submission_id}:{question_order}:{attempt_number}:user|bot:{ordinal}"
    user_transcripts: dict[str, dict] = {}
    user_audio: dict[str, dict] = {}
    bot_transcripts: dict[str, dict] = {}
    bot_audio: dict[str, dict] = {}

    # Prevent concurrent flushes from double-popping
    user_flush_lock = asyncio.Lock()
    bot_flush_lock = asyncio.Lock()
    background_tasks: set[asyncio.Task] = set()
    
    # Track bot speaking state for interruption detection
    # speaking: True while bot is outputting audio (set by frame handlers)
    # current_bot_interrupted: set to True when user starts speaking while bot is speaking
    bot_state = {"speaking": False, "current_bot_interrupted": False}
    
    language = LANGUAGES[language_key]["pipecat_language"]
    language_name = LANGUAGES[language_key]["name"]
    cartesia_voice_id = LANGUAGES[language_key]["cartesia_voice_id"]
    static_greeting = LANGUAGES[language_key].get("greeting")

    # Build prompt based on whether it's an assessment or general conversation
    # TTS instruction to append for voice mode
    TTS_INSTRUCTION = "\n\nThe text you generate will be used by TTS to speak to the student, so don't include any special characters or formatting. Use colloquial language and be friendly."
    
    if custom_system_prompt:
        # New path: Use frontend-provided prompt (already interpolated)
        logger.info(f"Using custom system prompt from frontend - Assignment: {assignment_id}, Question: {question_order}")
        prompt = custom_system_prompt + TTS_INSTRUCTION
    elif question_prompt:
        # Assessment mode: focused on specific question (backward compatibility)
        logger.info(f"Assessment mode (legacy) - Assignment: {assignment_id}, Question: {question_order}")
        rubric_text = "\n".join([f"- {item['item']} ({item['points']} points)" for item in rubric]) if rubric else "No specific rubric provided."
        
        # Build prompt with two-phase instructions for first question
        if question_order == 0:
            prompt = f"""You are a teacher conducting a voice-based formative assessment in {language_name}. Your name is Konvo.

The student needs to answer this question:
{question_prompt}

Evaluation criteria:
{rubric_text}

CONVERSATION FLOW (for first question only):
Phase 1 - Introduction:
1. Introduce yourself as "Konvo" in {language_name}
2. Say: "We are going to do an activity today"
3. Ask: "If you are ready to start, say that we can start"
4. Wait for the student to acknowledge they are ready (they may say "yes", "ready", "let's start", "we can start", or similar phrases)
5. Do NOT explain the question until the student acknowledges readiness

Phase 2 - Question Explanation (after student acknowledges readiness):
1. Once the student acknowledges they are ready, explain the question clearly
2. Ask the student to answer the question
3. Proceed with the normal assessment flow below

NORMAL ASSESSMENT FLOW (after question is explained):
1. Have a natural conversation to understand their thinking
2. Ask follow-up questions to gauge depth of understanding
3. Be encouraging and supportive
4. Help them elaborate if they're stuck, but don't give away the answer
5. Keep the questions short and concise
6. Keep your responses very short and concise and conversational.
7. Use English for concept-specific words while keeping the conversation in {language_name}

The text you generate will be used by TTS to speak to the student, so don't include any special characters or formatting. Use colloquial language and be friendly.
"""
        else:
            # For subsequent questions, use the original prompt structure
            prompt = f"""You are a teacher conducting a voice-based formative assessment in {language_name}. 

The student needs to answer this question:
{question_prompt}

Evaluation criteria:
{rubric_text}

Your role:
1. Ask the student to answer the question
2. Have a natural conversation to understand their thinking
3. Ask follow-up questions to gauge depth of understanding
4. Be encouraging and supportive
5. Help them elaborate if they're stuck, but don't give away the answer
6. Keep the questions short and concise.
7. Use English for concept-specific words while keeping the conversation in {language_name}.

The text you generate will be used by TTS to speak to the student, so don't include any special characters or formatting. Use colloquial language and be friendly. Keep your responses concise and conversational.
"""
    else:
        # General conversation mode (legacy)
        topic_arg = body.get("topic", "newton's laws of motion and gravity")
        prompt = f"""You are a friendly science teacher who speaks in {language_name}. You have to quiz the student on {topic_arg}. You have to ask the student to solve the problems and give the correct answer. The text you generate will be used by TTS to speak to the student, so don't include any special characters or formatting. Use colloquial language and be friendly. Ask conceptual questions to check the student's understanding of the concepts.
"""

    cartesia_language = language_to_cartesia_language(language)
    # stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"), live_options=deepgram_live_options)
    stt = OpenAISTTService(api_key=os.getenv("OPENAI_API_KEY"), language=language)

    input_params = CartesiaTTSService.InputParams(language=cartesia_language)

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=cartesia_voice_id,
        params=input_params,
        text_aggregation_mode=TextAggregationMode.TOKEN,
    )
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(
            model="gpt-4o",
            max_completion_tokens=250,
        ),
    )



    # llm = GoogleLLMService(
    #     api_key=os.getenv("GEMINI_API_KEY"),
    #     settings=GoogleLLMService.Settings(
    #         model="gemini-2.5-flash",
    #         thinking = GoogleThinkingConfig(thinking_budget = 0)
    #     ),
    # )

    # Define end_conversation function schema
    # The description guides when the LLM should call this function
    end_conversation_function = FunctionSchema(
        name="end_conversation",
        description="End the conversation gracefully. Call this when: (1) the student explicitly refuses to answer (e.g., says 'I refuse', 'I don't want to', 'I can't answer'), or (2) the student has thoroughly answered the question and you're satisfied with their response. Always provide a polite ending message thanking the student.",
        properties={
            "reason": {
                "type": "string",
                "enum": ["refusal", "thorough"],
                "description": "Use 'refusal' if the student explicitly refuses to answer. Use 'thorough' if the student has thoroughly answered the question.",
            },
            "message": {
                "type": "string",
                "description": "A polite ending message in the conversation language thanking the student and indicating the conversation is ending.",
            },
        },
        required=["reason", "message"],
    )

    # Create tools schema with end_conversation function
    tools = ToolsSchema(standard_tools=[end_conversation_function])

    # Function handler for end_conversation
    async def handle_end_conversation(params: FunctionCallParams):
        """Handle end_conversation function call to gracefully terminate the conversation."""
        reason = params.arguments.get("reason", "")
        message = params.arguments.get("message", "")
        
        # Generate default message if not provided (fallback - LLM should provide message per prompt)
        if not message:
            if reason == "refusal":
                message = "I understand. Thank you for your time. The conversation is ending."
            elif reason == "thorough":
                message = "Thank you for your thorough response. The conversation is now ending."
            else:
                message = "Thank you. The conversation is now ending."
            logger.warning(f"No ending message provided by LLM, using default English message for reason: {reason}")
        
        logger.info(f"Ending conversation with reason: {reason}, message: {message[:50]}...")
        
        # Push ending message through TTS
        await params.llm.push_frame(TTSSpeakFrame(message))
        
        # Wait a brief moment to ensure the TTS frame is processed and added to conversation context
        # This ensures the ending message appears in the frontend transcript
        await asyncio.sleep(0.5)
        
        # Push EndTaskFrame upstream to gracefully end the conversation
        await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
    
    # Register the function handler
    llm.register_function("end_conversation", handle_end_conversation)

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
    ]

    context = LLMContext(messages, tools=tools)
    context_aggregator_pair = LLMContextAggregatorPair(context)
    user_aggregator = context_aggregator_pair.user()
    assistant_aggregator = context_aggregator_pair.assistant()

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    class LLMFullResponseCaptureProcessor(FrameProcessor):
        """Captures full assistant responses from streamed LLM frames without affecting TTS."""

        def __init__(self):
            super().__init__()
            self._in_full_response = False
            self._chunks: list[str] = []

        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, LLMFullResponseStartFrame):
                self._in_full_response = True
                self._chunks = []
            elif isinstance(frame, LLMTextFrame) and self._in_full_response:
                # Different Pipecat versions use .text or .content
                text = getattr(frame, "text", None)
                if text is None:
                    text = getattr(frame, "content", "")
                if text:
                    self._chunks.append(text)
            elif isinstance(frame, LLMFullResponseEndFrame) and self._in_full_response:
                full_text = "".join(self._chunks).strip()
                full_assistant_text_queue.append(full_text)
                self._in_full_response = False
                self._chunks = []

            # Always forward frame unchanged
            await self.push_frame(frame, direction)
    
    # Sample rate for buffer_size and WAV conversion (Pipecat/Daily typically 24kHz or 16kHz)
    SAMPLE_RATE = 16000
    CHUNK_DURATION_SEC = 30
    # Audio buffer: per-turn audio + 60-second composite chunks
    audiobuffer = AudioBufferProcessor(
        num_channels=1,
        enable_turn_audio=True,
        user_continuous_stream=True,
        sample_rate=SAMPLE_RATE,
        buffer_size=SAMPLE_RATE * 2 * CHUNK_DURATION_SEC,  # 16-bit mono, 60s chunks
    )
    session_chunk_counter = 0
    session_recording_started_at: str | None = None

    llm_full_capture = LLMFullResponseCaptureProcessor()

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            stt,
            user_aggregator,
            llm,
            llm_full_capture,
            tts,
            transport.output(),
            audiobuffer,
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            # Without this, Pipecat will detect interruption frames but won't stop bot speech.
            allow_interruptions=True,
        ),
        observers=[RTVIObserver(rtvi)],

    )

    # Only fire reached-(up|down)stream events for frames we care about.
    # This also makes debugging interruptions less noisy.
    task.set_reached_upstream_filter(
        (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, UserStartedSpeakingFrame)
    )
    task.set_reached_downstream_filter(
        (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, UserStartedSpeakingFrame)
    )

    def _track_task(t: asyncio.Task):
        background_tasks.add(t)
        t.add_done_callback(lambda _t: background_tasks.discard(_t))

    async def _attach_user_audio(row: dict, audio_item: dict):
        path = generate_audio_path(
            submission_id, question_order, attempt_number, "user", audio_item["audio_num"]
        )
        audio_url = await asyncio.to_thread(upload_audio, audio_item["wav_bytes"], path)
        await asyncio.to_thread(update_voice_message_audio, row["id"], audio_url, None, None)
        logger.info(f"Attached user audio #{audio_item['audio_num']} (row={row['id']})")

    async def _attach_bot_audio(row: dict, audio_item: dict):
        path = generate_audio_path(
            submission_id, question_order, attempt_number, "bot", audio_item["audio_num"]
        )
        audio_url = await asyncio.to_thread(upload_audio, audio_item["wav_bytes"], path)
        interrupted = audio_item.get("interrupted", False)
        await asyncio.to_thread(update_voice_message_audio, row["id"], audio_url, None, interrupted)
        logger.info(f"Attached bot audio #{audio_item['audio_num']} (row={row['id']}), interrupted={interrupted}")

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        """Create voice_messages row from turn event and attach audio if already available (async, non-blocking)."""
        if not submission_id or not assignment_id:
            return
        transcript_counter["user"] += 1
        utterance_id = f"{submission_id}:{question_order}:{attempt_number}:user:{transcript_counter['user']}"
        timestamp = getattr(message, "timestamp", None) or datetime.now(timezone.utc).isoformat()
        if isinstance(timestamp, str):
            spoken_at = timestamp
        else:
            spoken_at = getattr(timestamp, "isoformat", lambda: datetime.now(timezone.utc).isoformat())()
        content = (getattr(message, "content", None) or "").strip()
        logger.debug(f"User turn stopped #{transcript_counter['user']}: {content[:80] if content else '(empty)'}...")

        async def _log_user_turn():
            record = await asyncio.to_thread(
                log_voice_message,
                submission_id,
                assignment_id,
                question_order,
                "student",
                content,
                attempt_number,
                None,
                None,
                False,
                spoken_at,
                None,
                utterance_id,
            )
            if not record or not record.get("id"):
                return
            async with user_flush_lock:
                user_transcripts[utterance_id] = {"id": record["id"]}
                audio_item = user_audio.pop(utterance_id, None)
            if audio_item:
                _track_task(asyncio.create_task(_attach_user_audio({"id": record["id"]}, audio_item)))

        _track_task(asyncio.create_task(_log_user_turn()))

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        """Create voice_messages row from turn event or update disconnect placeholder."""
        nonlocal disconnect_bot_row_id
        if not submission_id or not assignment_id:
            return
        content = (getattr(message, "content", None) or "").strip()
        timestamp = getattr(message, "timestamp", None) or datetime.now(timezone.utc).isoformat()
        spoken_at = timestamp if isinstance(timestamp, str) else getattr(timestamp, "isoformat", lambda: datetime.now(timezone.utc).isoformat())()
        full_text = (full_assistant_text_queue.popleft() if full_assistant_text_queue else "") or content

        if is_disconnecting and disconnect_bot_row_id:
            try:
                await asyncio.to_thread(update_voice_message_content, disconnect_bot_row_id, content, spoken_at, None)
                await asyncio.to_thread(update_voice_message_interrupted, disconnect_bot_row_id, True)
                logger.info("Updated disconnect bot row with late assistant transcript")
            except Exception as e:
                logger.error(f"Failed updating disconnect bot row with transcript: {e}")
            finally:
                disconnect_bot_row_id = None
            return

        transcript_counter["bot"] += 1
        utterance_id = f"{submission_id}:{question_order}:{attempt_number}:bot:{transcript_counter['bot']}"
        logger.debug(f"Assistant turn stopped #{transcript_counter['bot']}: {content[:80] if content else '(empty)'}...")

        async def _log_bot_turn():
            record = await asyncio.to_thread(
                log_voice_message,
                submission_id,
                assignment_id,
                question_order,
                "assistant",
                content,
                attempt_number,
                None,
                None,
                False,
                spoken_at,
                full_text,
                utterance_id,
            )
            if not record or not record.get("id"):
                return
            async with bot_flush_lock:
                bot_transcripts[utterance_id] = {"id": record["id"]}
                audio_item = bot_audio.pop(utterance_id, None)
            if audio_item:
                _track_task(asyncio.create_task(_attach_bot_audio({"id": record["id"]}, audio_item)))

        _track_task(asyncio.create_task(_log_bot_turn()))

    # Frame handlers for bot speaking state (proper interruption detection)
    @task.event_handler("on_frame_reached_upstream")
    async def on_frame_reached_upstream(task, frame):
        """Track bot speaking state for interruption detection."""
        if isinstance(frame, BotStartedSpeakingFrame):
            bot_state["speaking"] = True
            bot_state["current_bot_interrupted"] = False
            logger.debug("Bot started speaking")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            bot_state["speaking"] = False
            logger.debug("Bot stopped speaking")
        elif isinstance(frame, UserStartedSpeakingFrame):
            # If user starts speaking while bot is speaking, it's an interruption
            if bot_state["speaking"]:
                bot_state["current_bot_interrupted"] = True
                logger.info("User started speaking while bot speaking - marking as interruption")

    @task.event_handler("on_frame_reached_downstream")
    async def on_frame_reached_downstream(task, frame):
        """Mirror speaking-state updates downstream (some transports emit frames both ways)."""
        if isinstance(frame, BotStartedSpeakingFrame):
            bot_state["speaking"] = True
            bot_state["current_bot_interrupted"] = False
        elif isinstance(frame, BotStoppedSpeakingFrame):
            bot_state["speaking"] = False
        elif isinstance(frame, UserStartedSpeakingFrame):
            if bot_state["speaking"]:
                bot_state["current_bot_interrupted"] = True

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected - waiting for client_ready signal")

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        nonlocal full_assistant_text_queue, is_disconnecting, disconnect_bot_row_id, session_chunk_counter, session_recording_started_at
        logger.info(f"Client ready - starting conversation")
        await rtvi.set_bot_ready()
        
        if submission_id:
            await audiobuffer.start_recording()
            session_recording_started_at = datetime.now(timezone.utc).isoformat()
            transcript_counter["user"] = 0
            transcript_counter["bot"] = 0
            audio_counter["user"] = 0
            audio_counter["bot"] = 0
            full_assistant_text_queue.clear()
            is_disconnecting = False
            disconnect_bot_row_id = None
            user_transcripts.clear()
            user_audio.clear()
            bot_transcripts.clear()
            bot_audio.clear()
            session_chunk_counter = 0
            bot_state["speaking"] = False
            bot_state["current_bot_interrupted"] = False
            logger.info(f"Audio recording started for submission {submission_id} at {session_recording_started_at}")
        
        # Play a quick static greeting for the selected language while the LLM prepares its first response.
        if static_greeting:
            try:
                logger.info(f"Playing static greeting for language {language_key}")
                await task.queue_frames([TTSSpeakFrame(static_greeting)])
            except Exception as e:
                logger.error(f"Error playing static greeting for language {language_key}: {e}")
        
        # Determine greeting based on question order
        if custom_greeting:
            # New path: Use frontend-provided greeting (already interpolated)
            logger.info(f"Using custom greeting from frontend")
            greeting = custom_greeting
        elif question_order == 0:
            # Legacy path: First question greeting
            greeting = f"Speaking in {language_name}, start by introducing yourself as Konvo and tell that you are their teacher's assistant. Say that we are going to do an activity today. Explain that the student will take part by responding to your questions and they can also ask for doubts if any. Ask them to be in a quieter place to avoid disturbances and ask if the student is ready to start. Ask them to say 'yes' or 'ready' when they are ready to start. Wait for their acknowledgment before explaining the question."
        else:
            # Legacy path: Subsequent question greeting
            # Convert order to ordinal (1 -> second, 2 -> third, etc.)
            ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
            ordinal = ordinals[question_order] if question_order < len(ordinals) else f"{question_order + 1}th"
            greeting = f"Speaking in {language_name}, acknowledge we're moving to the {ordinal} question, then ask the student to answer it."
        
        # Start the conversation (LLM-driven greeting/intro will follow the static greeting)
        messages.append({"role": "system", "content": greeting})
        # Note: bot_state["speaking"] will be set by BotStartedSpeakingFrame handler
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        nonlocal is_disconnecting, disconnect_bot_row_id
        logger.info("Client disconnecting - saving any pending data...")
        is_disconnecting = True
        
        if submission_id:
            # If we disconnect while the bot is speaking, treat the current bot utterance as interrupted.
            bot_was_speaking_on_disconnect = bool(bot_state.get("speaking"))
            if bot_was_speaking_on_disconnect:
                bot_state["current_bot_interrupted"] = True

            # Stop recording - may trigger turn events with accumulated audio
            try:
                await audiobuffer.stop_recording()
            except Exception as e:
                logger.warning(f"Error stopping recording: {e}")
            
            # Wait briefly for turn events / audio to enqueue after stop_recording
            for _ in range(30):
                if any(uid in user_audio for uid in user_transcripts):
                    break
                if any(uid in bot_audio for uid in bot_transcripts):
                    break
                if bot_was_speaking_on_disconnect and bot_audio:
                    break
                await asyncio.sleep(0.1)

            # Bot was speaking at disconnect: ensure one row for in-flight utterance
            if bot_was_speaking_on_disconnect and assignment_id and disconnect_bot_row_id is None:
                # Reuse latest pending bot row (by ordinal) or create placeholder
                bot_uid_keys = [k for k in bot_transcripts if ":bot:" in k]
                if bot_uid_keys:
                    def _bot_ordinal(k):
                        try:
                            return int(k.split(":")[-1])
                        except (IndexError, ValueError):
                            return 0
                    latest_uid = max(bot_uid_keys, key=_bot_ordinal)
                    disconnect_bot_row_id = bot_transcripts[latest_uid]["id"]
                    try:
                        await asyncio.to_thread(update_voice_message_interrupted, disconnect_bot_row_id, True)
                        logger.info("Marked latest pending bot row interrupted on disconnect")
                    except Exception as e:
                        logger.error(f"Error marking pending bot row interrupted: {e}")
                else:
                    spoken_at = datetime.now(timezone.utc).isoformat()
                    generated = full_assistant_text_queue.popleft() if full_assistant_text_queue else ""
                    if not generated:
                        for msg in reversed(messages):
                            if msg.get("role") == "assistant":
                                generated = msg.get("content", "") or ""
                                break
                    placeholder_uid = f"{submission_id}:{question_order}:{attempt_number}:bot:{audio_counter['bot'] + 1}"
                    try:
                        record = await asyncio.to_thread(
                            log_voice_message,
                            submission_id,
                            assignment_id,
                            question_order,
                            "assistant",
                            "",
                            attempt_number,
                            None,
                            None,
                            True,
                            spoken_at,
                            generated or None,
                            placeholder_uid,
                        )
                        if record and record.get("id"):
                            disconnect_bot_row_id = record["id"]
                            bot_transcripts[placeholder_uid] = {"id": record["id"]}
                            logger.info("Logged interrupted bot row on disconnect (no spoken transcript yet)")
                    except Exception as e:
                        logger.error(f"Error logging interrupted bot row on disconnect: {e}")

            # Flush: attach audio to rows by utterance_id
            async with user_flush_lock:
                for uid in list(user_transcripts.keys()):
                    audio_item = user_audio.pop(uid, None)
                    if audio_item:
                        row = user_transcripts.pop(uid)
                        _track_task(asyncio.create_task(_attach_user_audio(row, audio_item)))
            async with bot_flush_lock:
                for uid in list(bot_transcripts.keys()):
                    audio_item = bot_audio.pop(uid, None)
                    if audio_item:
                        row = bot_transcripts.pop(uid)
                        _track_task(asyncio.create_task(_attach_bot_audio(row, audio_item)))

            if bot_was_speaking_on_disconnect and bot_transcripts and not bot_audio:
                for _ in range(15):
                    if bot_audio:
                        break
                    await asyncio.sleep(0.1)
                async with bot_flush_lock:
                    for uid in list(bot_transcripts.keys()):
                        audio_item = bot_audio.pop(uid, None)
                        if audio_item:
                            row = bot_transcripts.pop(uid)
                            _track_task(asyncio.create_task(_attach_bot_audio(row, audio_item)))

            if background_tasks:
                await asyncio.gather(*list(background_tasks), return_exceptions=True)

            # Remaining audio without transcript: insert placeholder row and attach
            async with user_flush_lock:
                remaining_user = list(user_audio.items())
                for uid in (k for k, _ in remaining_user):
                    user_audio.pop(uid, None)
            for uid, audio_item in remaining_user:
                try:
                    if not audio_item.get("wav_bytes"):
                        continue
                    path = generate_audio_path(
                        submission_id, question_order, attempt_number, "user", audio_item["audio_num"]
                    )
                    audio_url = await asyncio.to_thread(upload_audio, audio_item["wav_bytes"], path)
                    spoken_at = datetime.now(timezone.utc).isoformat()
                    await asyncio.to_thread(
                        log_voice_message,
                        submission_id,
                        assignment_id,
                        question_order,
                        "student",
                        "",
                        attempt_number,
                        audio_url,
                        None,
                        False,
                        spoken_at,
                        None,
                        uid,
                    )
                    logger.warning(f"Logged user audio without transcript on disconnect (audio #{audio_item.get('audio_num')})")
                except Exception as e:
                    logger.error(f"Error saving user audio on disconnect: {e}")

            async with bot_flush_lock:
                remaining_bot = [(uid, audio_item, bot_transcripts.pop(uid, None)) for uid, audio_item in list(bot_audio.items())]
                for uid, _ in remaining_bot:
                    bot_audio.pop(uid, None)
            for uid, audio_item, row in remaining_bot:
                try:
                    spoken_at = datetime.now(timezone.utc).isoformat()
                    if row:
                        interrupted = audio_item.get("interrupted", False) or bot_was_speaking_on_disconnect
                        if not audio_item.get("wav_bytes"):
                            await asyncio.to_thread(update_voice_message_interrupted, row["id"], True)
                            continue
                        path = generate_audio_path(
                            submission_id, question_order, attempt_number, "bot", audio_item["audio_num"]
                        )
                        audio_url = await asyncio.to_thread(upload_audio, audio_item["wav_bytes"], path)
                        await asyncio.to_thread(update_voice_message_audio, row["id"], audio_url, None, interrupted)
                        logger.warning(f"Attached bot audio without transcript on disconnect to existing row")
                        continue
                    if not audio_item.get("wav_bytes"):
                        await asyncio.to_thread(
                            log_voice_message,
                            submission_id,
                            assignment_id,
                            question_order,
                            "assistant",
                            "",
                            attempt_number,
                            None,
                            None,
                            True,
                            spoken_at,
                            None,
                            uid,
                        )
                        continue
                    path = generate_audio_path(
                        submission_id, question_order, attempt_number, "bot", audio_item["audio_num"]
                    )
                    audio_url = await asyncio.to_thread(upload_audio, audio_item["wav_bytes"], path)
                    await asyncio.to_thread(
                        log_voice_message,
                        submission_id,
                        assignment_id,
                        question_order,
                        "assistant",
                        "",
                        attempt_number,
                        audio_url,
                        None,
                        True,
                        spoken_at,
                        None,
                        uid,
                    )
                    logger.warning(f"Logged bot audio without transcript on disconnect (audio #{audio_item.get('audio_num')})")
                except Exception as e:
                    logger.error(f"Error saving bot audio on disconnect: {e}")
        
        logger.info(
            f"Client disconnected - logged transcripts: {transcript_counter['user']} user / {transcript_counter['bot']} bot, "
            f"audio chunks: {audio_counter['user']} user / {audio_counter['bot']} bot"
        )
        await task.cancel()
    
    # Per-utterance audio recording handlers (ID-based correlation)
    @audiobuffer.event_handler("on_user_turn_audio_data")
    async def on_user_turn_audio_data(buffer, audio, sample_rate, num_channels):
        """Store user audio by utterance_id; attach to row if transcript already present."""
        if not submission_id or not assignment_id:
            return
        audio_counter["user"] += 1
        audio_num = audio_counter["user"]
        utterance_id = f"{submission_id}:{question_order}:{attempt_number}:user:{audio_num}"
        try:
            wav_bytes = audio_to_wav(audio, sample_rate, num_channels)
            audio_item = {"audio_num": audio_num, "wav_bytes": wav_bytes}
            async with user_flush_lock:
                user_audio[utterance_id] = audio_item
                row = user_transcripts.pop(utterance_id, None)
                if row:
                    audio_item = user_audio.pop(utterance_id, None)
            if row and audio_item:
                _track_task(asyncio.create_task(_attach_user_audio(row, audio_item)))
            else:
                logger.debug(f"Queued user audio (utterance_id={utterance_id})")
        except Exception as e:
            logger.error(f"Error handling user audio #{audio_num}: {e}")

    @audiobuffer.event_handler("on_bot_turn_audio_data")
    async def on_bot_turn_audio_data(buffer, audio, sample_rate, num_channels):
        """Store bot audio by utterance_id; attach to row if transcript already present."""
        was_interrupted = bot_state["current_bot_interrupted"]
        if not submission_id or not assignment_id:
            return
        audio_counter["bot"] += 1
        audio_num = audio_counter["bot"]
        utterance_id = f"{submission_id}:{question_order}:{attempt_number}:bot:{audio_num}"
        try:
            wav_bytes = audio_to_wav(audio, sample_rate, num_channels)
            audio_item = {"audio_num": audio_num, "wav_bytes": wav_bytes, "interrupted": was_interrupted}
            async with bot_flush_lock:
                bot_audio[utterance_id] = audio_item
                row = bot_transcripts.pop(utterance_id, None)
                if row:
                    audio_item = bot_audio.pop(utterance_id, None)
            if row and audio_item:
                _track_task(asyncio.create_task(_attach_bot_audio(row, audio_item)))
            else:
                logger.debug(f"Queued bot audio (utterance_id={utterance_id}), interrupted={was_interrupted}")
        except Exception as e:
            logger.error(f"Error handling bot audio #{audio_num}: {e}")

    @audiobuffer.event_handler("on_audio_data")
    async def on_session_audio_data(buffer, audio, sample_rate, num_channels):
        """Upload composite session chunk (60s) and append URL to submission_session_audio."""
        if not submission_id or not assignment_id:
            return
        nonlocal session_chunk_counter
        try:
            session_chunk_counter += 1
            chunk_index = session_chunk_counter
            wav_bytes = audio_to_wav(audio, sample_rate, num_channels)
            path = generate_session_audio_chunk_path(submission_id, question_order, attempt_number, chunk_index)
            audio_url = await asyncio.to_thread(upload_audio, wav_bytes, path)
            # Pass recording_started_at only for the first chunk (inserted as a new row)
            started_at = session_recording_started_at if chunk_index == 1 else None
            await asyncio.to_thread(append_session_audio_chunk, submission_id, question_order, attempt_number, audio_url, started_at)
            logger.info(f"Session composite chunk #{chunk_index} uploaded for {submission_id}/{question_order}/{attempt_number}")
        except Exception as e:
            logger.error(f"Error uploading session audio chunk: {e}")


    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""

    body = getattr(runner_args, "body", {})
    supabase_env = body.get("supabase_env")
    effective_env = supabase_env or os.getenv("SUPABASE_ENV", "development")
    use_krisp_filter = effective_env == "production"
    running_locally = os.getenv("RUN_LOCAL", "0").lower() in ("1", "true")
    use_krisp_filter = use_krisp_filter and not running_locally
    krisp_filter = None
    if use_krisp_filter:
        from pipecat.audio.filters.krisp_viva_filter import KrispVivaFilter
        krisp_filter = KrispVivaFilter()

    transport = DailyTransport(
        runner_args.room_url,
        runner_args.token,
        "Pipecat Bot",
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,
            video_out_width=1024,
            video_out_height=576,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
            audio_in_filter=krisp_filter,
        ),
    )

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
