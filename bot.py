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
import wave
from collections import deque
from datetime import datetime, timezone

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

logger.info("✅ Local Smart Turn Analyzer V3 loaded")
logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams

logger.info("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    LLMRunFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    LLMTextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService, language_to_cartesia_language
from pipecat.services.cartesia.stt import CartesiaSTTService, CartesiaLiveOptions
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport

from LANGUAGE_CONSTANTS import LANGUAGES
from firebase_storage import upload_audio, generate_audio_path
from supabase_client import (
    log_voice_message,
    update_voice_message_audio,
    update_voice_message_interrupted,
    update_voice_message_content,
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
    language_arg = body.get("language", "en")
    
    # Check if this is an assessment question or a general topic
    question_prompt = body.get("question_prompt")
    rubric = body.get("rubric", [])
    assignment_id = body.get("assignment_id")
    question_order = body.get("question_order")
    
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

    # Pending audio queues - populated by AudioBufferProcessor, consumed when transcript arrives
    # Each entry is a dict with: audio_num, wav_bytes
    pending_user_audio_queue = deque()
    pending_bot_audio_queue = deque()

    # Pending DB rows (transcript-first). Each entry is a dict: {id}
    pending_user_rows = deque()
    pending_bot_rows = deque()

    # Prevent concurrent flushes from double-popping
    user_flush_lock = asyncio.Lock()
    bot_flush_lock = asyncio.Lock()
    background_tasks: set[asyncio.Task] = set()
    
    # Track bot speaking state for interruption detection
    # speaking: True while bot is outputting audio (set by frame handlers)
    # current_bot_interrupted: set to True when user starts speaking while bot is speaking
    bot_state = {"speaking": False, "current_bot_interrupted": False}
    
    language = LANGUAGES[language_arg]["pipecat_language"]
    cartesia_voice_id = LANGUAGES[language_arg]["cartesia_voice_id"]

    # Build prompt based on whether it's an assessment or general conversation
    if question_prompt:
        # Assessment mode: focused on specific question
        logger.info(f"Assessment mode - Assignment: {assignment_id}, Question: {question_order}")
        rubric_text = "\n".join([f"- {item['item']} ({item['points']} points)" for item in rubric]) if rubric else "No specific rubric provided."
        
        prompt = f"""You are a teacher conducting a voice-based formative assessment in {language.value}. 

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
7. Use English for concept-specific words while keeping the conversation in {language.value}.

The text you generate will be used by TTS to speak to the student, so don't include any special characters or formatting. Use colloquial language and be friendly. Keep your responses concise and conversational.
"""
    else:
        # General conversation mode (legacy)
        topic_arg = body.get("topic", "newton's laws of motion and gravity")
        prompt = f"""You are a friendly science teacher who speaks in {language.value}. You have to quiz the student on {topic_arg}. You have to ask the student to solve the problems and give the correct answer. The text you generate will be used by TTS to speak to the student, so don't include any special characters or formatting. Use colloquial language and be friendly. Ask conceptual questions to check the student's understanding of the concepts.
"""

    cartesia_language = language_to_cartesia_language(language)
    # stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"), live_options=deepgram_live_options)
    stt = OpenAISTTService(api_key=os.getenv("OPENAI_API_KEY"), language=language)

    input_params = CartesiaTTSService.InputParams(language=cartesia_language)

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=cartesia_voice_id,
        params=input_params
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

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
    
    # Audio buffer for per-utterance recording
    audiobuffer = AudioBufferProcessor(
        num_channels=1,
        enable_turn_audio=True,  # Enable per-turn audio events
        user_continuous_stream=True,
    )
    
    # TranscriptProcessor to capture transcripts with proper timing
    transcript_processor = TranscriptProcessor()
    llm_full_capture = LLMFullResponseCaptureProcessor()

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            transcript_processor.user(),  # Capture user transcripts AFTER STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            llm_full_capture,  # capture full LLM output for generated_content
            tts,  # TTS
            transport.output(),  # Transport bot output
            transcript_processor.assistant(),  # Capture assistant transcripts AFTER TTS output (what was spoken)
            audiobuffer,  # Audio recording (after output for both user and bot audio)
            context_aggregator.assistant(),  # Assistant spoken responses (keeps conversation memory)
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
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

    # TranscriptProcessor event handler - captures transcripts as they flow through pipeline
    @transcript_processor.event_handler("on_transcript_update")
    async def on_transcript_update(processor, frame):
        """Queue transcripts as they arrive from STT/TTS for matching with audio."""
        nonlocal is_disconnecting, disconnect_bot_row_id
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
            logger.info(
                f"Attached bot audio #{audio_item['audio_num']} (row={row['id']}), interrupted={interrupted}"
            )

        def _track_task(t: asyncio.Task):
            background_tasks.add(t)
            t.add_done_callback(lambda _t: background_tasks.discard(_t))

        async def _flush_user():
            if not submission_id or not assignment_id:
                return
            async with user_flush_lock:
                while pending_user_audio_queue and pending_user_rows:
                    audio_item = pending_user_audio_queue.popleft()
                    row = pending_user_rows.popleft()
                    _track_task(asyncio.create_task(_attach_user_audio(row, audio_item)))

        async def _flush_bot():
            if not submission_id or not assignment_id:
                return
            async with bot_flush_lock:
                while pending_bot_audio_queue and pending_bot_rows:
                    audio_item = pending_bot_audio_queue.popleft()
                    row = pending_bot_rows.popleft()
                    _track_task(asyncio.create_task(_attach_bot_audio(row, audio_item)))

        def _as_iso_timestamp(ts) -> str:
            """TranscriptProcessor timestamps may be datetime, str, or None."""
            if ts is None:
                return datetime.now(timezone.utc).isoformat()
            if isinstance(ts, str):
                return ts
            iso = getattr(ts, "isoformat", None)
            if callable(iso):
                return iso()
            return datetime.now(timezone.utc).isoformat()

        for message in getattr(frame, "messages", []) or []:
            # Be defensive: message can be an object or a dict-like
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
            ts = getattr(message, "timestamp", None)
            if role is None and isinstance(message, dict):
                role = message.get("role")
                content = message.get("content")
                ts = message.get("timestamp")

            timestamp = _as_iso_timestamp(ts)
            content = content or ""

            if role in ("user", "student"):
                transcript_counter["user"] += 1
                logger.debug(
                    f"User transcript #{transcript_counter['user']}: "
                    f"{content[:80] if content else '(empty)'}..."
                )
                # Always log transcript immediately (prevents missing utterances if audio event never fires)
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
                    timestamp,
                    None,
                )
                if record and record.get("id"):
                    pending_user_rows.append({"id": record["id"]})
                await _flush_user()
            elif role in ("assistant", "bot"):
                transcript_counter["bot"] += 1
                logger.debug(
                    f"Assistant transcript #{transcript_counter['bot']}: "
                    f"{content[:80] if content else '(empty)'}..."
                )
                full_text = (full_assistant_text_queue.popleft() if full_assistant_text_queue else "") or content

                # If we disconnected while bot was speaking, we may have already created (or selected)
                # a bot row to represent that in-flight utterance. In that case, UPDATE it instead of INSERTing
                # a second row (which caused duplicates with interrupted True/False).
                if is_disconnecting and disconnect_bot_row_id:
                    try:
                        await asyncio.to_thread(update_voice_message_content, disconnect_bot_row_id, content, timestamp, None)
                        await asyncio.to_thread(update_voice_message_interrupted, disconnect_bot_row_id, True)
                        logger.info("Updated disconnect bot row with late assistant transcript (no duplicate insert)")
                    except Exception as e:
                        logger.error(f"Failed updating disconnect bot row with transcript: {e}")
                    finally:
                        disconnect_bot_row_id = None
                    await _flush_bot()
                    continue

                # Log transcript immediately; interruption is finalized when audio is attached.
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
                    timestamp,
                    full_text,
                )
                if record and record.get("id"):
                    pending_bot_rows.append({"id": record["id"]})
                await _flush_bot()
            else:
                logger.debug(f"Ignoring transcript update message with role={role!r}")

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
        nonlocal full_assistant_text_queue
        nonlocal is_disconnecting, disconnect_bot_row_id
        logger.info(f"Client ready - starting conversation")
        await rtvi.set_bot_ready()
        
        # Start audio recording
        if submission_id:
            await audiobuffer.start_recording()
            transcript_counter["user"] = 0
            transcript_counter["bot"] = 0
            audio_counter["user"] = 0
            audio_counter["bot"] = 0
            full_assistant_text_queue.clear()
            is_disconnecting = False
            disconnect_bot_row_id = None
            pending_user_audio_queue.clear()
            pending_bot_audio_queue.clear()
            pending_user_rows.clear()
            pending_bot_rows.clear()
            bot_state["speaking"] = False
            bot_state["current_bot_interrupted"] = False
            logger.info(f"Audio recording started for submission {submission_id}")
        
        # Determine greeting based on question order
        if question_order == 0:
            greeting = f"Speaking in {language.value}, acknowledge we're starting with the first question, then ask the student to answer it."
        else:
            # Convert order to ordinal (1 -> second, 2 -> third, etc.)
            ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
            ordinal = ordinals[question_order] if question_order < len(ordinals) else f"{question_order + 1}th"
            greeting = f"Speaking in {language.value}, acknowledge we're moving to the {ordinal} question, then ask the student to answer it."
        
        # Start the conversation
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
            
            # Wait briefly for AudioBufferProcessor turn events to enqueue audio.
            # In practice, bot/user turn audio can arrive slightly AFTER stop_recording returns.
            for _ in range(30):  # up to ~3s
                if pending_user_audio_queue and pending_user_rows:
                    break
                if pending_bot_audio_queue and pending_bot_rows:
                    break
                # If we were specifically mid-bot-speech, wait until we see *some* bot audio
                if bot_was_speaking_on_disconnect and pending_bot_audio_queue:
                    break
                await asyncio.sleep(0.1)

            # If bot was speaking at disconnect, we might never receive the assistant transcript (spoken text).
            # Ensure we have exactly ONE row representing this utterance:
            # - Prefer reusing the latest pending bot row (if any)
            # - Otherwise create a placeholder row (content="") with generated_content
            if bot_was_speaking_on_disconnect and assignment_id and disconnect_bot_row_id is None:
                if pending_bot_rows:
                    disconnect_bot_row_id = pending_bot_rows[-1]["id"]
                    try:
                        await asyncio.to_thread(update_voice_message_interrupted, disconnect_bot_row_id, True)
                        logger.info("Marked latest pending bot row interrupted on disconnect")
                    except Exception as e:
                        logger.error(f"Error marking pending bot row interrupted: {e}")
                else:
                    spoken_at = datetime.now(timezone.utc).isoformat()
                    generated = ""
                    if full_assistant_text_queue:
                        generated = full_assistant_text_queue.popleft()
                    else:
                        for msg in reversed(messages):
                            if msg.get("role") == "assistant":
                                generated = msg.get("content", "") or ""
                                break

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
                        )
                        if record and record.get("id"):
                            disconnect_bot_row_id = record["id"]
                            pending_bot_rows.append({"id": record["id"]})
                            logger.info("Logged interrupted bot row on disconnect (no spoken transcript yet)")
                    except Exception as e:
                        logger.error(f"Error logging interrupted bot row on disconnect: {e}")

            # Backwards-compat: old placeholder insertion block removed (handled above)
            if False and bot_was_speaking_on_disconnect and assignment_id:
                spoken_at = datetime.now(timezone.utc).isoformat()
                generated = ""
                if full_assistant_text_queue:
                    generated = full_assistant_text_queue.popleft()
                else:
                    # Fallback: last assistant message in context (if any)
                    for msg in reversed(messages):
                        if msg.get("role") == "assistant":
                            generated = msg.get("content", "") or ""
                            break

                try:
                    record = await asyncio.to_thread(
                        log_voice_message,
                        submission_id,
                        assignment_id,
                        question_order,
                        "assistant",
                        "",  # spoken content unknown (disconnect mid-speech)
                        attempt_number,
                        None,
                        None,
                        True,  # interrupted
                        spoken_at,
                        generated or None,
                    )
                    if record and record.get("id"):
                        pending_bot_rows.append({"id": record["id"]})
                        logger.info("Logged interrupted bot row on disconnect (no spoken transcript yet)")
                except Exception as e:
                    logger.error(f"Error logging interrupted bot row on disconnect: {e}")

            # Try attaching any audio that arrived before disconnect
            # (transcripts were already inserted in on_transcript_update)
            async with user_flush_lock:
                while pending_user_audio_queue and pending_user_rows:
                    audio_item = pending_user_audio_queue.popleft()
                    row = pending_user_rows.popleft()
                    if not audio_item.get("wav_bytes"):
                        logger.warning(f"Skipping empty user audio #{audio_item.get('audio_num')} on disconnect")
                        continue
                    path = generate_audio_path(
                        submission_id, question_order, attempt_number, "user", audio_item["audio_num"]
                    )
                    audio_url = await asyncio.to_thread(upload_audio, audio_item["wav_bytes"], path)
                    await asyncio.to_thread(update_voice_message_audio, row["id"], audio_url, None, None)

            async with bot_flush_lock:
                while pending_bot_audio_queue and pending_bot_rows:
                    audio_item = pending_bot_audio_queue.popleft()
                    row = pending_bot_rows.popleft()
                    interrupted = audio_item.get("interrupted", False)
                    if not audio_item.get("wav_bytes"):
                        # No audio bytes - still update interruption flag on the row
                        await asyncio.to_thread(update_voice_message_interrupted, row["id"], interrupted or True)
                        logger.warning(f"Bot audio #{audio_item.get('audio_num')} empty; updated interrupted only")
                        continue
                    path = generate_audio_path(
                        submission_id, question_order, attempt_number, "bot", audio_item["audio_num"]
                    )
                    audio_url = await asyncio.to_thread(upload_audio, audio_item["wav_bytes"], path)
                    await asyncio.to_thread(
                        update_voice_message_audio,
                        row["id"],
                        audio_url,
                        None,
                        interrupted,
                    )

            # One more short wait + flush: sometimes the bot turn chunk is enqueued after the first flush.
            if bot_was_speaking_on_disconnect and pending_bot_rows and not pending_bot_audio_queue:
                for _ in range(15):  # up to ~1.5s
                    if pending_bot_audio_queue:
                        break
                    await asyncio.sleep(0.1)
                async with bot_flush_lock:
                    while pending_bot_audio_queue and pending_bot_rows:
                        audio_item = pending_bot_audio_queue.popleft()
                        row = pending_bot_rows.popleft()
                        interrupted = audio_item.get("interrupted", False) or True
                        if not audio_item.get("wav_bytes"):
                            await asyncio.to_thread(update_voice_message_interrupted, row["id"], True)
                            continue
                        path = generate_audio_path(
                            submission_id, question_order, attempt_number, "bot", audio_item["audio_num"]
                        )
                        audio_url = await asyncio.to_thread(upload_audio, audio_item["wav_bytes"], path)
                        await asyncio.to_thread(update_voice_message_audio, row["id"], audio_url, None, interrupted)

            # Best-effort wait for background upload/log tasks so we don't exit
            if background_tasks:
                await asyncio.gather(*list(background_tasks), return_exceptions=True)

            # If we have audio but never got transcripts (rare), log placeholders without transcript.
            while pending_user_audio_queue:
                audio_item = pending_user_audio_queue.popleft()
                try:
                    if not audio_item.get("wav_bytes"):
                        logger.warning(f"Dropping empty user audio #{audio_item.get('audio_num')} on disconnect")
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
                    )
                    logger.warning(f"Logged user audio without transcript on disconnect (audio #{audio_item['audio_num']})")
                except Exception as e:
                    logger.error(f"Error saving user audio on disconnect: {e}")

            while pending_bot_audio_queue:
                audio_item = pending_bot_audio_queue.popleft()
                try:
                    spoken_at = datetime.now(timezone.utc).isoformat()
                    # If we have a pending bot row, attach audio to it instead of inserting a new row.
                    if pending_bot_rows:
                        row = pending_bot_rows.popleft()
                        interrupted = audio_item.get("interrupted", False) or bot_was_speaking_on_disconnect
                        if not audio_item.get("wav_bytes"):
                            await asyncio.to_thread(update_voice_message_interrupted, row["id"], True)
                            continue
                        path = generate_audio_path(
                            submission_id, question_order, attempt_number, "bot", audio_item["audio_num"]
                        )
                        audio_url = await asyncio.to_thread(upload_audio, audio_item["wav_bytes"], path)
                        await asyncio.to_thread(update_voice_message_audio, row["id"], audio_url, None, interrupted)
                        logger.warning(f"Attached bot audio without transcript on disconnect to existing row (audio #{audio_item.get('audio_num')})")
                        continue

                    if not audio_item.get("wav_bytes"):
                        # No audio to upload; still log a placeholder row as interrupted.
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
                        )
                        logger.warning(f"Logged bot placeholder (no audio) on disconnect (audio #{audio_item.get('audio_num')})")
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
                    )
                    logger.warning(f"Logged bot audio without transcript on disconnect (audio #{audio_item['audio_num']})")
                except Exception as e:
                    logger.error(f"Error saving bot audio on disconnect: {e}")
        
        logger.info(
            f"Client disconnected - logged transcripts: {transcript_counter['user']} user / {transcript_counter['bot']} bot, "
            f"audio chunks: {audio_counter['user']} user / {audio_counter['bot']} bot"
        )
        await task.cancel()
    
    # Per-utterance audio recording handlers
    @audiobuffer.event_handler("on_user_turn_audio_data")
    async def on_user_turn_audio_data(buffer, audio, sample_rate, num_channels):
        """Handle each user (student) utterance - fires when student stops speaking."""
        if not submission_id or not assignment_id:
            logger.warning("Missing submission_id or assignment_id, skipping audio logging")
            return

        audio_counter["user"] += 1
        audio_num = audio_counter["user"]
        
        try:
            # Convert to WAV bytes
            wav_bytes = audio_to_wav(audio, sample_rate, num_channels)
            pending_user_audio_queue.append({"audio_num": audio_num, "wav_bytes": wav_bytes})
            logger.debug(f"Queued user audio (q={len(pending_user_audio_queue)}) audio #{audio_num}")
            
        except Exception as e:
            logger.error(f"Error handling user audio #{audio_num}: {e}")
    
    @audiobuffer.event_handler("on_bot_turn_audio_data")
    async def on_bot_turn_audio_data(buffer, audio, sample_rate, num_channels):
        """Handle each bot (teacher) utterance - fires when bot stops speaking."""
        # Check if this utterance was interrupted (set by UserStartedSpeakingFrame while bot was speaking)
        was_interrupted = bot_state["current_bot_interrupted"]
        
        if not submission_id or not assignment_id:
            logger.warning("Missing submission_id or assignment_id, skipping audio logging")
            return
        
        audio_counter["bot"] += 1
        audio_num = audio_counter["bot"]
        
        try:
            # Convert to WAV bytes
            wav_bytes = audio_to_wav(audio, sample_rate, num_channels)
            pending_bot_audio_queue.append(
                {"audio_num": audio_num, "wav_bytes": wav_bytes, "interrupted": was_interrupted}
            )
            logger.debug(
                f"Queued bot audio (q={len(pending_bot_audio_queue)}) audio #{audio_num}, interrupted={was_interrupted}"
            )
            
        except Exception as e:
            logger.error(f"Error handling bot audio #{audio_num}: {e}")


    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""

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
        ),
    )

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
