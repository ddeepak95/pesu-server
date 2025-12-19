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
from pipecat.frames.frames import LLMRunFrame

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
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
from supabase_client import log_voice_message

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
    
    # Track utterance counts for unique filenames
    utterance_counter = {"user": 0, "bot": 0}
    
    # Track latest transcripts for pairing with audio
    latest_transcripts = {"user": "", "bot": ""}
    
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
    
    # Audio buffer for per-utterance recording
    audiobuffer = AudioBufferProcessor(
        num_channels=1,
        enable_turn_audio=True,  # Enable per-turn audio events
        user_continuous_stream=True,
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            audiobuffer,  # Audio recording (after output for both user and bot audio)
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],

    )


    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected - waiting for client_ready signal")

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info(f"Client ready - starting conversation")
        await rtvi.set_bot_ready()
        
        # Start audio recording
        if submission_id:
            await audiobuffer.start_recording()
            utterance_counter["user"] = 0
            utterance_counter["bot"] = 0
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
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected - logged {utterance_counter['user']} user and {utterance_counter['bot']} bot utterances")
        await task.cancel()
    
    # Per-utterance audio recording handlers
    @audiobuffer.event_handler("on_user_turn_audio_data")
    async def on_user_turn_audio_data(buffer, audio, sample_rate, num_channels):
        """Handle each user (student) utterance - fires when student stops speaking."""
        if not submission_id or not assignment_id:
            logger.warning("Missing submission_id or assignment_id, skipping audio logging")
            return
        
        utterance_counter["user"] += 1
        utterance_num = utterance_counter["user"]
        
        try:
            # Convert to WAV bytes
            wav_bytes = audio_to_wav(audio, sample_rate, num_channels)
            
            # Generate storage path
            path = generate_audio_path(
                submission_id, question_order, attempt_number, "user", utterance_num
            )
            
            # Upload to Firebase (run in thread to avoid blocking)
            audio_url = await asyncio.to_thread(upload_audio, wav_bytes, path)
            
            # Get transcript from context (latest user message)
            user_transcript = latest_transcripts.get("user", "")
            if not user_transcript and len(messages) >= 2:
                # Try to get from messages context
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        user_transcript = msg.get("content", "")
                        break
            
            # Log to Supabase
            await asyncio.to_thread(
                log_voice_message,
                submission_id,
                assignment_id,
                question_order,
                "student",
                user_transcript,
                audio_url,
                attempt_number
            )
            
            logger.info(f"Logged user utterance #{utterance_num}: {len(wav_bytes)} bytes, transcript: {user_transcript[:50]}...")
            
        except Exception as e:
            logger.error(f"Error logging user utterance #{utterance_num}: {e}")
    
    @audiobuffer.event_handler("on_bot_turn_audio_data")
    async def on_bot_turn_audio_data(buffer, audio, sample_rate, num_channels):
        """Handle each bot (teacher) utterance - fires when bot stops speaking."""
        if not submission_id or not assignment_id:
            logger.warning("Missing submission_id or assignment_id, skipping audio logging")
            return
        
        utterance_counter["bot"] += 1
        utterance_num = utterance_counter["bot"]
        
        try:
            # Convert to WAV bytes
            wav_bytes = audio_to_wav(audio, sample_rate, num_channels)
            
            # Generate storage path
            path = generate_audio_path(
                submission_id, question_order, attempt_number, "bot", utterance_num
            )
            
            # Upload to Firebase
            audio_url = await asyncio.to_thread(upload_audio, wav_bytes, path)
            
            # Get transcript from context (latest assistant message)
            bot_transcript = latest_transcripts.get("bot", "")
            if not bot_transcript and len(messages) >= 1:
                # Try to get from messages context
                for msg in reversed(messages):
                    if msg.get("role") == "assistant":
                        bot_transcript = msg.get("content", "")
                        break
            
            # Log to Supabase
            await asyncio.to_thread(
                log_voice_message,
                submission_id,
                assignment_id,
                question_order,
                "assistant",
                bot_transcript,
                audio_url,
                attempt_number
            )
            
            logger.info(f"Logged bot utterance #{utterance_num}: {len(wav_bytes)} bytes, transcript: {bot_transcript[:50]}...")
            
        except Exception as e:
            logger.error(f"Error logging bot utterance #{utterance_num}: {e}")



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
