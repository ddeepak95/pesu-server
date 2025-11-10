#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Quickstart Example.

The example runs a simple voice AI bot that you can connect to using your
browser and speak with it. You can also deploy this bot to Pipecat Cloud.

Required AI services:
- Deepgram (Speech-to-Text)
- OpenAI (LLM)
- Cartesia (Text-to-Speech)

Run the bot using::

    uv run bot.py
"""

import os

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
logger.info("✅ All components loaded successfully!")

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

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
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
        logger.info(f"Client disconnected")
        await task.cancel()



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
