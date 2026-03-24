import asyncio

from loguru import logger
from pipecat.frames.frames import EndTaskFrame, LLMMessagesAppendFrame
from pipecat.processors.frame_processor import FrameDirection


class IdleHandler:
    """Two-stage user-idle escalation.

    Stage 1 (first timeout): gently ask if the student is still there.
    Stage 2 (second timeout): say goodbye and end the pipeline.
    """

    def __init__(self):
        self._retry_count = 0

    def reset(self):
        self._retry_count = 0

    async def handle_idle(self, aggregator):
        self._retry_count += 1

        if self._retry_count == 1:
            logger.info("User idle (30s) - checking in")
            message = {
                "role": "system",
                "content": (
                    "The student has been quiet for a while. "
                    "Politely and briefly ask if they are still there, "
                    "in the conversation language."
                ),
            }
            await aggregator.push_frame(
                LLMMessagesAppendFrame([message], run_llm=True)
            )
        else:
            logger.info("User idle (60s) - ending conversation")
            message = {
                "role": "system",
                "content": (
                    "The student is still inactive. Say a brief, polite goodbye "
                    "in the conversation language and let them know the session "
                    "is ending due to inactivity."
                ),
            }
            await aggregator.push_frame(
                LLMMessagesAppendFrame([message], run_llm=True)
            )
            await asyncio.sleep(8)
            await aggregator.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
