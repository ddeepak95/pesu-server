"""Supabase client utility for logging voice messages.

This module provides functions to log voice messages to the Supabase database,
storing transcript text along with audio file URLs.
"""

import os
from typing import Optional

from loguru import logger
from supabase import create_client, Client


_supabase_client: Optional[Client] = None


def get_supabase() -> Client:
    """Get or create a Supabase client (singleton pattern).
    
    Returns:
        Supabase client instance
    
    Raises:
        ValueError: If required environment variables are not set
    """
    global _supabase_client
    
    if _supabase_client is not None:
        return _supabase_client
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    if not supabase_url:
        raise ValueError("SUPABASE_URL environment variable is not set")
    if not supabase_key:
        raise ValueError("SUPABASE_ANON_KEY environment variable is not set")
    
    _supabase_client = create_client(supabase_url, supabase_key)
    logger.info("Supabase client initialized")
    
    return _supabase_client


def log_voice_message(
    submission_id: str,
    assignment_id: str,
    question_order: int,
    role: str,
    content: str,
    attempt_number: int,
    audio_url: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    interrupted: bool = False,
    spoken_at: Optional[str] = None,
    generated_content: Optional[str] = None,
) -> dict:
    """Log a voice message to the voice_messages table in Supabase.
    
    Args:
        submission_id: ID of the submission
        assignment_id: ID of the assignment
        question_order: Order of the question (0-indexed)
        role: "student" or "assistant"
        content: Transcript text of the utterance
        attempt_number: Attempt number for this question
        audio_url: Optional URL to the audio file in Firebase Storage (can be added later via update)
        duration_seconds: Optional duration of the audio in seconds
        interrupted: True if the utterance was interrupted (audio is partial, transcript is full)
        spoken_at: ISO timestamp when the utterance was spoken (for accurate timing analysis)
    
    Returns:
        The inserted record data (includes 'id' for later updates)
    
    Raises:
        Exception: If the insert operation fails
    """
    supabase = get_supabase()
    
    data = {
        "submission_id": submission_id,
        "assignment_id": assignment_id,
        "question_order": question_order,
        "role": role,
        "content": content,
        "attempt_number": attempt_number,
        "interrupted": interrupted,
    }
    
    # Only include audio_url if provided
    if audio_url is not None:
        data["audio_file_url"] = audio_url
    
    if duration_seconds is not None:
        data["duration_seconds"] = duration_seconds
    
    if spoken_at is not None:
        data["spoken_at"] = spoken_at

    if generated_content is not None:
        data["generated_content"] = generated_content

    try:
        result = supabase.table("voice_messages").insert(data).execute()
    except Exception as e:
        # If the DB schema was not migrated yet (or PostgREST schema cache not refreshed),
        # inserting generated_content will fail with PGRST204. Fallback to inserting without it
        # so we don't lose the assistant message entirely.
        err = e.args[0] if getattr(e, "args", None) else None
        message = ""
        code = None
        if isinstance(err, dict):
            message = str(err.get("message", ""))
            code = err.get("code")
        else:
            message = str(e)

        if code == "PGRST204" and "generated_content" in message:
            logger.warning(
                "voice_messages.generated_content not found in schema cache; "
                "inserting without generated_content (run migration + reload schema)."
            )
            data.pop("generated_content", None)
            result = supabase.table("voice_messages").insert(data).execute()
        else:
            raise
    
    logger.info(f"Logged voice message: role={role}, question={question_order}, attempt={attempt_number}, has_audio={audio_url is not None}, interrupted={interrupted}")
    
    return result.data[0] if result.data else {}


def update_voice_message_audio(
    message_id: str,
    audio_url: str,
    duration_seconds: Optional[float] = None,
    interrupted: Optional[bool] = None,
) -> dict:
    """Update an existing voice message with audio URL.
    
    This is used when transcript is logged first, then audio is added
    after the turn audio is processed and uploaded.
    
    Args:
        message_id: UUID of the voice message record to update
        audio_url: URL to the audio file in Firebase Storage
        duration_seconds: Optional duration of the audio in seconds
    
    Returns:
        The updated record data
    
    Raises:
        Exception: If the update operation fails
    """
    supabase = get_supabase()
    
    data = {"audio_file_url": audio_url}
    
    if duration_seconds is not None:
        data["duration_seconds"] = duration_seconds

    if interrupted is not None:
        data["interrupted"] = interrupted
    
    result = supabase.table("voice_messages").update(data).eq("id", message_id).execute()
    
    logger.info(f"Updated voice message {message_id} with audio URL")
    
    return result.data[0] if result.data else {}
