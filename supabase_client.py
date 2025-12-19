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
    audio_url: str,
    attempt_number: int,
    duration_seconds: Optional[float] = None
) -> dict:
    """Log a voice message to the voice_messages table in Supabase.
    
    Args:
        submission_id: ID of the submission
        assignment_id: ID of the assignment
        question_order: Order of the question (0-indexed)
        role: "student" or "assistant"
        content: Transcript text of the utterance
        audio_url: URL to the audio file in Firebase Storage
        attempt_number: Attempt number for this question
        duration_seconds: Optional duration of the audio in seconds
    
    Returns:
        The inserted record data
    
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
        "audio_file_url": audio_url,
        "attempt_number": attempt_number,
    }
    
    if duration_seconds is not None:
        data["duration_seconds"] = duration_seconds
    
    result = supabase.table("voice_messages").insert(data).execute()
    
    logger.info(f"Logged voice message: role={role}, question={question_order}, attempt={attempt_number}")
    
    return result.data[0] if result.data else {}
