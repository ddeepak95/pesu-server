"""Firebase Storage utility for uploading audio files.

This module provides functions to upload audio recordings to Firebase Storage
and retrieve their public URLs.
"""

import json
import os
import base64
import firebase_admin
from firebase_admin import credentials, storage
from loguru import logger


def init_firebase():
    """Initialize Firebase Admin SDK (singleton pattern).
    
    Uses either FIREBASE_SERVICE_ACCOUNT_KEY (JSON string) or 
    FIREBASE_SERVICE_ACCOUNT_PATH (file path) for authentication.
    """
    if firebase_admin._apps:
        return  # Already initialized
    
    # cred_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY")
    # cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
    base64_cred = os.getenv("FIREBASE_SERVICE_ACCOUNT_BASE64")
    bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET")
    
    if not bucket_name:
        raise ValueError("FIREBASE_STORAGE_BUCKET environment variable is not set")
    
    if base64_cred:
        sa_dict = json.loads(base64.b64decode(base64_cred).decode('utf-8'))
        try:
            cred = credentials.Certificate(sa_dict)
            logger.info(f"Firebase Admin initialized with credentials from base64")
        except Exception as e:
            raise ValueError(f"Failed to initialize Firebase Admin: {e}")
    
    firebase_admin.initialize_app(cred, {
        'storageBucket': bucket_name
    })


def upload_audio(audio_bytes: bytes, path: str, content_type: str = 'audio/wav') -> str:
    """Upload audio bytes to Firebase Storage and return public URL.
    
    Args:
        audio_bytes: Raw audio data as bytes
        path: Storage path (e.g., "voice-recordings/sub123/0/1/user-1.wav")
        content_type: MIME type of the audio (default: audio/wav)
    
    Returns:
        Public URL of the uploaded file
    """
    init_firebase()
    
    bucket = storage.bucket()
    blob = bucket.blob(path)
    
    # Upload the audio data
    blob.upload_from_string(audio_bytes, content_type=content_type)
    
    # Make the file publicly accessible
    blob.make_public()
    
    public_url = blob.public_url
    logger.info(f"Uploaded audio to {path}, URL: {public_url}")
    
    return public_url


def generate_audio_path(
    submission_id: str,
    question_order: int,
    attempt_number: int,
    role: str,
    utterance_num: int,
    extension: str = "wav"
) -> str:
    """Generate a unique path for audio file storage.
    
    Args:
        submission_id: ID of the submission
        question_order: Order of the question (0-indexed)
        attempt_number: Attempt number for this question
        role: "user" or "bot"
        utterance_num: Sequential number of this utterance
        extension: File extension (default: wav)
    
    Returns:
        Storage path like "voice-recordings/{submission_id}/{question}/{attempt}/{role}-{n}.wav"
    """
    # Sanitize submission_id to be safe for file paths
    safe_submission_id = submission_id.replace("/", "_").replace("\\", "_")
    
    return f"voice-recordings/{safe_submission_id}/{question_order}/{attempt_number}/{role}-{utterance_num}.{extension}"
