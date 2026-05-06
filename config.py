import os
import json

from dotenv import load_dotenv


load_dotenv()


def _json_env(name, default=None):
    raw_value = os.getenv(name)
    if not raw_value:
        return default if default is not None else {}
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        raise ValueError(f"{name} must contain valid JSON")


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev")
    TEMPLATES_AUTO_RELOAD = os.getenv("TEMPLATES_AUTO_RELOAD", "true").lower() == "true"
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(8 * 1024 * 1024)))

    MODEL_PATH = os.getenv("MODEL_PATH", "best_swin_nocurr_seed42.pth")
    IMG_SIZE = (224, 224)
    LABELS = ["Healthy", "Patient"]

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-09-2025")

    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "meta.llama3-70b-instruct-v1:0")

    REMOTE_LLM_URL = os.getenv("REMOTE_LLM_URL", "http://195.251.252.25:5000/ask")
    REMOTE_LLM_TIMEOUT = int(os.getenv("REMOTE_LLM_TIMEOUT", "120"))

    WAY_TO_SCHOOL_FIREBASE_CONFIG = _json_env("WAY_TO_SCHOOL_FIREBASE_CONFIG")
    WALKFREE_FIREBASE_CONFIG = _json_env("WALKFREE_FIREBASE_CONFIG")
