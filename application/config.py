# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- App Config ---
VIDEO_SOURCE = 0  # 0 = default webcam
SUMMARY_DIR = "summaries"
CLIP_DIR = "clips"
ALERTS_DIR = "alerts"

# Create directories
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(CLIP_DIR, exist_ok=True)
os.makedirs(ALERTS_DIR, exist_ok=True)

# --- Secrets ---
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- Gemini Config ---
if GEMINI_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-pro") # Using 2.5 Pro
else:
    GEMINI_MODEL = None