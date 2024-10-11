import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s): %(message)s",
)
logger = logging.getLogger("audio-books")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ELEVENLABS_API_KEY = os.environ["ELEVEN_LABS_API_KEY"]

FILE_SIZE_MAX = 0.5  # in mb

OPENAI_MAX_PARALLEL = 8  # empirically set
ELEVENLABS_MAX_PARALLEL = 15  # current limitation of available subscription

# VOICES_CSV_FP = "data/11labs_available_tts_voices.csv"
VOICES_CSV_FP = "data/11labs_available_tts_voices.reviewed.csv"
