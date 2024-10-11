import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s): %(message)s",
)
logger = logging.getLogger("audio-books")


ELEVENLABS_API_KEY = os.environ["ELEVEN_LABS_API_KEY"]
AI_ML_API_KEY = os.environ["AIML_API_KEY"]

FILE_SIZE_MAX = 0.5  # in mb

ELEVENLABS_MAX_PARALLEL = 15  # current limitation of available subscription
