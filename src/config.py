import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s): %(message)s",
)
logger = logging.getLogger("audio-books")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ELEVENLABS_API_KEY = os.environ["ELEVEN_LABS_API_KEY"]

FILE_SIZE_MAX = 0.5  # in mb

OPENAI_MAX_PARALLEL = 5  # empirically set

# current limitation of available subscription.
# see: https://elevenlabs.io/docs/api-reference/text-to-speech#generation-and-concurrency-limits
ELEVENLABS_MAX_PARALLEL = 15

# VOICES_CSV_FP = "data/11labs_available_tts_voices.csv"
VOICES_CSV_FP = "data/11labs_available_tts_voices.reviewed.csv"

MAX_TEXT_LEN = 5000

DESCRIPTION = """\
# AI Audiobooks Generator

Create an audiobook from the input text automatically, using Gen-AI!

All you need to do - is to input the book text or select it from the provided Sample Inputs.

AI will do the rest:
- split text into characters
- assign each character a voice
- preprocess text to better convey emotions during Text-to-Speech
- (optionally) add sound effects to create immersive atmosphere
- generate audiobook using Text-to-Speech model
"""
