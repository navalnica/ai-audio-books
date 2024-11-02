import wave
from enum import StrEnum
from pathlib import Path

from httpx import Timeout
from langchain_openai import ChatOpenAI
from pydub import AudioSegment
from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.config import logger

from openai import api_key
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)



class GPTModels(StrEnum):
    GPT_4o = "gpt-4o"
    GPT_4o_MINI = "gpt-4o-mini"
    GPT_4_TURBO_2024_04_09 = "gpt-4-turbo-2024-04-09"


def get_chat_llm(llm_model: GPTModels, temperature=0.0):
    llm = ChatOpenAI(model=llm_model, temperature=temperature, timeout=Timeout(60, connect=4))
    return llm


async def consume_aiter(aiterator):
    return [x async for x in aiterator]


def auto_retry(f):
    decorator = retry(
        wait=wait_random_exponential(min=2, max=6),
        stop=stop_after_attempt(10),
    )
    return decorator(f)


def write_bytes(data: bytes, fp: str):
    logger.info(f'saving to: "{fp}"')
    with open(fp, "wb") as fout:
        fout.write(data)


def write_raw_pcm_to_file(data: bytes, fp: str, n_channels: int, bytes_depth: int, sampling_rate):
    logger.info(f'saving to: "{fp}"')
    with wave.open(fp, "wb") as f:
        f.setnchannels(n_channels)
        f.setsampwidth(bytes_depth)
        f.setframerate(sampling_rate)
        f.writeframes(data)


def get_audio_duration(filepath: str) -> float:
    """
    Returns the duration of the audio file in seconds.

    :param filepath: Path to the audio file.
    :return: Duration of the audio file in seconds.
    """
    audio = AudioSegment.from_file(filepath)
    # Convert milliseconds to seconds
    duration_in_seconds = len(audio) / 1000
    return round(duration_in_seconds, 1)


def add_overlay_for_audio(
    main_audio_filename: str,
    sound_effect_filename: str,
    output_filename: str | None = None,
    cycling_effect: bool = True,
    decrease_effect_volume: int = 0,
) -> str:
    try:
        main_audio = AudioSegment.from_file(main_audio_filename)
        effect_audio = AudioSegment.from_file(sound_effect_filename)
    except Exception as e:
        raise RuntimeError(f"Error loading audio files: {e}")

    if cycling_effect:
        while len(effect_audio) < len(main_audio):
            effect_audio += effect_audio

    effect_audio = effect_audio[: len(main_audio)]

    if decrease_effect_volume > 0:
        effect_audio = effect_audio - decrease_effect_volume
    combined_audio = main_audio.overlay(effect_audio)

    if output_filename is None:
        output_filename = f"{Path(main_audio_filename).stem}_{Path(sound_effect_filename).stem}.wav"
    combined_audio.export(output_filename, format="wav")
    return output_filename

def get_audio_from_voice_id(voice_id: str, input_csv_path: str = "data/11labs_available_tts_voices.reviewed.csv") -> str:
    voices_df = pd.read_csv(input_csv_path)
    return voices_df[voices_df["voice_id"] == voice_id]["preview_url"].values[0]

