import shutil
import datetime
import wave
from enum import StrEnum
from pathlib import Path

from httpx import Timeout
from langchain_openai import ChatOpenAI
from pydub import AudioSegment
from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.config import logger


class GPTModels(StrEnum):
    GPT_4o = "gpt-4o"
    GPT_4o_MINI = "gpt-4o-mini"
    GPT_4_TURBO_2024_04_09 = "gpt-4-turbo-2024-04-09"


def get_chat_llm(llm_model: GPTModels, temperature=0.0):
    llm = ChatOpenAI(model=llm_model, temperature=temperature, timeout=Timeout(60, connect=4))
    return llm


def rm_dir_conditional(dp: str, to_remove=True):
    if not to_remove:
        return
    logger.info(f'removing dir: "{dp}"')
    try:
        shutil.rmtree(dp)
    except Exception:
        logger.exception(f'failed to remove dir')


def get_utc_now_str():
    now = datetime.datetime.now(tz=datetime.UTC)
    now_str = now.strftime('%Y%m%d-%H%M%S')
    return now_str


async def consume_aiter(aiterator):
    return [x async for x in aiterator]


def auto_retry(f):
    decorator = retry(
        wait=wait_random_exponential(min=3, max=10),
        stop=stop_after_attempt(20),
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


def normalize_audio(audio_segment: AudioSegment, target_dBFS: float = -20.0) -> AudioSegment:
    """Normalize an audio segment to the target dBFS level."""

    # TODO: does it work as expected?

    delta = target_dBFS - audio_segment.dBFS
    res = audio_segment.apply_gain(delta)
    return res


# TODO: outdated code
def add_overlay_for_audio(
    audio1_fp: str,
    audio2_fp: str,
    out_fp: str | None = None,
    cycling_effect: bool = False,
    decrease_effect_volume: int = 0,
) -> str:
    try:
        main_audio = AudioSegment.from_file(audio1_fp)
        effect_audio = AudioSegment.from_file(audio2_fp)
    except Exception as e:
        raise RuntimeError(f"Error loading audio files: {e}")

    if cycling_effect:
        while len(effect_audio) < len(main_audio):
            effect_audio += effect_audio

    effect_audio = effect_audio[: len(main_audio)]

    if decrease_effect_volume > 0:
        effect_audio = effect_audio - decrease_effect_volume
    combined_audio = main_audio.overlay(effect_audio)

    if out_fp is None:
        out_fp = f"{Path(audio1_fp).stem}_{Path(audio2_fp).stem}.wav"
    combined_audio.export(out_fp, format="wav")

    return out_fp
