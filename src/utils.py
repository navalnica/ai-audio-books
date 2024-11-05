import datetime
import json
import re
import shutil
import typing as t
import wave
from enum import StrEnum
from pathlib import Path

import pandas as pd
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
    llm = ChatOpenAI(
        model=llm_model,
        temperature=temperature,
        timeout=Timeout(60, connect=4),
    )
    return llm


def write_txt(txt: str, fp: str):
    with open(fp, 'w', encoding='utf-8') as fout:
        fout.write(txt)


def write_json(data, fp: str, indent=2):
    with open(fp, 'w', encoding='utf-8') as fout:
        json.dump(data, fout, indent=indent, ensure_ascii=False)


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


def write_chunked_bytes(data: t.Iterable[bytes], fp: str):
    logger.info(f'saving to: "{fp}"')
    with open(fp, "wb") as fout:
        for chunk in data:
            if chunk:
                fout.write(chunk)


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


# def add_overlay_for_audio(
#     audio1_fp: str,
#     audio2_fp: str,
#     out_fp: str | None = None,
#     cycling_effect: bool = False,
#     decrease_effect_volume: int = 0,
# ):
# NOTE: deprecated
#     try:
#         main_audio = AudioSegment.from_file(audio1_fp)
#         effect_audio = AudioSegment.from_file(audio2_fp)
#     except Exception as e:
#         raise RuntimeError(f"Error loading audio files: {e}")

#     if cycling_effect:
#         while len(effect_audio) < len(main_audio):
#             effect_audio += effect_audio

#     effect_audio = effect_audio[: len(main_audio)]

#     if decrease_effect_volume > 0:
#         effect_audio = effect_audio - decrease_effect_volume
#     combined_audio = main_audio.overlay(effect_audio)

#     if out_fp:
#         logger.info(f'saving overlayed audio to: "{out_fp}"')
#         combined_audio.export(out_fp, format="wav")


def overlay_multiple_audio(
    main_audio_fp: str,
    audios_to_overlay_fps: list[str],
    starts_sec: list[float],  # list of start positions, in seconds
    out_fp: str,
):
    main_audio = AudioSegment.from_file(main_audio_fp)
    for fp, cur_start_sec in zip(audios_to_overlay_fps, starts_sec):
        audio_to_overlay = AudioSegment.from_file(fp)
        # NOTE: quote from the documentation:
        # "The result is always the same length as this AudioSegment"
        # reference: https://github.com/jiaaro/pydub/blob/master/API.markdown#audiosegmentoverlay
        # NOTE: `position` params is offset time in milliseconds
        start_ms = int(cur_start_sec * 1000)
        main_audio = main_audio.overlay(audio_to_overlay, position=start_ms)

    logger.info(f'saving overlayed audio to: "{out_fp}"')
    main_audio.export(out_fp, format='wav')


def get_audio_from_voice_id(
    voice_id: str, input_csv_path: str = "data/11labs_available_tts_voices.reviewed.csv"
) -> str:
    voices_df = pd.read_csv(input_csv_path)
    return voices_df[voices_df["voice_id"] == voice_id]["preview_url"].values[0]


def get_character_color(character: str) -> str:
    if not character or character == "Unassigned":
        return "#808080"
    colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#96CEB4",
        "#FFEEAD",
        "#D4A5A5",
        "#9B59B6",
        "#3498DB",
    ]
    hash_val = sum(ord(c) for c in character)
    return colors[hash_val % len(colors)]


def replace_labels(text):
    return re.sub(r'\bc(\d+)\b', r'Character\1', text)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return f"{int(hex_color[0:2], 16)},{int(hex_color[2:4], 16)},{int(hex_color[4:6], 16)}"
