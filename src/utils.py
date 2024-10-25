from enum import StrEnum

from httpx import Timeout
from langchain_openai import ChatOpenAI
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
    llm = ChatOpenAI(
        model=llm_model, temperature=temperature, timeout=Timeout(60, connect=4)
    )
    return llm


async def consume_aiter(aiterator):
    return [x async for x in aiterator]


def auto_retry(f):
    decorator = retry(
        wait=wait_random_exponential(min=2, max=6),
        stop=stop_after_attempt(10),
    )
    return decorator(f)


def get_audio_from_voice_id(voice_id: str, input_csv_path: str = "data/11labs_available_tts_voices.reviewed.csv") -> str:
    voices_df = pd.read_csv(input_csv_path)
    return voices_df[voices_df["voice_id"] == voice_id]["preview_url"].values[0]