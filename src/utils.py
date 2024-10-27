import wave
from enum import StrEnum

from httpx import Timeout
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.config import logger


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


def write_bytes(data: bytes, fp: str):
    logger.info(f'saving to: "{fp}"')
    with open(fp, "wb") as fout:
        fout.write(data)


def write_raw_pcm_to_file(
    data: bytes, fp: str, n_channels: int, bytes_depth: int, sampling_rate
):
    logger.info(f'saving to: "{fp}"')
    with wave.open(fp, "wb") as f:
        f.setnchannels(n_channels)
        f.setsampwidth(bytes_depth)
        f.setframerate(sampling_rate)
        f.writeframes(data)
