from enum import StrEnum

from httpx import Timeout
from langchain_openai import ChatOpenAI


class GPTModels(StrEnum):
    GPT_4o = "gpt-4o"
    GPT_4o_MINI = "gpt-4o-mini"
    GPT_4_TURBO_2024_04_09 = "gpt-4-turbo-2024-04-09"


def get_chat_llm(llm_model: GPTModels, temperature=0.0):
    llm = ChatOpenAI(
        model=llm_model, temperature=temperature, timeout=Timeout(60, connect=4)
    )
    return llm
