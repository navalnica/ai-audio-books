from enum import StrEnum

from httpx import Timeout
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from prompts import SplitTextPrompt


class GPTModels(StrEnum):
    GPT_4_TURBO_2024_04_09 = "gpt-4-turbo-2024-04-09"
    GPT_4o_MINI = "gpt-4o-mini"


class TextPart(BaseModel):
    character: str
    text: str


class SplitTextOutput(BaseModel):
    characters: list[str]
    parts: list[TextPart]

    def to_pretty_text(self):
        lines = []
        lines.append(f"characters: {self.characters}")
        lines.extend(f"[{part.character}] {part.text}" for part in self.parts)
        res = "\n".join(lines)
        return res


def create_split_text_chain(llm_model: GPTModels):
    llm = ChatOpenAI(model=llm_model, temperature=0.0, timeout=Timeout(60, connect=4))
    llm = llm.with_structured_output(SplitTextOutput)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SplitTextPrompt.SYSTEM),
            HumanMessagePromptTemplate.from_template(SplitTextPrompt.USER),
        ]
    )

    chain = prompt | llm
    return chain
