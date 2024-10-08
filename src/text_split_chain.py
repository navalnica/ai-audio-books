import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic import BaseModel

from src.prompts import SplitTextPromptV1, SplitTextPromptV2
from src.utils import GPTModels, get_chat_llm


class CharacterPhrase(BaseModel):
    character: str
    text: str


class CharacterAnnotatedText(BaseModel):
    phrases: list[CharacterPhrase]
    _characters: list[str]

    def __init__(self, **data):
        super().__init__(**data)
        self._characters = list(set(phrase.character for phrase in self.phrases))

    @property
    def characters(self):
        return self._characters

    def to_pretty_text(self):
        lines = []
        lines.append(f"characters: {self.characters}")
        lines.append("-" * 20)
        lines.extend(f"[{phrase.character}] {phrase.text}" for phrase in self.phrases)
        res = "\n".join(lines)
        return res


class SplitTextOutputV1(BaseModel):
    characters: list[str]
    parts: list[CharacterPhrase]

    def to_character_annotated_text(self):
        return CharacterAnnotatedText(phrases=self.parts)


def create_split_text_chain_v1(llm_model: GPTModels):
    llm = get_chat_llm(llm_model=llm_model, temperature=0.0)
    llm = llm.with_structured_output(SplitTextOutputV1)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SplitTextPromptV1.SYSTEM),
            HumanMessagePromptTemplate.from_template(SplitTextPromptV1.USER),
        ]
    )

    chain = prompt | llm
    return chain


class SplitTextOutputV2(BaseModel):
    text_raw: str
    _phrases: list[CharacterPhrase]

    @staticmethod
    def _parse_phrases_from_xml_tags(text):
        """
        we rely on LLM to format response correctly.
        so we don't check that opening xml tags match closing ones
        """
        pattern = re.compile(r"(?:<([^<>]+)>)(.*?)(?:</\1>)")
        res = pattern.findall(text)
        res = [CharacterPhrase(character=x[0], text=x[1]) for x in res]
        return res

    def __init__(self, **data):
        super().__init__(**data)
        self._phrases = self._parse_phrases_from_xml_tags(self.text_raw)

    @property
    def phrases(self):
        return self._phrases

    def to_character_annotated_text(self):
        return CharacterAnnotatedText(phrases=self.phrases)


def create_split_text_chain_v2(llm_model: GPTModels):
    llm = get_chat_llm(llm_model=llm_model, temperature=0.0)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SplitTextPromptV2.SYSTEM),
            HumanMessagePromptTemplate.from_template(SplitTextPromptV2.USER),
        ]
    )

    chain = prompt | llm | StrOutputParser() | (lambda x: SplitTextOutputV2(text_raw=x))
    return chain
