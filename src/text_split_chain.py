import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel

from src.prompts import SplitTextPromptV1, SplitTextPromptV2
from src.utils import GPTModels, get_chat_llm


class CharacterPhrase(BaseModel):
    character: str
    text: str


class SplitTextOutput(BaseModel):
    text_raw: str
    text_annotated: str
    _phrases: list[CharacterPhrase]
    _characters: list[str]

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
        self._phrases = self._parse_phrases_from_xml_tags(self.text_annotated)
        self._characters = list(set(phrase.character for phrase in self.phrases))
        # TODO: can apply post-processing to merge same adjacent xml tags

    @property
    def phrases(self) -> list[CharacterPhrase]:
        return self._phrases

    @property
    def characters(self) -> list[str]:
        return self._characters

    def to_pretty_text(self):
        lines = []
        lines.append(f"characters: {self.characters}")
        lines.append("-" * 20)
        lines.extend(f"[{phrase.character}] {phrase.text}" for phrase in self.phrases)
        res = "\n".join(lines)
        return res


def create_split_text_chain(llm_model: GPTModels):
    llm = get_chat_llm(llm_model=llm_model, temperature=0.0)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SplitTextPromptV2.SYSTEM),
            HumanMessagePromptTemplate.from_template(SplitTextPromptV2.USER),
        ]
    )

    chain = RunnablePassthrough.assign(
        text_annotated=prompt | llm | StrOutputParser()
    ) | (
        lambda inputs: SplitTextOutput(
            text_raw=inputs["text"], text_annotated=inputs["text_annotated"]
        )
    )
    return chain


###### old code ######


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


class SplitTextOutputOld(BaseModel):
    characters: list[str]
    parts: list[CharacterPhrase]

    def to_character_annotated_text(self):
        return CharacterAnnotatedText(phrases=self.parts)


def create_split_text_chain_old(llm_model: GPTModels):
    llm = get_chat_llm(llm_model=llm_model, temperature=0.0)
    llm = llm.with_structured_output(SplitTextOutputOld, method="json_mode")

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SplitTextPromptV1.SYSTEM),
            HumanMessagePromptTemplate.from_template(SplitTextPromptV1.USER),
        ]
    )

    chain = prompt | llm
    return chain


## end of old code ##
