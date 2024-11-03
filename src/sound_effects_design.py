import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel

from src import prompts
from src.utils import GPTModels, get_chat_llm


class SoundEffectDescription(BaseModel):
    prompt: str
    text_between_tags: str
    # indices relative to LLM response
    ix_start_llm_response: int
    ix_end_llm_response: int
    # indices relative to original text passed to LLM
    ix_start_orig_text: int
    ix_end_orig_text: int
    # NOTE: start_sec and duration_sec fields
    # are going to be filled once TTS audio is generated
    start_sec: float = -1.0
    duration_sec: float = -1.0


class SoundEffectsDesignOutput(BaseModel):
    text_raw: str
    text_annotated: str
    _sound_effects_descriptions: list[SoundEffectDescription]

    @staticmethod
    def _parse_effects_xml_tags(text) -> list[SoundEffectDescription]:
        """
        we rely on LLM to format response correctly.
        and currently don't try to fix possible errors.
        """
        # TODO: allow to open-close tags
        # <effect prompt=\"(.*?)\" duration=\"(.*)\"/>

        pattern = re.compile(r"<effect prompt=\"(.*?)\">(.*?)</effect>")
        all_matches = list(pattern.finditer(text))

        sound_effects_descriptions = []

        rm_chars_running_total = 0
        for m in all_matches:
            mstart, mend = m.span()
            prompt = m.group(1)
            text_between_tags = m.group(2)

            ix_start_orig = mstart - rm_chars_running_total
            ix_end_orig = ix_start_orig + len(text_between_tags)

            sound_effects_descriptions.append(
                SoundEffectDescription(
                    prompt=prompt,
                    text_between_tags=text_between_tags,
                    ix_start_llm_response=mstart,
                    ix_end_llm_response=mend,
                    ix_start_orig_text=ix_start_orig,
                    ix_end_orig_text=ix_end_orig,
                )
            )

            mlen = mend - mstart
            rm_chars_running_total += mlen - len(text_between_tags)

        return sound_effects_descriptions

    def __init__(self, **data):
        super().__init__(**data)
        self._sound_effects_descriptions = self._parse_effects_xml_tags(self.text_annotated)

    @property
    def sound_effects_descriptions(self) -> list[SoundEffectDescription]:
        return self._sound_effects_descriptions


def create_sound_effects_design_chain(llm_model: GPTModels):
    llm = get_chat_llm(llm_model=llm_model, temperature=0.0)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(prompts.SoundEffectsPrompt.SYSTEM),
            HumanMessagePromptTemplate.from_template(prompts.SoundEffectsPrompt.USER),
        ]
    )

    chain = RunnablePassthrough.assign(text_annotated=prompt | llm | StrOutputParser()) | (
        lambda inputs: SoundEffectsDesignOutput(
            text_raw=inputs["text"], text_annotated=inputs["text_annotated"]
        )
    )
    return chain
