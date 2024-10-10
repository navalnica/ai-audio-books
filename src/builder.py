import asyncio
import os
import re
from pathlib import Path
from uuid import uuid4

import pandas as pd
from langchain_community.callbacks import get_openai_callback

from src.lc_callbacks import LCMessageLoggerAsync
from src.select_voice_chain import AllCharactersProperties, create_voice_mapping_chain
from src.text_split_chain import SplitTextOutput, create_split_text_chain
from src.tts import tts_astream
from src.utils import GPTModels, consume_aiter

VOICES = pd.read_csv("data/11labs_tts_voices.csv").query("language == 'en'")


class AudiobookBuilder:
    def __init__(self) -> None:
        self._default_narrator_voice = "ALY2WaJPY0oBJlqpQbfW"

    async def split_text(self, text: str) -> SplitTextOutput:
        chain = create_split_text_chain(llm_model=GPTModels.GPT_4o)
        with get_openai_callback() as cb:
            chain_out = chain.ainvoke(
                {"text": text}, config={"callbacks": [LCMessageLoggerAsync()]}
            )
        return chain_out

    async def map_character_to_voice_properties(
        self, text_split: SplitTextOutput
    ) -> AllCharactersProperties:
        chain = create_voice_mapping_chain(llm_model=GPTModels.GPT_4_TURBO_2024_04_09)
        with get_openai_callback() as cb:
            chain_out = await chain.ainvoke(
                {
                    "text": text_split.text_annotated,
                    "characters": text_split.characters,
                },
                config={"callbacks": [LCMessageLoggerAsync()]},
            )
        return chain_out

    def map_characters_to_voices(
        self, character_voice_properties: AllCharactersProperties
    ):
        # TODO
        raise NotImplementedError

    # NOTE: old function
    # def map_characters_to_voices(
    #     self, character_to_gender: dict[str, str]
    # ) -> dict[str, str]:
    #     character_to_voice = {"narrator": self._default_narrator_voice}

    #     # Damy vperyod!
    #     f_characters = [
    #         character
    #         for character, gender in character_to_gender.items()
    #         if gender.strip().lower() == "f"
    #     ]
    #     if f_characters:
    #         f_voices = (
    #             VOICES.query("gender == 'female'").iloc[: len(f_characters)].copy()
    #         )
    #         f_voices["character"] = f_characters
    #         character_to_voice |= f_voices.set_index("character")["voice_id"].to_dict()

    #     m_characters = [
    #         character
    #         for character, gender in character_to_gender.items()
    #         if gender.strip().lower() == "m"
    #     ]
    #     if m_characters:
    #         m_voices = VOICES.query("gender == 'male'").iloc[: len(m_characters)].copy()
    #         m_voices["character"] = m_characters
    #         character_to_voice |= m_voices.set_index("character")["voice_id"].to_dict()

    #     return character_to_voice

    async def generate_audio(
        self,
        annotated_text: str,
        character_to_voice: dict[str, str],
    ) -> Path:
        tasks = []
        current_character = "narrator"
        for line in annotated_text.splitlines():
            cleaned_line = line.strip().lower()
            if not cleaned_line:
                continue
            try:
                current_character = re.findall(r"\[[\w\s]+\]", cleaned_line)[0][1:-1]
            except:
                pass
            voice_id = character_to_voice[current_character]
            character_text = cleaned_line[cleaned_line.rfind("]") + 1 :].lstrip()
            tasks.append(tts_astream(voice_id=voice_id, text=character_text))

        results = await asyncio.gather(*(consume_aiter(t) for t in tasks))
        save_dir = Path("data") / "books"
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f"{uuid4()}.wav"
        with open(save_path, "wb") as ab:
            for result in results:
                for chunk in result:
                    ab.write(chunk)
        return save_path

    async def run(self, text: str):
        text_split = await self.split_text(text)
        character_voice_properties = await self.map_character_to_voice_properties(
            text_split=text_split
        )
        character_to_voice = self.map_characters_to_voices(
            character_voice_properties=character_voice_properties
        )
        out_path = self.generate_audio(annotated_text, character_to_voice)
        return out_path
