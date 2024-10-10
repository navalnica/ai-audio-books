from enum import StrEnum

import pandas as pd
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel

from src.config import logger
from src.prompts import CharacterVoicePropertiesPrompt
from src.utils import GPTModels, get_chat_llm


class Property(StrEnum):
    gender = "gender"
    age_group = "age_group"


class CharacterProperties(BaseModel):
    gender: str
    age_group: str

    def __hash__(self):
        return hash((self.gender, self.age_group))


class AllCharactersProperties(BaseModel):
    character2props: dict[str, CharacterProperties]


class CharacterPropertiesNullable(BaseModel):
    gender: str | None
    age_group: str | None

    def __hash__(self):
        return hash((self.gender, self.age_group))


class AllCharactersPropertiesNullable(BaseModel):
    character2props: dict[str, CharacterPropertiesNullable]


class SelectVoiceChainOutput(BaseModel):
    character2props: dict[str, CharacterPropertiesNullable]
    character2voice: dict[str, str]


class VoiceSelector:
    PROPERTY_VALUES = {
        Property.gender: {"male", "female"},
        Property.age_group: {"young", "middle_aged", "old"},
    }

    def __init__(self, csv_table_fp: str):
        self.df = self.read_data_table(csv_table_fp=csv_table_fp)

    def read_data_table(self, csv_table_fp: str):
        logger.info(f'reading voice data from: "{csv_table_fp}"')
        df = pd.read_csv(csv_table_fp)
        df["age"] = df["age"].str.replace(" ", "_").str.replace("-", "_")
        return df

    def get_available_properties_str(self, prop: Property):
        vals = self.PROPERTY_VALUES[prop]
        res = ", ".join(f'"{v}"' for v in vals)
        return res

    def _get_voices_single_props(
        self, character_props: CharacterPropertiesNullable, n_characters: int
    ):
        if n_characters <= 0:
            raise ValueError(n_characters)

        df_filtered = self.df
        if val := character_props.gender:
            df_filtered = df_filtered[df_filtered["gender"] == val]
        if val := character_props.age_group:
            df_filtered = df_filtered[df_filtered["age"] == val]

        voice_ids = df_filtered.sample(n_characters)["voice_id"].to_list()
        return voice_ids

    def get_voices(self, inputs: dict) -> dict:
        character_props: AllCharactersPropertiesNullable = inputs["charater_props"]

        # check for Nones.
        # TODO: for simplicity we raise error if LLM failed to select valid property value.
        # else, we would need to implement clever mapping to avoid overlapping between voices.
        for char, props in character_props.character2props.items():
            if props.age_group is None or props.gender is None:
                raise ValueError(props)

        prop2character = {}
        for character, props in character_props.character2props.items():
            prop2character.setdefault(props, set()).add(character)

        character2voice = {}
        for props, characters in prop2character.items():
            voice_ids = self._get_voices_single_props(
                character_props=props, n_characters=len(characters)
            )
            character2voice.update(zip(characters, voice_ids))

        return character2voice

    def _remove_hallucinations_single_character(
        self, character_props: CharacterProperties
    ):
        def _process_prop(prop: Property, value: str):
            if value not in self.PROPERTY_VALUES[prop]:
                logger.warning(
                    f'LLM selected non-available {prop} value: "{value}". defaulting to None'
                )
                return None
            return value

        return CharacterPropertiesNullable(
            gender=_process_prop(prop=Property.gender, value=character_props.gender),
            age_group=_process_prop(
                prop=Property.age_group, value=character_props.age_group
            ),
        )

    def remove_hallucinations(
        self, props: AllCharactersProperties
    ) -> AllCharactersPropertiesNullable:
        res = AllCharactersPropertiesNullable(
            character2props={
                k: self._remove_hallucinations_single_character(character_props=v)
                for k, v in props.character2props.items()
            }
        )
        return res

    def pack_results(self, inputs: dict):
        character_props: AllCharactersPropertiesNullable = inputs["charater_props"]
        character2voice: dict[str, str] = inputs["character2voice"]
        return SelectVoiceChainOutput(
            character2props=character_props.character2props,
            character2voice=character2voice,
        )

    def create_voice_mapping_chain(self, llm_model: GPTModels):
        llm = get_chat_llm(llm_model=llm_model, temperature=0.0)
        llm = llm.with_structured_output(AllCharactersProperties, method="json_mode")

        output_parser = PydanticOutputParser(pydantic_object=AllCharactersProperties)
        format_instructions = output_parser.get_format_instructions()

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    CharacterVoicePropertiesPrompt.SYSTEM
                ),
                HumanMessagePromptTemplate.from_template(
                    CharacterVoicePropertiesPrompt.USER
                ),
            ]
        )
        prompt = prompt.partial(
            **{
                "available_genders": self.get_available_properties_str(Property.gender),
                "available_age_groups": self.get_available_properties_str(
                    Property.age_group
                ),
                "format_instructions": format_instructions,
            }
        )

        chain = (
            RunnablePassthrough.assign(
                charater_props=prompt | llm | self.remove_hallucinations
            )
            | RunnablePassthrough.assign(character2voice=self.get_voices)
            | self.pack_results
        )
        return chain
