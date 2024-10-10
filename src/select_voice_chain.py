from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from enum import StrEnum

from src.prompts import CharacterVoicePropertiesPrompt
from src.utils import GPTModels, get_chat_llm


class Property(StrEnum):
    gender = "gender"
    age_group = "age_group"


PROPERTY_VALUES = {
    Property.gender: {"male", "female"},
    Property.age_group: {"young", "middle_aged", "old"},
}


def get_available_properties_str(prop: Property):
    vals = PROPERTY_VALUES[prop]
    res = ", ".join(f'"{v}"' for v in vals)
    return res


class CharacterProperties(BaseModel):
    gender: str
    age_group: str


class AllCharactersProperties(BaseModel):
    character2props: dict[str, CharacterProperties]


def create_voice_mapping_chain(llm_model: GPTModels):
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
            "available_genders": get_available_properties_str(Property.gender),
            "available_age_groups": get_available_properties_str(Property.age_group),
            "format_instructions": format_instructions,
        }
    )

    chain = prompt | llm
    return chain
