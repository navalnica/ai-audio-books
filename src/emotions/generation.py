import json
import typing as t
from abc import ABC, abstractmethod

import openai
from pydantic import BaseModel
from requests import HTTPError

from src.config import OPENAI_API_KEY, logger
from src.utils import auto_retry

from .prompts import (
    SOUND_EFFECT_GENERATION,
    SOUND_EFFECT_GENERATION_WITHOUT_DURATION_PREDICTION,
    TEXT_MODIFICATION,
    TEXT_MODIFICATION_WITH_SSML,
)
from .utils import get_audio_duration


class TextPreparationForTTSTaskOutput(BaseModel):
    task: str
    output: t.Any


class AbstractEffectGenerator(ABC):
    @abstractmethod
    async def generate_text_for_sound_effect(self, text) -> dict:
        pass

    @abstractmethod
    async def generate_parameters_for_sound_effect(
        self, text: str, generated_audio_file: str | None
    ) -> TextPreparationForTTSTaskOutput:
        pass

    @abstractmethod
    async def add_emotion_to_text(self, text: str) -> TextPreparationForTTSTaskOutput:
        pass


# class EffectGenerator(AbstractEffectGenerator):
#     def __init__(self, predict_duration: bool = True, model_type: str = "gpt-4o"):
#         self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
#         self.sound_effect_prompt = (
#             SOUND_EFFECT_GENERATION
#             if predict_duration
#             else SOUND_EFFECT_GENERATION_WITHOUT_DURATION_PREDICTION
#         )
#         self.text_modification_prompt = TEXT_MODIFICATION_WITH_SSML
#         self.model_type = model_type
#         logger.info(
#             f"EffectGenerator initialized with model_type: {model_type}, predict_duration: {predict_duration}"
#         )

#     @auto_retry
#     def generate_text_for_sound_effect(self, text: str) -> dict:
#         """Generate sound effect description and parameters based on input text."""
#         try:
#             completion = self.client.chat.completions.create(
#                 model=self.model_type,
#                 messages=[
#                     {"role": "system", "content": self.sound_effect_prompt},
#                     {"role": "user", "content": text},
#                 ],
#                 response_format={"type": "json_object"},
#             )
#             # Extracting the output
#             chatgpt_output = completion.choices[0].message.content

#             # Parse and return JSON response
#             output_dict = json.loads(chatgpt_output)
#             logger.info(
#                 "Successfully generated sound effect description: %s", output_dict
#             )
#             return output_dict

#         except json.JSONDecodeError as e:
#             logger.error("Failed to parse the output text as JSON: %s", e)
#             raise RuntimeError(
#                 f"Error: Failed to parse the output text as JSON.\nOutput: {chatgpt_output}"
#             )

#         except HTTPError as e:
#             logger.error("HTTP error occurred: %s", e)
#             raise RuntimeError(f"HTTP Error: {e}")

#         except Exception as e:
#             logger.error("Unexpected error occurred: %s", e)
#             raise RuntimeError(f"Unexpected Error: {e}")

#     @auto_retry
#     def generate_parameters_for_sound_effect(
#         self, text: str, generated_audio_file: str = None
#     ) -> dict:
#         llm_output = self.generate_text_for_sound_effect(text)
#         if generated_audio_file is not None:
#             llm_output["duration_seconds"] = get_audio_duration(generated_audio_file)
#             logger.info(
#                 "Added duration_seconds to output based on generated audio file: %s",
#                 generated_audio_file,
#             )
#         return llm_output

#     @auto_retry
#     def add_emotion_to_text(self, text: str) -> dict:
#         completion = self.client.chat.completions.create(
#             model=self.model_type,
#             messages=[
#                 {"role": "system", "content": self.text_modification_prompt},
#                 {"role": "user", "content": text},
#             ],
#             response_format={"type": "json_object"},
#         )
#         chatgpt_output = completion.choices[0].message.content
#         try:
#             output_dict = json.loads(chatgpt_output)
#             logger.info(
#                 "Successfully modified text with emotional cues: %s", output_dict
#             )
#             return output_dict
#         except json.JSONDecodeError as e:
#             logger.error("Error in parsing the modified text: %s", e)
#             raise f"error, output_text: {chatgpt_output}"


class EffectGeneratorAsync(AbstractEffectGenerator):
    def __init__(self, predict_duration: bool = True, model_type: str = "gpt-4o"):
        self.client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.sound_effect_prompt = (
            SOUND_EFFECT_GENERATION
            if predict_duration
            else SOUND_EFFECT_GENERATION_WITHOUT_DURATION_PREDICTION
        )
        self.text_modification_prompt = TEXT_MODIFICATION_WITH_SSML
        self.model_type = model_type

    @auto_retry
    async def generate_text_for_sound_effect(self, text: str) -> dict:
        """Asynchronous version to generate sound effect description."""
        try:
            completion = await self.client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "system", "content": self.sound_effect_prompt},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
            )
            # Extracting the output
            chatgpt_output = completion.choices[0].message.content

            # Parse and return JSON response
            output_dict = json.loads(chatgpt_output)
            logger.info(
                "Successfully generated sound effect description: %s", output_dict
            )
            return output_dict

        except json.JSONDecodeError as e:
            logger.error("Failed to parse the output text as JSON: %s", e)
            raise RuntimeError(
                f"Error: Failed to parse the output text as JSON.\nOutput: {chatgpt_output}"
            )

        except HTTPError as e:
            logger.error("HTTP error occurred: %s", e)
            raise RuntimeError(f"HTTP Error: {e}")

        except Exception as e:
            logger.error("Unexpected error occurred: %s", e)
            raise RuntimeError(f"Unexpected Error: {e}")

    @auto_retry
    async def generate_parameters_for_sound_effect(
        self, text: str, generated_audio_file: str | None = None
    ) -> TextPreparationForTTSTaskOutput:
        llm_output = await self.generate_text_for_sound_effect(text)
        if generated_audio_file is not None:
            llm_output["duration_seconds"] = get_audio_duration(generated_audio_file)
            logger.info(
                "Added duration_seconds to output based on generated audio file: %s",
                generated_audio_file,
            )
        return TextPreparationForTTSTaskOutput(task="add_effects", output=llm_output)

    @auto_retry
    async def add_emotion_to_text(self, text: str) -> TextPreparationForTTSTaskOutput:
        completion = await self.client.chat.completions.create(
            model=self.model_type,
            messages=[
                {"role": "system", "content": self.text_modification_prompt},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
        )
        chatgpt_output = completion.choices[0].message.content
        try:
            output_dict = json.loads(chatgpt_output)
            logger.info(
                "Successfully modified text with emotional cues: %s", output_dict
            )
            return TextPreparationForTTSTaskOutput(
                task="add_emotion", output=output_dict
            )
        except json.JSONDecodeError as e:
            logger.error("Error in parsing the modified text: %s", e)
            raise f"error, output_text: {chatgpt_output}"
