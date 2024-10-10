import asyncio
import os
import re
from pathlib import Path
from uuid import uuid4
import random

import pandas as pd
from langchain_community.callbacks import get_openai_callback
from pydub import AudioSegment

from src.lc_callbacks import LCMessageLoggerAsync
from src.select_voice_chain import AllCharactersProperties, create_voice_mapping_chain
from src.text_split_chain import SplitTextOutput, create_split_text_chain
from src.tts import tts_astream, sound_generation_astream
from src.utils import GPTModels, consume_aiter
from src.emotions.generation import EffectGeneratorAsync
from src.emotions.utils import add_overlay_for_audio

VOICES = pd.read_csv("data/11labs_tts_voices.csv").query("language == 'en'")

api_key = os.getenv("AIML_API_KEY")

class AudiobookBuilder:
    def __init__(self) -> None:
        self._default_narrator_voice = "ALY2WaJPY0oBJlqpQbfW"
        self.effect_generator = EffectGeneratorAsync(api_key)

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
        tasks_for_text_modification = []
        tasks_for_tts = []
        sound_emotion_tasks = []
        current_character = "narrator"
        lines = annotated_text.splitlines()

        # Randomly select 20% of lines for sound effect generation
        num_lines = len(lines)
        lines_for_sound_effect = random.sample(range(num_lines), k=int(0.2 * num_lines))

        for idx, line in enumerate(annotated_text.splitlines()):
            cleaned_line = line.strip().lower()
            if not cleaned_line:
                continue
            try:
                current_character = re.findall(r"\[[\w\s]+\]", cleaned_line)[0][1:-1]
            except:
                pass

            character_text = cleaned_line[cleaned_line.rfind("]") + 1 :].lstrip()

            # emotion generation for random
            if idx in lines_for_sound_effect:
                sound_emotion_tasks.append(
                        self.effect_generator.generate_parameters_for_sound_effect(character_text)
                )
            tasks_for_text_modification.append(self.effect_generator.add_emotion_to_text(character_text))

        modified_texts = await asyncio.gather(*tasks_for_text_modification)
        text_for_emotion_generation = await asyncio.gather(*sound_emotion_tasks)

        for idx, modified_text, line in enumerate(zip(modified_texts, annotated_text.splitlines())):
            cleaned_line = line.strip().lower()
            try:
                current_character = re.findall(r"\[[\w\s]+\]", cleaned_line)[0][1:-1]
            except IndexError:
                pass
            voice_id = character_to_voice[current_character]
            tasks_for_tts.append(tts_astream(voice_id=voice_id, text=modified_text["text"], params=modified_texts['params']))

        tts_results = await asyncio.gather(*(consume_aiter(t) for t in tasks_for_tts))


        audio_chunks = []
        temp_files = []
        for idx, tts_result in enumerate(tts_results):
            # Save TTS audio to a temporary file
            tts_filename = f"tts_output_{idx}.wav"
            with open(tts_filename, "wb") as ab:
                for chunk in tts_result:
                    ab.write(chunk)
            temp_files.append(tts_filename)

            # If this line was selected for sound emotion, overlay the sound effect
            if idx in lines_for_sound_effect:
                sound_generation_data = text_for_emotion_generation.pop(0)  # Get the next sound effect data
                sound_effect_filename = f"sound_effect_{idx}.wav"
                # Generate the sound effect audio asynchronously
                sound_result = await consume_aiter(sound_generation_astream(sound_generation_data))
                with open(sound_effect_filename, "wb") as ab:
                    for chunk in sound_result:
                        ab.write(chunk)
                temp_files.append(sound_effect_filename)

                # Overlay the sound effect on the TTS audio
                output_filename = add_overlay_for_audio(
                    main_audio_filename=tts_filename,
                    sound_effect_filename=sound_effect_filename,
                    cycling_effect=True,
                    decrease_effect_volume=5
                )
                audio_chunks.append(output_filename)
                temp_files.append(output_filename)
            else:
                audio_chunks.append(tts_filename)

        # Merge all individual audio files into one final audiobook
        final_output = self.merge_audio_files(audio_chunks)
        #clean tmp files
        self.cleanup_temp_files(temp_files)

        return final_output

    def merge_audio_files(self, audio_filenames: list[str]) -> Path:
        """Helper function to merge multiple audio files into one."""
        combined = AudioSegment.from_file(audio_filenames[0])
        for filename in audio_filenames[1:]:
            next_audio = AudioSegment.from_file(filename)
            combined += next_audio  # Concatenate the audio

        save_dir = Path("data") / "books"
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f"{uuid4()}.wav"
        combined.export(save_path, format="wav")
        return Path(save_path)

    def cleanup_temp_files(self, temp_files: list[str]) -> None:
        """Helper function to delete all temporary files."""
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except FileNotFoundError:
                continue

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
