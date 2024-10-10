import asyncio
import os
import re
from pathlib import Path
from uuid import uuid4
import random

from langchain_community.callbacks import get_openai_callback
from pydub import AudioSegment

from src.lc_callbacks import LCMessageLoggerAsync
from src.tts import tts_astream, sound_generation_astream
from src.utils import consume_aiter
from src.emotions.generation import EffectGeneratorAsync
from src.emotions.utils import add_overlay_for_audio

api_key = os.getenv("AIML_API_KEY")


class AudioGeneratorSimple:

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


class AudioGeneratorWithEffects:

    def __init__(self):
        self.effect_generator = EffectGeneratorAsync(api_key)

    async def generate_audio_with_text_modification(
        self,
        annotated_text: str,
        character_to_voice: dict[str, str],
    ) -> Path:
        """Main method to generate the audiobook with TTS, emotion, and sound effects."""
        num_lines = len(annotated_text.splitlines())
        lines_for_sound_effect = self._select_lines_for_sound_effect(num_lines)

        # Step 1: Process and modify text
        modified_texts, sound_emotion_results = await self._process_and_modify_text(
            annotated_text, lines_for_sound_effect
        )

        # Step 2: Generate TTS audio for modified text
        tts_results, temp_files = await self._generate_tts_audio(
            annotated_text, modified_texts, character_to_voice
        )

        # Step 3: Add sound effects to selected lines
        audio_chunks = await self._add_sound_effects(
            tts_results, lines_for_sound_effect, sound_emotion_results, temp_files
        )

        # Step 4: Merge audio files
        final_output = self._merge_audio_files(audio_chunks)

        # Clean up temporary files
        self._cleanup_temp_files(temp_files)

        return final_output

    def _select_lines_for_sound_effect(self, num_lines: int) -> list[int]:
        """Select 20% of the lines randomly for sound effect generation."""
        return random.sample(range(num_lines), k=int(0.2 * num_lines))

    async def _process_and_modify_text(
        self, annotated_text: str, lines_for_sound_effect: list[int]
    ) -> tuple[list[dict], list[dict]]:
        """Process the text by modifying it and generating tasks for sound effects."""
        tasks_for_text_modification = []
        sound_emotion_tasks = []

        for idx, line in enumerate(annotated_text.splitlines()):
            cleaned_line = line.strip().lower()
            if not cleaned_line:
                continue

            # Extract character text
            character_text = cleaned_line[cleaned_line.rfind("]") + 1 :].lstrip()

            # Add text emotion modification task
            tasks_for_text_modification.append(
                self.effect_generator.add_emotion_to_text(character_text)
            )

            # If this line needs sound effects, generate parameters
            if idx in lines_for_sound_effect:
                sound_emotion_tasks.append(
                    self.effect_generator.generate_parameters_for_sound_effect(
                        character_text
                    )
                )

        # Await tasks for text modification and sound effects
        modified_texts = await asyncio.gather(*tasks_for_text_modification)
        sound_emotion_results = await asyncio.gather(*sound_emotion_tasks)

        return modified_texts, sound_emotion_results

    async def _generate_tts_audio(
        self,
        annotated_text: str,
        modified_texts: list[dict],  # TODO ? type ?
        character_to_voice: dict[str, str],
    ) -> tuple[list[str], list[str]]:
        """Generate TTS audio for modified text."""
        tasks_for_tts = []
        temp_files = []
        current_character = "narrator"

        for idx, (modified_text, line) in enumerate(
            zip(modified_texts, annotated_text.splitlines())
        ):
            cleaned_line = line.strip().lower()

            # Extract character
            try:
                current_character = re.findall(r"\[[\w\s]+\]", cleaned_line)[0][1:-1]
            except IndexError:
                pass

            # Get voice ID and generate TTS
            voice_id = character_to_voice[current_character]
            tasks_for_tts.append(
                tts_astream(
                    voice_id=voice_id,
                    text=modified_text["text"],  # TODO ? type ?
                    params=modified_texts["params"],  # TODO ? type ?
                )
            )

        # Gather all TTS results
        tts_results = await asyncio.gather(*(consume_aiter(t) for t in tasks_for_tts))

        # Save the results to temporary files
        tts_audio_files = []
        for idx, tts_result in enumerate(tts_results):
            tts_filename = f"tts_output_{idx}.wav"
            with open(tts_filename, "wb") as ab:
                for chunk in tts_result:
                    ab.write(chunk)
            tts_audio_files.append(tts_filename)
            temp_files.append(tts_filename)

        return tts_audio_files, temp_files

    async def _add_sound_effects(
        self,
        tts_audio_files: list[str],
        lines_for_sound_effect: list[int],
        sound_emotion_results: list[dict],
        temp_files: list[str],
    ) -> list[str]:
        """Add sound effects to the selected lines."""
        audio_chunks = []
        for idx, tts_filename in enumerate(tts_audio_files):
            # If the line has sound emotion data, generate sound effect and overlay
            if idx in lines_for_sound_effect:
                sound_effect_data = sound_emotion_results.pop(
                    0
                )  # Get next sound effect data
                sound_effect_filename = f"sound_effect_{idx}.wav"

                # Generate sound effect asynchronously
                sound_result = await consume_aiter(
                    sound_generation_astream(sound_effect_data)
                )
                with open(sound_effect_filename, "wb") as ab:
                    for chunk in sound_result:
                        ab.write(chunk)

                # Add sound effect overlay
                output_filename = add_overlay_for_audio(
                    main_audio_filename=tts_filename,
                    sound_effect_filename=sound_effect_filename,
                    cycling_effect=True,
                    decrease_effect_volume=5,
                )
                audio_chunks.append(output_filename)
                temp_files.append(sound_effect_filename)  # Track temp files
                temp_files.append(output_filename)
            else:
                audio_chunks.append(tts_filename)

        return audio_chunks

    def _merge_audio_files(self, audio_filenames: list[str]) -> Path:
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

    def _cleanup_temp_files(self, temp_files: list[str]) -> None:
        """Helper function to delete all temporary files."""
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except FileNotFoundError:
                continue
