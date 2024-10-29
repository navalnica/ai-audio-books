import asyncio
import json
import os
import random
import typing as t
from pathlib import Path
from uuid import uuid4

import openai
from pydantic import BaseModel
from pydub import AudioSegment
from requests import HTTPError

from src.config import ELEVENLABS_MAX_PARALLEL, OPENAI_API_KEY, OPENAI_MAX_PARALLEL, logger
from src.schemas import SoundEffectsParams
from src.text_split_chain import SplitTextOutput
from src.tts import sound_generation_consumed, tts_astream_consumed
from src.utils import add_overlay_for_audio, auto_retry, get_audio_duration

from .prompts import (
    SOUND_EFFECT_GENERATION,
    SOUND_EFFECT_GENERATION_WITHOUT_DURATION_PREDICTION,
    TEXT_MODIFICATION_WITH_SSML,
)


class AudioGeneratorSimple:

    async def generate_audio(
        self,
        text_split: SplitTextOutput,
        character_to_voice: dict[str, str],
    ) -> Path:
        semaphore = asyncio.Semaphore(ELEVENLABS_MAX_PARALLEL)

        async def tts_astream_with_semaphore(voice_id: str, text: str):
            async with semaphore:
                bytes_ = await tts_astream_consumed(voice_id=voice_id, text=text)
                # bytes_ = await consume_aiter(iter_)
                return bytes_

        tasks = []
        for character_phrase in text_split.phrases:
            voice_id = character_to_voice[character_phrase.character]
            task = tts_astream_with_semaphore(voice_id=voice_id, text=character_phrase.text)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        save_dir = Path("data") / "books"
        save_dir.mkdir(exist_ok=True)
        audio_combined_fp = save_dir / f"{uuid4()}.wav"

        logger.info(f'saving generated audio book to: "{audio_combined_fp}"')
        with open(audio_combined_fp, "wb") as ab:
            for result in results:
                for chunk in result:
                    ab.write(chunk)

        return audio_combined_fp


class TextPreparationForTTSTaskOutput(BaseModel):
    task: str
    output: t.Any


class EffectGeneratorAsync:
    def __init__(self, predict_duration: bool, model_type: str = "gpt-4o"):
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
            logger.info("Successfully generated sound effect description: %s", output_dict)
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
        return TextPreparationForTTSTaskOutput(task="sound_effects", output=llm_output)

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
            logger.info("Successfully modified text with emotional cues: %s", output_dict)
            return TextPreparationForTTSTaskOutput(task="tts", output=output_dict)
        except json.JSONDecodeError as e:
            logger.error("Error in parsing the modified text: %s", e)
            raise ValueError(f"error, output_text: {chatgpt_output}")


class AudioGeneratorWithEffects:

    def __init__(self):
        self.effect_generator = EffectGeneratorAsync(predict_duration=True)
        self.semaphore = asyncio.Semaphore(ELEVENLABS_MAX_PARALLEL)
        self.temp_files = []

    async def generate_audio(
        self,
        text_split: SplitTextOutput,
        character_to_voice: dict[str, str],
        out_path: Path | None = None,
        *,
        generate_effects: bool = True,
    ) -> Path:
        """Main method to generate the audiobook with TTS, emotion, and sound effects."""
        num_lines = len(text_split.phrases)
        lines_for_sound_effect = self._select_lines_for_sound_effect(
            num_lines,
            fraction=float(0.2 * generate_effects),
        )
        logger.info(f"{generate_effects = }, {lines_for_sound_effect = }")

        data_for_tts, data_for_sound_effects = await self._prepare_text_for_tts(
            text_split, lines_for_sound_effect
        )

        tts_results, self.temp_files = await self._generate_tts_audio(
            text_split, data_for_tts, character_to_voice
        )

        audio_chunks = await self._add_sound_effects(
            tts_results, lines_for_sound_effect, data_for_sound_effects, self.temp_files
        )

        normalized_audio_chunks = self._normalize_audio_chunks(audio_chunks, self.temp_files)

        final_output = self._merge_audio_files(normalized_audio_chunks, save_path=out_path)

        self._cleanup_temp_files(self.temp_files)

        return final_output

    def _select_lines_for_sound_effect(self, num_lines: int, fraction: float) -> list[int]:
        """Select % of the lines randomly for sound effect generation."""
        return random.sample(range(num_lines), k=int(fraction * num_lines))

    async def _prepare_text_for_tts(
        self, text_split: SplitTextOutput, lines_for_sound_effect: list[int]
    ) -> tuple[list[dict], list[dict]]:
        semaphore = asyncio.Semaphore(OPENAI_MAX_PARALLEL)

        async def run_task_with_semaphore(func, **params):
            async with semaphore:
                outputs = await func(**params)
                return outputs

        task_code_tts = "tts"
        task_code_sounde_effects = "sound_effects"

        tasks = []

        for idx, character_phrase in enumerate(text_split.phrases):
            character_text = character_phrase.text.strip().lower()

            tasks.append(
                run_task_with_semaphore(
                    func=self.effect_generator.add_emotion_to_text,
                    text=character_text,
                )
            )

            # If this line needs sound effects, generate parameters
            if idx in lines_for_sound_effect:
                tasks.append(
                    run_task_with_semaphore(
                        func=self.effect_generator.generate_parameters_for_sound_effect,
                        text=character_text,
                    )
                )

        tasks_results: list[TextPreparationForTTSTaskOutput] = []
        tasks_results = await asyncio.gather(*tasks)

        tts_tasks_results = [x.output for x in tasks_results if x.task == task_code_tts]
        sound_effects_tasks_results = [
            x.output for x in tasks_results if x.task == task_code_sounde_effects
        ]

        return tts_tasks_results, sound_effects_tasks_results

    async def _generate_tts_audio(
        self,
        text_split: SplitTextOutput,
        data_for_tts: list[dict],
        character_to_voice: dict[str, str],
    ) -> tuple[list[str], list[str]]:
        """Generate TTS audio for modified text."""
        tasks_for_tts = []
        temp_files = []

        async def tts_astream_with_semaphore(voice_id: str, text: str, params: dict):
            async with self.semaphore:
                bytes_ = await tts_astream_consumed(voice_id=voice_id, text=text, params=params)
                # bytes_ = await consume_aiter(iter_)
                return bytes_

        for idx, (data_item, character_phrase) in enumerate(zip(data_for_tts, text_split.phrases)):
            voice_id = character_to_voice[character_phrase.character]

            task = tts_astream_with_semaphore(
                voice_id=voice_id,
                text=data_item["modified_text"],
                params=data_item["params"],
            )
            tasks_for_tts.append(task)

        tts_results = await asyncio.gather(*tasks_for_tts)

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
        data_for_sound_effects: list[dict],
        temp_files: list[str],
    ) -> list[str]:
        """Add sound effects to the selected lines."""

        semaphore = asyncio.Semaphore(ELEVENLABS_MAX_PARALLEL)

        async def _process_single_phrase(
            tts_filename: str,
            sound_effects_params: SoundEffectsParams | None,
            sound_effect_filename: str,
        ):
            if sound_effects_params is None:
                return (tts_filename, [])

            async with semaphore:
                sound_result = await sound_generation_consumed(params=sound_effects_params)

            # save to file
            with open(sound_effect_filename, "wb") as ab:
                for chunk in sound_result:
                    ab.write(chunk)

            # overlay sound effect on TTS audio
            tts_with_effects_filename = add_overlay_for_audio(
                main_audio_filename=tts_filename,
                sound_effect_filename=sound_effect_filename,
                cycling_effect=True,
                decrease_effect_volume=5,
            )
            tmp_files = [sound_effect_filename, tts_with_effects_filename]
            return (tts_with_effects_filename, tmp_files)

        tasks = []
        for idx, tts_filename in enumerate(tts_audio_files):
            sound_effect_filename = f"sound_effect_{idx}.wav"

            if idx not in lines_for_sound_effect:
                tasks.append(
                    _process_single_phrase(
                        tts_filename=tts_filename,
                        sound_effects_params=None,
                        sound_effect_filename=sound_effect_filename,
                    )
                )
            else:
                sound_effects_params = data_for_sound_effects.pop(0)
                tasks.append(
                    _process_single_phrase(
                        tts_filename=tts_filename,
                        sound_effects_params=sound_effects_params,
                        sound_effect_filename=sound_effect_filename,
                    )
                )

        outputs = await asyncio.gather(*tasks)
        audio_chunks = [x[0] for x in outputs]
        tmp_files_to_add = [item for x in outputs for item in x[1]]
        temp_files.extend(tmp_files_to_add)

        return audio_chunks

    def _normalize_audio(
        self, audio_segment: AudioSegment, target_dBFS: float = -20.0
    ) -> AudioSegment:
        """Normalize an audio segment to the target dBFS level."""
        change_in_dBFS = target_dBFS - audio_segment.dBFS
        return audio_segment.apply_gain(change_in_dBFS)

    def _normalize_audio_chunks(
        self, audio_filenames: list[str], temp_files, target_dBFS: float = -20.0
    ) -> list[str]:
        """Normalize all audio chunks to the target volume level."""
        normalized_files = []
        for audio_file in audio_filenames:
            audio_segment = AudioSegment.from_file(audio_file)
            normalized_audio = self._normalize_audio(audio_segment, target_dBFS)

            normalized_filename = f"normalized_{Path(audio_file).stem}.wav"
            normalized_audio.export(normalized_filename, format="wav")
            normalized_files.append(normalized_filename)
            temp_files.append(normalized_filename)

        return normalized_files

    def _merge_audio_files(self, audio_filenames: list[str], save_path: Path | None = None) -> Path:
        """Helper function to merge multiple audio files into one."""
        combined = AudioSegment.from_file(audio_filenames[0])
        for filename in audio_filenames[1:]:
            next_audio = AudioSegment.from_file(filename)
            combined += next_audio  # Concatenate the audio

        if save_path is None:
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
