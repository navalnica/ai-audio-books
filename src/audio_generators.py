import asyncio
import os
import re
from pathlib import Path
from uuid import uuid4
import random

from langchain_community.callbacks import get_openai_callback
from pydub import AudioSegment

from src.lc_callbacks import LCMessageLoggerAsync
from src.tts import tts_astream_consumed, sound_generation_astream
from src.utils import auto_retry, consume_aiter
from src.emotions.generation import (
    EffectGeneratorAsync,
    TextPreparationForTTSTaskOutput,
)
from src.emotions.utils import add_overlay_for_audio
from src.config import ELEVENLABS_MAX_PARALLEL, logger, OPENAI_MAX_PARALLEL
from src.text_split_chain import SplitTextOutput


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
            task = tts_astream_with_semaphore(
                voice_id=voice_id, text=character_phrase.text
            )
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


class AudioGeneratorWithEffects:

    def __init__(self):
        self.effect_generator = EffectGeneratorAsync()
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
            num_lines, fraction=float(0.2 * generate_effects),
        )
        logger.info(f"{generate_effects = }, {lines_for_sound_effect = }")

        modified_texts, sound_emotion_results = await self._prepare_text_for_tts(
            text_split, lines_for_sound_effect
        )

        tts_results, self.temp_files = await self._generate_tts_audio(
            text_split, modified_texts, character_to_voice
        )

        # Step 3: Add sound effects to selected lines
        audio_chunks = await self._add_sound_effects(
            tts_results, lines_for_sound_effect, sound_emotion_results, self.temp_files
        )

        # Step 4: Merge audio files
        normalized_audio_chunks = self._normalize_audio_chunks(
            audio_chunks, self.temp_files
        )
        final_output = self._merge_audio_files(
            normalized_audio_chunks, save_path=out_path
        )

        # Clean up temporary files
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

        task_emotion_code = "add_emotion"
        task_effects_code = "add_effects"

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

        emotion_tasks_results = [
            x.output for x in tasks_results if x.task == task_emotion_code
        ]
        effects_tasks_results = [
            x.output for x in tasks_results if x.task == task_effects_code
        ]

        return emotion_tasks_results, effects_tasks_results

    async def _generate_tts_audio(
        self,
        text_split: SplitTextOutput,
        modified_texts: list[dict],
        character_to_voice: dict[str, str],
    ) -> tuple[list[str], list[str]]:
        """Generate TTS audio for modified text."""
        tasks_for_tts = []
        temp_files = []

        async def tts_astream_with_semaphore(voice_id: str, text: str, params: dict):
            async with self.semaphore:
                bytes_ = await tts_astream_consumed(
                    voice_id=voice_id, text=text, params=params
                )
                # bytes_ = await consume_aiter(iter_)
                return bytes_

        for idx, (modified_text, character_phrase) in enumerate(
            zip(modified_texts, text_split.phrases)
        ):
            voice_id = character_to_voice[character_phrase.character]

            # Use the semaphore-protected TTS function
            task = tts_astream_with_semaphore(
                voice_id=voice_id,
                text=modified_text["modified_text"],
                params=modified_text["params"],
            )
            tasks_for_tts.append(task)

        # Gather all TTS results
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
        sound_emotion_results: list[dict],
        temp_files: list[str],
    ) -> list[str]:
        """Add sound effects to the selected lines."""
        audio_chunks = []
        for idx, tts_filename in enumerate(tts_audio_files):
            # If the line has sound emotion data, generate sound effect and overlay
            if idx in lines_for_sound_effect:
                # Get next sound effect data
                sound_effect_data = sound_emotion_results.pop(0)
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

    def _merge_audio_files(
        self, audio_filenames: list[str], save_path: Path | None = None
    ) -> Path:
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
