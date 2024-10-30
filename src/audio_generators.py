import asyncio
import os
from pathlib import Path
from typing import Callable, Any, AsyncIterator, Iterator
from uuid import uuid4
import random

from pydub import AudioSegment

from src.tts import tts_astream_consumed, sound_generation_consumed
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
        self.effect_generator = EffectGeneratorAsync(predict_duration=True)
        self.semaphore = asyncio.Semaphore(ELEVENLABS_MAX_PARALLEL)
        self.temp_files = []

    async def generate_audio(
        self,
        text_split: SplitTextOutput,
        data_for_tts: list[dict],
        data_for_sound_effects: list[dict],
        character_to_voice: dict[str, str],
        lines_for_sound_effect: list[int],
        out_path: Path | None = None,
    ):
        tts_results, sound_effects_results = await self._generate_tts_with_effects(
            text_split=text_split,
            data_for_tts=data_for_tts,
            data_for_sound_effects=data_for_sound_effects,
            character_to_voice=character_to_voice,
            lines_for_sound_effect=lines_for_sound_effect,
        )
        self.temp_files.extend(tts_results)
        self.temp_files.extend(sound_effects_results)

        audio_chunks = await self._combine_tts_and_effects(
            tts_results, sound_effects_results, lines_for_sound_effect
        )

        final_output = self._merge_audio_files(
            self._normalize_audio_chunks(audio_chunks, self.temp_files),
            save_path=out_path,
        )

        self._cleanup_temp_files(self.temp_files)
        return final_output

    def select_lines_for_sound_effect(
        self, text_split: SplitTextOutput, fraction: float
    ) -> list[int]:
        """Select % of the lines randomly for sound effect generation."""
        num_lines = len(text_split.phrases)
        return random.sample(range(num_lines), k=int(fraction * num_lines))

    async def process_single_task(
        self, semaphore: asyncio.Semaphore, func: Callable, **params
    ) -> Any:
        async with semaphore:
            return await func(**params)

    def save_audio_stream(
        self, audio_stream: Iterator[bytes], prefix: str, idx: int
    ) -> str:
        filename = f"{prefix}_{idx}.wav"
        with open(filename, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)
        self.temp_files.append(str(filename))
        return str(filename)

    async def create_emotion_text_task(
        self, semaphore: asyncio.Semaphore, text: str
    ) -> asyncio.Task:
        return asyncio.create_task(
            self.process_single_task(
                semaphore=semaphore,
                func=self.effect_generator.add_emotion_to_text,
                text=text.strip().lower(),
            )
        )

    async def create_effect_text_task(
        self, semaphore: asyncio.Semaphore, text: str
    ) -> asyncio.Task:
        return asyncio.create_task(
            self.process_single_task(
                semaphore=semaphore,
                func=self.effect_generator.generate_parameters_for_sound_effect,
                text=text.strip().lower(),
            )
        )

    async def _process_tts(
        self,
        semaphore: asyncio.Semaphore,
        voice_id: str,
        text: str,
        prev_text: str,
        next_text: str,
        params: dict,
        idx: int,
    ) -> str:
        """Process TTS generation and save the result."""
        audio_stream = await self.process_single_task(
            semaphore=semaphore,
            func=tts_astream_consumed,
            voice_id=voice_id,
            text=text,
            prev_text=prev_text,
            next_text=next_text,
            params=params,
        )

        return self.save_audio_stream(
            audio_stream=audio_stream, prefix="tts_output", idx=idx
        )

    async def _create_tts_task(
        self,
        semaphore: asyncio.Semaphore,
        voice_id: str,
        text: str,
        prev_text: str,
        next_text: str,
        params: dict,
        idx: int,
    ) -> asyncio.Task:
        """Create a TTS generation task."""
        return asyncio.create_task(
            self._process_tts(semaphore, voice_id, text, prev_text, next_text, params, idx)
        )

    async def _process_sound_effect(
        self, semaphore: asyncio.Semaphore, sound_effect_data: dict, idx: int
    ) -> str:
        """Process TTS generation and save the result."""
        audio_stream = await self.process_single_task(
            semaphore=semaphore,
            func=sound_generation_consumed,
            sound_generation_data=sound_effect_data,
        )

        return self.save_audio_stream(
            audio_stream=audio_stream, prefix="sound_effect", idx=idx
        )

    async def _create_sound_effect_task(
        self, semaphore: asyncio.Semaphore, sound_effect_data: dict, idx: int
    ) -> asyncio.Task:
        """Create a sound effect generation task."""
        return asyncio.create_task(
            self._process_sound_effect(semaphore, sound_effect_data, idx)
        )

    def get_context_texts(self, items: list[dict], current_idx: int, window: int = 2) -> tuple[str, str]:
        prev_texts = []
        next_texts = []

        for i in range(current_idx - window, current_idx):
            if i >= 0:
                prev_texts.append(items[i]["modified_text"])
            else:
                prev_texts.append("")

        for i in range(current_idx + 1, current_idx + window + 1):
            if i < len(items):
                next_texts.append(items[i]["modified_text"])
            else:
                next_texts.append("")

        prev_text, next_text = " ".join(prev_texts), " ".join(next_texts)
        return prev_text, next_text

    async def _generate_tts_with_effects(
        self,
        text_split: SplitTextOutput,
        data_for_tts: list[dict],
        data_for_sound_effects: list[dict],
        character_to_voice: dict[str, str],
        lines_for_sound_effect: list[int],
    ):
        semaphore = asyncio.Semaphore(ELEVENLABS_MAX_PARALLEL)

        async with asyncio.TaskGroup() as tg:
            tts_tasks = [
                tg.create_task(
                    self._process_tts(
                        semaphore=semaphore,
                        voice_id=character_to_voice[phrase.character],
                        text=data_item["modified_text"],
                        prev_text=self.get_context_texts(data_for_tts, idx, window=2)[0],
                        next_text=self.get_context_texts(data_for_tts, idx, window=2)[1],
                        params=data_item["params"],
                        idx=idx,
                    ),
                    name=f"tts_{idx}",
                )
                for idx, (data_item, phrase) in enumerate(
                    zip(data_for_tts, text_split.phrases)
                )
            ]

            effect_tasks = []
            if lines_for_sound_effect and data_for_sound_effects:
                effect_tasks = [
                    tg.create_task(
                        self._process_sound_effect(
                            semaphore=semaphore,
                            sound_effect_data=data_for_sound_effects[i],
                            idx=idx,
                        ),
                        name=f"effect_{idx}",
                    )
                    for i, idx in enumerate(lines_for_sound_effect)
                ]

        tts_results = [task.result() for task in tts_tasks]
        sound_effects_results = [task.result() for task in effect_tasks]

        return tts_results, sound_effects_results

    async def _combine_tts_and_effects(
        self,
        tts_audio_files: list[str],
        sound_effects: list[str | None],
        lines_for_sound_effect: list[int],
    ) -> list[str]:
        """Combine TTS audio with sound effects where applicable."""
        combined_files = []

        for idx, tts_filename in enumerate(tts_audio_files):
            if idx in lines_for_sound_effect:
                effect_idx = lines_for_sound_effect.index(idx)
                sound_effect_filename = sound_effects[effect_idx]
                new_tts_filename = add_overlay_for_audio(
                    main_audio_filename=tts_filename,
                    sound_effect_filename=sound_effect_filename,
                    cycling_effect=True,
                    decrease_effect_volume=5,
                )
                combined_files.append(new_tts_filename)
            else:
                combined_files.append(tts_filename)

        return combined_files

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
            temp_files.append(audio_file)

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
