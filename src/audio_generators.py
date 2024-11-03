import asyncio
import json
import os
import random
import typing as t
from collections import Counter, defaultdict

from pathlib import Path
from typing import Callable, Any, AsyncIterator, Iterator
from uuid import uuid4


import openai
from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel

from pydub import AudioSegment
from requests import HTTPError
from setuptools.command.build_ext import use_stubs

from src.config import ELEVENLABS_MAX_PARALLEL, OPENAI_API_KEY, OPENAI_MAX_PARALLEL, logger
from src.schemas import SoundEffectsParams, TTSTimestampsResponse, TTSParams
from elevenlabs import VoiceSettings
from src.tts import tts_astream_consumed, sound_generation_consumed

# from src.emotions.generation import (
#     EffectGeneratorAsync,
#     TextPreparationForTTSTaskOutput,
# )
from src.utils import add_overlay_for_audio, GPTModels
from src.config import ELEVENLABS_MAX_PARALLEL, logger, OPENAI_MAX_PARALLEL
from src.text_split_chain import SplitTextOutput
from src import tts
from src.utils import add_overlay_for_audio, auto_retry, get_audio_duration
from .lc_callbacks import LCMessageLoggerAsync

from .prompts import (
    SOUND_EFFECT_GENERATION,
    SOUND_EFFECT_GENERATION_WITHOUT_DURATION_PREDICTION,
    TEXT_MODIFICATION_WITH_SSML,
)
from .sound_effects_design_chain import (
    create_sound_effects_design_chain,
    SoundEffectDescription,
    SoundEffectsDesignOutput,
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
                bytes_ = await tts.tts_astream_consumed(voice_id=voice_id, text=text)
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
        data_for_tts: list[dict],
        data_for_sound_effects: list[str],
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

    def save_audio_stream(self, audio_stream: Iterator[bytes], prefix: str, idx: str) -> str:
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

    async def generate_sound_effects(self, text: str) -> SoundEffectsDesignOutput:
        chain = create_sound_effects_design_chain(llm_model=GPTModels.GPT_4o)
        with get_openai_callback() as cb:
            res = chain.invoke({"text": text}, config={"callbacks": [LCMessageLoggerAsync()]})
        return res

    async def create_effect_text_task(
        self, semaphore: asyncio.Semaphore, text: str
    ) -> asyncio.Task:
        return asyncio.create_task(
            self.process_single_task(
                semaphore=semaphore,
                func=self.generate_sound_effects,
                text=text,
            )
        )

    async def _process_tts(
        self,
        semaphore: asyncio.Semaphore,
        params: TTSParams,
        idx: int,
    ) -> str:
        """Process TTS generation and save the result."""
        audio_stream = await self.process_single_task(
            semaphore=semaphore,
            func=tts.tts_w_timestamps,
            params=params,
        )

        return audio_stream.write_audio_to_file(
            filepath_no_ext=f'tts_output_{idx}', audio_format=params.output_format
        )

    async def _create_tts_task(
        self,
        semaphore: asyncio.Semaphore,
        params: TTSParams,
        idx: int,
    ) -> asyncio.Task:
        """Create a TTS generation task."""
        return asyncio.create_task(self._process_tts(semaphore, params, idx))

    async def _process_sound_effect(
        self, semaphore: asyncio.Semaphore, params: SoundEffectsParams, idx: str
    ) -> str:
        """Process TTS generation and save the result."""
        audio_stream = await self.process_single_task(
            semaphore=semaphore,
            func=sound_generation_consumed,
            params=params,
        )

        return self.save_audio_stream(audio_stream=audio_stream, prefix="sound_effect", idx=idx)

    async def _create_sound_effect_task(
        self, semaphore: asyncio.Semaphore, params: SoundEffectsParams, idx: str
    ) -> asyncio.Task:
        """Create a sound effect generation task."""
        return asyncio.create_task(self._process_sound_effect(semaphore, params, idx))

    def get_context_texts(
        self, items: list[dict], current_idx: int, window: int = 2
    ) -> tuple[str, str]:
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
        data_for_sound_effects: list[str],
        character_to_voice: dict[str, str],
        lines_for_sound_effect: list[int],
    ):
        semaphore = asyncio.Semaphore(ELEVENLABS_MAX_PARALLEL)

        async with asyncio.TaskGroup() as tg_tts:
            tts_tasks = [
                tg_tts.create_task(
                    self._process_tts(
                        semaphore=semaphore,
                        params=TTSParams(
                            voice_id=character_to_voice[phrase.character],
                            text=data_item["modified_text"],
                            previous_text=self.get_context_texts(data_for_tts, idx, window=2)[0],
                            next_text=self.get_context_texts(data_for_tts, idx, window=2)[1],
                            voice_settings=VoiceSettings(
                                stability=data_item["params"].get("stability"),
                                similarity_boost=data_item["params"].get("similarity_boost"),
                                style=data_item["params"].get("style"),
                                use_speaker_boost=True,
                            ),
                        ),
                        idx=idx,
                    ),
                    name=f"tts_{idx}",
                )
                for idx, (data_item, phrase) in enumerate(zip(data_for_tts, text_split.phrases))
            ]
        tts_results = [task.result() for task in tts_tasks]

        index_counts = Counter(lines_for_sound_effect)
        current_counts = defaultdict(int)
        effect_tasks = []

        async with asyncio.TaskGroup() as tg_effects:
            for i, idx in enumerate(lines_for_sound_effect):
                # Calculate duration based on total occurrences
                duration = get_audio_duration(tts_results[idx]) / index_counts[idx]

                # Create task with current count
                task = tg_effects.create_task(
                    self._process_sound_effect(
                        semaphore=semaphore,
                        params=SoundEffectsParams(
                            text=data_for_sound_effects[i],
                            duration_seconds=duration,
                            prompt_influence=0.6,
                        ),
                        idx=f"{idx}_{current_counts[idx]}",
                    ),
                    name=f"effect_{idx}_{current_counts[idx]}",
                )
                effect_tasks.append(task)

                # Increment counter after creating task
                current_counts[idx] += 1

        # tts_results = [task.result() for task in tts_tasks]
        sound_effects_results = [task.result() for task in effect_tasks]
        final_effect_paths = self.merge_sound_effects(sound_effects_results, index_counts)

        return tts_results, final_effect_paths

    def merge_sound_effects(self, effect_results, index_counts):
        merged_effects = {}

        for effect_path in effect_results:
            # Extract the effect name (e.g., 'effect_1_0') and split to get idx
            print(effect_path)
            base_name = effect_path.split("_")[2]

            if index_counts[int(base_name)] > 1:
                if base_name in merged_effects:
                    merged_effects[base_name] += AudioSegment.from_file(effect_path)
                else:
                    merged_effects[base_name] = AudioSegment.from_file(effect_path)
            else:
                merged_effects[base_name] = AudioSegment.from_file(effect_path)

        final_paths = []
        for idx, audio_segment in merged_effects.items():
            output_name = f"sound_effect_{idx}.wav"
            audio_segment.export(output_name, format="wav")
            final_paths.append(output_name)

        return final_paths

    async def _combine_tts_and_effects(
        self,
        tts_audio_files: list[str],
        sound_effects: list[str | None],
        lines_for_sound_effect: list[int],
    ) -> list[str]:
        """Combine TTS audio with sound effects where applicable."""
        combined_files = []
        lines_for_sound_effect = sorted(list(set(lines_for_sound_effect)))

        for idx, tts_filename in enumerate(tts_audio_files):
            if idx in lines_for_sound_effect:
                effect_idx = lines_for_sound_effect.index(idx)
                sound_effect_filename = sound_effects[effect_idx]
                new_tts_filename = add_overlay_for_audio(
                    main_audio_filename=tts_filename,
                    sound_effect_filename=sound_effect_filename,
                    cycling_effect=True,
                    effect_volume=0,
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
