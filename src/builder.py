import asyncio
import os
from pathlib import Path
from uuid import uuid4

from langchain_community.callbacks import get_openai_callback
from pydub import AudioSegment

from src import tts, utils
from src.config import ELEVENLABS_MAX_PARALLEL, OPENAI_MAX_PARALLEL, logger
from src.lc_callbacks import LCMessageLoggerAsync
from src.preprocess_tts_text_chain import TTSTextProcessorWithSSML
from src.schemas import SoundEffectsParams, TTSParams, TTSTimestampsResponse
from src.select_voice_chain import SelectVoiceChainOutput, VoiceSelector
from src.text_split_chain import SplitTextOutput, create_split_text_chain
from src.utils import GPTModels


class AudiobookBuilder:

    def __init__(self, rm_artifacts: bool = False):
        self.voice_selector = VoiceSelector()
        self.text_tts_processor = TTSTextProcessorWithSSML()
        self.rm_artifacts = rm_artifacts
        self.name = type(self).__name__

    async def _split_text(self, text: str) -> SplitTextOutput:
        chain = create_split_text_chain(llm_model=GPTModels.GPT_4o)
        with get_openai_callback() as cb:
            chain_out = await chain.ainvoke(
                {"text": text}, config={"callbacks": [LCMessageLoggerAsync()]}
            )
        logger.info(f'end of splitting text into characters. openai callback stats: {cb}')
        return chain_out

    async def _map_characters_to_voices(
        self, text_split: SplitTextOutput
    ) -> SelectVoiceChainOutput:
        chain = self.voice_selector.create_voice_mapping_chain(llm_model=GPTModels.GPT_4o)
        with get_openai_callback() as cb:
            chain_out = await chain.ainvoke(
                {
                    "text": text_split.text_annotated,
                    "characters": text_split.characters,
                },
                config={"callbacks": [LCMessageLoggerAsync()]},
            )
        logger.info(f'end of mapping characters to voices. openai callback stats: {cb}')
        return chain_out

    async def _prepare_text_for_tts(self, text_split: SplitTextOutput) -> list[TTSParams]:
        semaphore = asyncio.Semaphore(OPENAI_MAX_PARALLEL)

        async def run_task_with_semaphore(func, **params):
            async with semaphore:
                outputs = await func(**params)
                return outputs

        tasks = []

        for character_phrase in text_split.phrases:
            tasks.append(
                run_task_with_semaphore(
                    func=self.text_tts_processor.run,
                    text=character_phrase.text,
                )
            )

        tts_tasks_results = await asyncio.gather(*tasks)

        return tts_tasks_results

    @staticmethod
    async def _generate_tts_audio(
        text_split: SplitTextOutput,
        tts_params_list: list[TTSParams],
        character_to_voice: dict[str, str],
        out_dp: str,
    ) -> list[str]:
        semaphore = asyncio.Semaphore(ELEVENLABS_MAX_PARALLEL)

        async def _tts_with_semaphore(params: TTSParams) -> TTSTimestampsResponse:
            async with semaphore:
                return await tts.tts_w_timestamps(params=params)

        tasks_for_tts = []
        for character_phrase, cur_tts_params in zip(text_split.phrases, tts_params_list):
            # TODO: verify that voice is assigned correctly
            cur_tts_params.voice_id = character_to_voice[character_phrase.character]
            task = _tts_with_semaphore(params=cur_tts_params)
            tasks_for_tts.append(task)
        tts_results: list[TTSTimestampsResponse] = await asyncio.gather(*tasks_for_tts)

        tts_audio_fps = []
        for ix, (params, res) in enumerate(zip(tts_params_list, tts_results)):
            out_fp_no_ext = os.path.join(out_dp, f'tts_output_{ix}')
            out_fp = res.write_audio_to_file(
                filepath_no_ext=out_fp_no_ext, audio_format=params.output_format
            )
            tts_audio_fps.append(out_fp)

        return tts_audio_fps

    async def _add_sound_effects(
        self,
        tts_audio_fps: list[str],
        data_for_sound_effects: list[dict],
        temp_files: list[str],
    ) -> list[str]:
        """Add sound effects to the selected lines."""

        semaphore = asyncio.Semaphore(ELEVENLABS_MAX_PARALLEL)

        async def _process_single_phrase(
            tts_fp: str,
            sound_effects_params: SoundEffectsParams | None,
            sound_effect_fp: str,
        ):
            if sound_effects_params is None:
                return (tts_fp, [])

            async with semaphore:
                sound_result = await tts.sound_generation_consumed(params=sound_effects_params)

            with open(sound_effect_fp, "wb") as ab:
                for chunk in sound_result:
                    ab.write(chunk)

            tts_with_effects_fp = utils.add_overlay_for_audio(
                audio1_fp=tts_fp, audio2_fp=sound_effect_fp, decrease_effect_volume=5
            )
            tmp_files = [sound_effect_fp, tts_with_effects_fp]
            return (tts_with_effects_fp, tmp_files)

        tasks = []
        for idx, tts_fp in enumerate(tts_audio_fps):
            sound_effect_fp = f"sound_effect_{idx}.wav"

            # TODO: broken code
            sound_effects_params = data_for_sound_effects.pop(0)

            tasks.append(
                _process_single_phrase(
                    tts_fp=tts_fp,
                    sound_effects_params=sound_effects_params,
                    sound_effect_fp=sound_effect_fp,
                )
            )

        outputs = await asyncio.gather(*tasks)
        tts_with_effects_fps = [x[0] for x in outputs]
        tmp_files_to_add = [item for x in outputs for item in x[1]]
        temp_files.extend(tmp_files_to_add)

        return tts_with_effects_fps

    def _normalize_audio_chunks(
        self, audio_fps: list[str], out_dp: str, target_dBFS: float = -20.0
    ) -> list[str]:
        """Normalize all audio chunks to the target volume level."""
        fps = []
        for in_fp in audio_fps:
            audio_segment = AudioSegment.from_file(in_fp)
            normalized_audio = utils.normalize_audio(audio_segment, target_dBFS)

            out_fp = os.path.join(out_dp, f"normalized_{Path(in_fp).stem}.wav")
            normalized_audio.export(out_fp, format="wav")
            fps.append(out_fp)

        return fps

    def _concatenate_audiofiles(self, audio_fps: list[str], out_wav_fp: str):
        concat = AudioSegment.from_file(audio_fps[0])
        for filename in audio_fps[1:]:
            next_audio = AudioSegment.from_file(filename)
            concat += next_audio
        logger.info(f'saving concatenated audiobook to: "{out_wav_fp}"')
        concat.export(out_wav_fp, format="wav")

    async def run(self, text: str, *, generate_effects: bool):
        now_str = utils.get_utc_now_str()
        uuid_trimmed = str(uuid4()).split('-')[0]
        dir_name = f'{now_str}-{uuid_trimmed}'
        out_dp_root = os.path.join('data', 'audiobooks', dir_name)
        os.makedirs(out_dp_root, exist_ok=False)

        with get_openai_callback() as cb:
            # NOTE: we don't use langchain in every LLM calls.
            # thus, usage from this callback is not representative

            # TODO: currenly, we are constantly writing and reading audio segments from files.
            # I think it will be more efficient to keep all audio in memory.

            # TODO: call sound effects chain in parallel with this chain
            text_split = await self._split_text(text)

            select_voice_chain_out = await self._map_characters_to_voices(text_split=text_split)
            character_to_voice = select_voice_chain_out.character2voice

            tts_params_list = await self._prepare_text_for_tts(text_split=text_split)

            tts_dp = os.path.join(out_dp_root, 'tts')
            os.makedirs(tts_dp)
            tts_audio_fps = await self._generate_tts_audio(
                text_split=text_split,
                tts_params_list=tts_params_list,
                character_to_voice=character_to_voice,
                out_dp=tts_dp,
            )

            # tts_with_effects_fps = await self._add_sound_effects(
            #     tts_audio_fps, data_for_sound_effects, self.temp_files
            # )

            normalized_dp = os.path.join(out_dp_root, 'normalized')
            os.makedirs(normalized_dp)
            normalized_audio_fps = self._normalize_audio_chunks(
                audio_fps=tts_audio_fps,
                # audio_fps=tts_with_effects_fps,
                out_dp=normalized_dp,
            )

            final_wav_fp = os.path.join(out_dp_root, f'audiobook_{now_str}.wav')
            self._concatenate_audiofiles(audio_fps=normalized_audio_fps, out_wav_fp=final_wav_fp)

            utils.rm_dir_conditional(dp=out_dp_root, to_remove=self.rm_artifacts)

        logger.info(f'end of {self.name}.run(). openai callback stats: {cb}')
        return final_wav_fp
