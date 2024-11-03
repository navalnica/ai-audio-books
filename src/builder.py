import asyncio
import os
from pathlib import Path
from uuid import uuid4

from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel
from pydub import AudioSegment

from src import tts, utils
from src.config import ELEVENLABS_MAX_PARALLEL, OPENAI_MAX_PARALLEL, logger
from src.lc_callbacks import LCMessageLoggerAsync
from src.preprocess_tts_text_chain import TTSTextProcessorWithSSML
from src.schemas import SoundEffectsParams, TTSParams, TTSTimestampsAlignment, TTSTimestampsResponse
from src.select_voice_chain import SelectVoiceChainOutput, VoiceSelector
from src.sound_effects_design import (
    SoundEffectDescription,
    SoundEffectsDesignOutput,
    create_sound_effects_design_chain,
)
from src.text_split_chain import SplitTextOutput, create_split_text_chain
from src.utils import GPTModels


class TTSPhrasesGenerationOutput(BaseModel):
    audio_fps: list[str]
    char2time: TTSTimestampsAlignment


class AudiobookBuilder:

    def __init__(self, rm_artifacts: bool = False):
        self.voice_selector = VoiceSelector()
        self.text_tts_processor = TTSTextProcessorWithSSML()
        self.rm_artifacts = rm_artifacts
        self.min_sound_effect_duration_sec = 1
        self.sound_effects_prompt_influence = 0.75  # seems to work nicely
        self.name = type(self).__name__

    @staticmethod
    async def _split_text(text: str) -> SplitTextOutput:
        chain = create_split_text_chain(llm_model=GPTModels.GPT_4o)
        with get_openai_callback() as cb:
            chain_out = await chain.ainvoke(
                {"text": text}, config={"callbacks": [LCMessageLoggerAsync()]}
            )
        logger.info(f'end of splitting text into characters. openai callback stats: {cb}')
        return chain_out

    @staticmethod
    async def _design_sound_effects(text: str) -> SoundEffectsDesignOutput:
        chain = create_sound_effects_design_chain(llm_model=GPTModels.GPT_4o)
        with get_openai_callback() as cb:
            res = await chain.ainvoke(
                {"text": text}, config={"callbacks": [LCMessageLoggerAsync()]}
            )
        logger.info(
            f'designed {len(res.sound_effects_descriptions)} sound effects. '
            f'openai callback stats: {cb}'
        )
        return res

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
        character2voice: dict[str, str],
        out_dp: str,
    ) -> TTSPhrasesGenerationOutput:
        semaphore = asyncio.Semaphore(ELEVENLABS_MAX_PARALLEL)

        async def _tts_with_semaphore(params: TTSParams) -> TTSTimestampsResponse:
            async with semaphore:
                return await tts.tts_w_timestamps(params=params)

        tasks_for_tts = []
        for character_phrase, params in zip(text_split.phrases, tts_params_list):
            params.voice_id = character2voice[character_phrase.character]
            task = _tts_with_semaphore(params=params)
            tasks_for_tts.append(task)

        tts_responses: list[TTSTimestampsResponse] = await asyncio.gather(*tasks_for_tts)

        tts_audio_fps = []
        for ix, (params, res) in enumerate(zip(tts_params_list, tts_responses), start=1):
            out_fp_no_ext = os.path.join(out_dp, f'tts_output_{ix}')
            out_fp = res.write_audio_to_file(
                filepath_no_ext=out_fp_no_ext, audio_format=params.output_format
            )
            tts_audio_fps.append(out_fp)

        # combine alignments
        alignments = [response.alignment for response in tts_responses]
        char2time = TTSTimestampsAlignment.combine_alignments(alignments=alignments)

        return TTSPhrasesGenerationOutput(audio_fps=tts_audio_fps, char2time=char2time)

    @staticmethod
    def _update_sound_effects_descriptions_with_durations(
        sound_effects_descriptions: list[SoundEffectDescription],
        char2time: TTSTimestampsAlignment,
    ) -> list[SoundEffectDescription]:
        for sed in sound_effects_descriptions:
            ix_start, ix_end = sed.ix_start_llm_response, sed.ix_end_llm_response
            time_start = char2time.get_start_time_by_char_ix(ix_start, safe=True)
            time_end = char2time.get_end_time_by_char_ix(ix_end, safe=True)
            duration = time_end - time_start
            # update inplace
            sed.start_sec = time_start
            sed.duration_sec = duration
        return sound_effects_descriptions

    def _filter_short_sound_effects(
        self,
        sound_effects_descriptions: list[SoundEffectDescription],
    ) -> list[SoundEffectDescription]:
        filtered = [
            sed
            for sed in sound_effects_descriptions
            if sed.duration_sec > self.min_sound_effect_duration_sec
        ]

        len_orig = len(sound_effects_descriptions)
        len_new = len(filtered)
        logger.info(
            f'{len_new} out of {len_orig} original sound effects are kept '
            f'after filtering by min duration: {self.min_sound_effect_duration_sec}'
        )

        return filtered

    def _sound_effects_description_2_generation_params(
        self,
        sound_effects_descriptions: list[SoundEffectDescription],
    ) -> list[SoundEffectsParams]:
        params = [
            SoundEffectsParams(
                text=sed.prompt,
                duration_seconds=sed.duration_sec,
                prompt_influence=self.sound_effects_prompt_influence,
            )
            for sed in sound_effects_descriptions
        ]
        return params

    @staticmethod
    async def _generate_sound_effects(
        sound_effects_params: list[SoundEffectsParams],
        out_dp: str,
    ) -> list[str]:
        semaphore = asyncio.Semaphore(ELEVENLABS_MAX_PARALLEL)

        async def _se_gen_with_semaphore(params: SoundEffectsParams) -> list[bytes]:
            async with semaphore:
                return await tts.sound_generation_consumed(params=params)

        tasks = [_se_gen_with_semaphore(params=params) for params in sound_effects_params]
        results = await asyncio.gather(*tasks)

        se_fps = []
        for ix, task_res in enumerate(results, start=1):
            out_fp = os.path.join(out_dp, f'sound_effect_{ix}.wav')
            utils.write_chunked_bytes(data=task_res, fp=out_fp)
            se_fps.append(out_fp)

        return se_fps

    @staticmethod
    def _normalize_audio_files(
        audio_fps: list[str], out_dp: str, target_dBFS: float = -20.0
    ) -> list[str]:
        """Normalize all audio chunks to the target volume level."""
        fps = []
        for in_fp in audio_fps:
            audio_segment = AudioSegment.from_file(in_fp)
            normalized_audio = utils.normalize_audio(audio_segment, target_dBFS)

            out_fp = os.path.join(out_dp, f"{Path(in_fp).stem}.normalized.wav")
            normalized_audio.export(out_fp, format="wav")
            fps.append(out_fp)

        return fps

    @staticmethod
    def _concatenate_audiofiles(audio_fps: list[str], out_wav_fp: str):
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

            text_split = await self._split_text(text=text)

            # TODO: call sound effects chain in parallel with text split chain
            if generate_effects:
                designed_sound_effects = await self._design_sound_effects(text=text)

            select_voice_chain_out = await self._map_characters_to_voices(text_split=text_split)

            tts_params_list = await self._prepare_text_for_tts(text_split=text_split)

            tts_dp = os.path.join(out_dp_root, 'tts')
            os.makedirs(tts_dp)
            tts_out = await self._generate_tts_audio(
                text_split=text_split,
                tts_params_list=tts_params_list,
                character2voice=select_voice_chain_out.character2voice,
                out_dp=tts_dp,
            )

            if generate_effects:
                se_descriptions = designed_sound_effects.sound_effects_descriptions

                se_descriptions = self._update_sound_effects_descriptions_with_durations(
                    sound_effects_descriptions=se_descriptions, char2time=tts_out.char2time
                )

                se_descriptions = self._filter_short_sound_effects(
                    sound_effects_descriptions=se_descriptions
                )

                se_params = self._sound_effects_description_2_generation_params(
                    sound_effects_descriptions=se_descriptions
                )

                if len(se_descriptions) != len(se_params):
                    raise ValueError(
                        f'expected {len(se_descriptions)} sound effects params, got: {len(se_params)}'
                    )

                effects_dp = os.path.join(out_dp_root, 'sound_effects')
                os.makedirs(effects_dp)
                se_fps = await self._generate_sound_effects(
                    sound_effects_params=se_params, out_dp=effects_dp
                )

                if len(se_descriptions) != len(se_fps):
                    raise ValueError(
                        f'expected {len(se_descriptions)} generated sound effects, got: {len(se_fps)}'
                    )

            tts_normalized_dp = os.path.join(out_dp_root, 'tts_normalized')
            os.makedirs(tts_normalized_dp)
            tts_norm_fps = self._normalize_audio_files(
                audio_fps=tts_out.audio_fps, out_dp=tts_normalized_dp
            )

            if generate_effects:
                se_normalized_dp = os.path.join(out_dp_root, 'sound_effects_normalized')
                os.makedirs(se_normalized_dp)
                se_norm_fps = self._normalize_audio_files(audio_fps=se_fps, out_dp=se_normalized_dp)

                # TODO: apply fade-in and fade-out for each sound effect
                # TODO: decrease volume of overlayed sound effects
                # (prev pydub code used gain decrease of 5 units)

            tts_concat_fp = os.path.join(out_dp_root, f'audiobook_{now_str}.wav')
            self._concatenate_audiofiles(audio_fps=tts_norm_fps, out_wav_fp=tts_concat_fp)

            if not generate_effects:
                final_audio_fp = tts_concat_fp
            else:
                tts_concat_with_effects_fp = os.path.join(
                    out_dp_root, f'audiobook_with_effects_{now_str}.wav'
                )
                se_starts_sec = [sed.start_sec for sed in se_descriptions]
                utils.overlay_multiple_audio(
                    main_audio_fp=tts_concat_fp,
                    audios_to_overlay_fps=se_norm_fps,
                    starts_sec=se_starts_sec,
                    out_fp=tts_concat_with_effects_fp,
                )
                final_audio_fp = tts_concat_with_effects_fp

            utils.rm_dir_conditional(dp=out_dp_root, to_remove=self.rm_artifacts)

        logger.info(f'end of {self.name}.run(). openai callback stats: {cb}')
        return final_audio_fp
