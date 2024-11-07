import asyncio
import os
from asyncio import TaskGroup
from pathlib import Path
from typing import Any, Callable, List
from uuid import uuid4

from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel
from pydub import AudioSegment

from src import tts, utils
from src.config import (
    CONTEXT_CHAR_LEN_FOR_TTS,
    ELEVENLABS_MAX_PARALLEL,
    OPENAI_MAX_PARALLEL,
    logger,
)
from src.lc_callbacks import LCMessageLoggerAsync
from src.preprocess_tts_emotions_chain import TTSParamProcessor
from src.schemas import SoundEffectsParams, TTSParams, TTSTimestampsAlignment, TTSTimestampsResponse
from src.select_voice_chain import (
    CharacterPropertiesNullable,
    SelectVoiceChainOutput,
    VoiceSelector,
)
from src.sound_effects_design import (
    SoundEffectDescription,
    SoundEffectsDesignOutput,
    create_sound_effects_design_chain,
)
from src.text_modification_chain import modify_text_chain
from src.text_split_chain import SplitTextOutput, create_split_text_chain
from src.utils import GPTModels, prettify_unknown_character_label
from src.web.constructor import HTMLGenerator
from src.web.utils import (
    create_status_html,
    generate_text_split_inner_html_no_effect,
    generate_text_split_inner_html_with_effects,
    generate_voice_mapping_inner_html,
)


class TTSPhrasesGenerationOutput(BaseModel):
    audio_fps: list[str]
    char2time: TTSTimestampsAlignment


class AudiobookBuilder:
    def __init__(self, rm_artifacts: bool = False):
        self.voice_selector = VoiceSelector()
        self.params_tts_processor = TTSParamProcessor()
        self.rm_artifacts = rm_artifacts
        self.min_sound_effect_duration_sec = 1
        self.sound_effects_prompt_influence = 0.75  # seems to work nicely
        self.html_generator = HTMLGenerator()
        self.name = type(self).__name__

    @staticmethod
    async def _prepare_text_for_tts(text: str) -> str:
        chain = modify_text_chain(llm_model=GPTModels.GPT_4o)
        with get_openai_callback() as cb:
            result = await chain.ainvoke(
                {"text": text}, config={"callbacks": [LCMessageLoggerAsync()]}
            )
        logger.info(
            f'End of modifying text with caps and symbols(?, !, ...). Openai callback stats: {cb}'
        )
        return result.text_modified

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

    async def _prepare_params_for_tts(self, text_split: SplitTextOutput) -> list[TTSParams]:
        semaphore = asyncio.Semaphore(OPENAI_MAX_PARALLEL)

        async def run_task_with_semaphore(func, **params):
            async with semaphore:
                outputs = await func(**params)
                return outputs

        tasks = []

        for character_phrase in text_split.phrases:
            tasks.append(
                run_task_with_semaphore(
                    func=self.params_tts_processor.run,
                    text=character_phrase.text,
                )
            )

        tts_tasks_results = await asyncio.gather(*tasks)

        return tts_tasks_results

    @staticmethod
    def _add_voice_ids_to_tts_params(
        text_split: SplitTextOutput,
        tts_params_list: list[TTSParams],
        character2voice: dict[str, str],
    ) -> list[TTSParams]:
        for character_phrase, params in zip(text_split.phrases, tts_params_list):
            params.voice_id = character2voice[character_phrase.character]
        return tts_params_list

    @staticmethod
    def _get_left_and_right_contexts_for_each_phrase(
        phrases, context_length=CONTEXT_CHAR_LEN_FOR_TTS
    ):
        """
        Return phrases from left and right sides which don't exceed `context_length`.
        Approx. number of words/tokens based on `context_length` can be calculated by dividing it by 5.
        """
        # TODO: split first context phrase if it exceeds `context_length`, currently it's not added.
        # TODO: optimize algorithm to linear time using sliding window on top of cumulative length sums.
        left_right_contexts = []
        for i in range(len(phrases)):
            left_text, right_text = '', ''
            for j in range(i - 1, -1, -1):
                if len(left_text) + len(phrases[j].text) < context_length:
                    left_text = phrases[j].text + left_text
                else:
                    break
            for phrase in phrases[i + 1 :]:
                if len(right_text) + len(phrase.text) < context_length:
                    right_text += phrase.text
                else:
                    break
            left_right_contexts.append((left_text, right_text))
        return left_right_contexts

    def _add_previous_and_next_context_to_tts_params(
        self,
        text_split: SplitTextOutput,
        tts_params_list: list[TTSParams],
    ) -> list[TTSParams]:
        left_right_contexts = self._get_left_and_right_contexts_for_each_phrase(text_split.phrases)
        for cur_contexts, params in zip(left_right_contexts, tts_params_list):
            left_context, right_context = cur_contexts
            params.previous_text = left_context
            params.next_text = right_context
        return tts_params_list

    @staticmethod
    async def _generate_tts_audio(
        tts_params_list: list[TTSParams],
        out_dp: str,
    ) -> TTSPhrasesGenerationOutput:
        semaphore = asyncio.Semaphore(ELEVENLABS_MAX_PARALLEL)

        async def _tts_with_semaphore(params: TTSParams) -> TTSTimestampsResponse:
            async with semaphore:
                return await tts.tts_w_timestamps(params=params)

        tasks = [_tts_with_semaphore(params=params) for params in tts_params_list]
        tts_responses: list[TTSTimestampsResponse] = await asyncio.gather(*tasks)

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
        # filter alignments
        char2time = char2time.filter_chars_without_duration()

        return TTSPhrasesGenerationOutput(audio_fps=tts_audio_fps, char2time=char2time)

    def _update_sound_effects_descriptions_with_durations(
        self,
        sound_effects_descriptions: list[SoundEffectDescription],
        char2time: TTSTimestampsAlignment,
    ) -> list[SoundEffectDescription]:
        for sed in sound_effects_descriptions:
            ix_start, ix_end = sed.ix_start_orig_text, sed.ix_end_orig_text
            time_start = char2time.get_start_time_by_char_ix(ix_start, safe=True)
            time_end = char2time.get_end_time_by_char_ix(ix_end, safe=True)
            duration = time_end - time_start
            # apply min effect duration
            duration = max(self.min_sound_effect_duration_sec, duration)
            # update inplace
            sed.start_sec = time_start
            sed.duration_sec = duration
        return sound_effects_descriptions

    # def _filter_short_sound_effects(
    #     self,
    #     sound_effects_descriptions: list[SoundEffectDescription],
    # ) -> list[SoundEffectDescription]:
    #     filtered = [
    #         sed
    #         for sed in sound_effects_descriptions
    #         if sed.duration_sec > self.min_sound_effect_duration_sec
    #     ]

    #     len_orig = len(sound_effects_descriptions)
    #     len_new = len(filtered)
    #     logger.info(
    #         f'{len_new} out of {len_orig} original sound effects are kept '
    #         f'after filtering by min duration: {self.min_sound_effect_duration_sec}'
    #     )

    #     return filtered

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
    def _save_text_split_debug_data(
        text_split: SplitTextOutput,
        out_dp: str,
    ):
        out_fp = os.path.join(out_dp, 'text_split.json')
        # NOTE: use `to_dict()` for correct conversion
        data = text_split.model_dump()
        utils.write_json(data, fp=out_fp)

    @staticmethod
    def _save_tts_debug_data(
        tts_params_list: list[TTSParams],
        tts_out: TTSPhrasesGenerationOutput,
        out_dp: str,
    ):
        out_fp = os.path.join(out_dp, 'tts.json')
        # NOTE: use `to_dict()` for correct conversion
        data = [param.to_dict() for param in tts_params_list]
        utils.write_json(data, fp=out_fp)

        out_dp = os.path.join(out_dp, 'tts_char2time.csv')
        df_char2time = tts_out.char2time.to_dataframe()
        df_char2time.to_csv(out_dp, index=True)

    @staticmethod
    def _save_sound_effects_debug_data(
        sound_effect_design_output: SoundEffectsDesignOutput,
        sound_effect_descriptions: list[SoundEffectDescription],
        out_dp: str,
    ):
        out_fp = os.path.join(out_dp, 'sound_effects_raw_llm_output.txt')
        utils.write_txt(sound_effect_design_output.text_annotated, fp=out_fp)

        out_fp = os.path.join(out_dp, 'sound_effects_descriptions.json')
        data = [sed.model_dump() for sed in sound_effect_descriptions]
        utils.write_json(data, fp=out_fp)

    @staticmethod
    def _postprocess_tts_audio(audio_fps: list[str], out_dp: str, target_dBFS: float) -> list[str]:
        fps = []
        for in_fp in audio_fps:
            audio_segment = AudioSegment.from_file(in_fp)
            normalized_audio = utils.normalize_audio(audio_segment, target_dBFS)

            out_fp = os.path.join(out_dp, f"{Path(in_fp).stem}.normalized.wav")
            normalized_audio.export(out_fp, format="wav")
            fps.append(out_fp)

        return fps

    @staticmethod
    def _postprocess_sound_effects(
        audio_fps: list[str], out_dp: str, target_dBFS: float, fade_ms: int
    ) -> list[str]:
        fps = []
        for in_fp in audio_fps:
            audio_segment = AudioSegment.from_file(in_fp)

            processed = utils.normalize_audio(audio_segment, target_dBFS)

            processed = processed.fade_in(duration=fade_ms)
            processed = processed.fade_out(duration=fade_ms)

            out_fp = os.path.join(out_dp, f"{Path(in_fp).stem}.postprocessed.wav")
            processed.export(out_fp, format="wav")
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

    def _get_text_split_html(
        self,
        text_split: SplitTextOutput,
        sound_effects_descriptions: list[SoundEffectDescription] | None,
    ):
        # modify copies of original phrases, keep original intact
        character_phrases = [p.model_copy(deep=True) for p in text_split.phrases]
        for phrase in character_phrases:
            phrase.character = prettify_unknown_character_label(phrase.character)

        if not sound_effects_descriptions:
            inner = generate_text_split_inner_html_no_effect(character_phrases=character_phrases)
        else:
            inner = generate_text_split_inner_html_with_effects(
                character_phrases=character_phrases,
                sound_effects_descriptions=sound_effects_descriptions,
            )

        final = self.html_generator.generate_text_split(inner)
        return final

    def _get_voice_mapping_html(
        self, use_user_voice: bool, select_voice_chain_out: SelectVoiceChainOutput
    ):
        if use_user_voice:
            return ''
        inner = generate_voice_mapping_inner_html(select_voice_chain_out)
        final = self.html_generator.generate_voice_assignments(inner)
        return final

    STAGE_1 = 'Text Analysis'
    STAGE_2 = 'Voices Selection'
    STAGE_3 = 'Audio Generation'

    def _get_yield_data_stage_0(self):
        status = self.html_generator.generate_status("Starting", [("Analyzing Text...", False)])
        return None, "", status

    def _get_yield_data_stage_1(self, text_split_html: str):
        status_html = create_status_html(
            "Text Analysis Complete",
            [(self.STAGE_1, True), ("Selecting Voices...", False)],
        )
        html = status_html + text_split_html
        return None, "", html

    def _get_yield_data_stage_2(self, text_split_html: str, voice_mapping_html: str):
        status_html = create_status_html(
            "Voice Selection Complete",
            [(self.STAGE_1, True), (self.STAGE_2, True), ("Generating Audio...", False)],
        )
        html = status_html + text_split_html + voice_mapping_html + '</div>'
        return None, "", html

    def _get_yield_data_stage_3(
        self, final_audio_fp: str, text_split_html: str, voice_mapping_html: str
    ):
        status_html = create_status_html(
            "Audiobook is ready ✨",
            [(self.STAGE_1, True), (self.STAGE_2, True), (self.STAGE_3, True)],
        )
        third_stage_result_html = (
            status_html
            + text_split_html
            + voice_mapping_html
            + self.html_generator.generate_final_message()
            + '</div>'
        )
        return final_audio_fp, "", third_stage_result_html

    async def run(
        self,
        text: str,
        generate_effects: bool,
        use_user_voice: bool = False,
        voice_id: str | None = None,
    ):
        now_str = utils.get_utc_now_str()
        uuid_trimmed = str(uuid4()).split('-')[0]
        dir_name = f'{now_str}-{uuid_trimmed}'
        out_dp_root = os.path.join('data', 'audiobooks', dir_name)
        os.makedirs(out_dp_root, exist_ok=False)

        debug_dp = os.path.join(out_dp_root, 'debug')
        os.makedirs(debug_dp)

        # TODO: currently, we are constantly writing and reading audio segments from files.
        # I think it will be more efficient to keep all audio in memory.

        # zero stage
        if use_user_voice and not voice_id:
            yield None, "", self.html_generator.generate_message_without_voice_id()

        else:
            yield self._get_yield_data_stage_0()

            text_for_tts = await self._prepare_text_for_tts(text=text)

            # TODO: call sound effects chain in parallel with text split chain
            text_split = await self._split_text(text=text_for_tts)
            self._save_text_split_debug_data(text_split=text_split, out_dp=debug_dp)
            # yield stage 1
            text_split_html = self._get_text_split_html(
                text_split=text_split, sound_effects_descriptions=None
            )
            yield self._get_yield_data_stage_1(text_split_html=text_split_html)

            if generate_effects:
                se_design_output = await self._design_sound_effects(text=text_for_tts)
                se_descriptions = se_design_output.sound_effects_descriptions
                text_split_html = self._get_text_split_html(
                    text_split=text_split, sound_effects_descriptions=se_descriptions
                )

            # TODO: run voice mapping and tts params selection in parallel
            if not use_user_voice:
                select_voice_chain_out = await self._map_characters_to_voices(text_split=text_split)
            else:
                if voice_id is None:
                    raise ValueError(f'voice_id is None')
                select_voice_chain_out = SelectVoiceChainOutput(
                    character2props={
                        char: CharacterPropertiesNullable(gender=None, age_group=None)
                        for char in text_split.characters
                    },
                    character2voice={char: voice_id for char in text_split.characters},
                )
            tts_params_list = await self._prepare_params_for_tts(text_split=text_split)

            # yield stage 2
            voice_mapping_html = self._get_voice_mapping_html(
                use_user_voice=use_user_voice, select_voice_chain_out=select_voice_chain_out
            )
            yield self._get_yield_data_stage_2(
                text_split_html=text_split_html, voice_mapping_html=voice_mapping_html
            )

            tts_params_list = self._add_voice_ids_to_tts_params(
                text_split=text_split,
                tts_params_list=tts_params_list,
                character2voice=select_voice_chain_out.character2voice,
            )

            tts_params_list = self._add_previous_and_next_context_to_tts_params(
                text_split=text_split,
                tts_params_list=tts_params_list,
            )

            tts_dp = os.path.join(out_dp_root, 'tts')
            os.makedirs(tts_dp)
            tts_out = await self._generate_tts_audio(tts_params_list=tts_params_list, out_dp=tts_dp)

            self._save_tts_debug_data(
                tts_params_list=tts_params_list, tts_out=tts_out, out_dp=debug_dp
            )

            if generate_effects:
                se_descriptions = self._update_sound_effects_descriptions_with_durations(
                    sound_effects_descriptions=se_descriptions, char2time=tts_out.char2time
                )

                # no need in filtering, since we ensure the min duration above
                # se_descriptions = self._filter_short_sound_effects(
                #     sound_effects_descriptions=se_descriptions
                # )

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

                self._save_sound_effects_debug_data(
                    sound_effect_design_output=se_design_output,
                    sound_effect_descriptions=se_descriptions,
                    out_dp=debug_dp,
                )

            tts_normalized_dp = os.path.join(out_dp_root, 'tts_normalized')
            os.makedirs(tts_normalized_dp)
            tts_norm_fps = self._postprocess_tts_audio(
                audio_fps=tts_out.audio_fps,
                out_dp=tts_normalized_dp,
                target_dBFS=-20,
            )

            if generate_effects:
                se_normalized_dp = os.path.join(out_dp_root, 'sound_effects_postprocessed')
                os.makedirs(se_normalized_dp)
                se_norm_fps = self._postprocess_sound_effects(
                    audio_fps=se_fps,
                    out_dp=se_normalized_dp,
                    target_dBFS=-27,
                    fade_ms=500,
                )

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

            # yield stage 3
            yield self._get_yield_data_stage_3(
                final_audio_fp=final_audio_fp,
                text_split_html=text_split_html,
                voice_mapping_html=voice_mapping_html,
            )

        logger.info(f'end of {self.name}.run()')
