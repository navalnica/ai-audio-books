import asyncio
import os
from pathlib import Path
from typing import Callable, Any
from uuid import uuid4

from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel
from pydub import AudioSegment

from src.web.constructor import HTMLGenerator
from src.web.utils import (
    create_status_html,
    generate_full_text_split_without_effect_html,
    generate_full_text_split_with_effect_html,
    generate_voice_assignments_html,
)
from src import tts, utils
from src.config import ELEVENLABS_MAX_PARALLEL, OPENAI_MAX_PARALLEL, logger
from src.lc_callbacks import LCMessageLoggerAsync
from src.preprocess_tts_text_chain import TTSTextProcessorWithSSML
from src.schemas import SoundEffectsParams, TTSParams, TTSTimestampsAlignment, TTSTimestampsResponse
from src.select_voice_chain import (
    SelectVoiceChainOutput,
    VoiceSelector,
    CharacterPropertiesNullable,
)
from src.sound_effects_design import (
    SoundEffectDescription,
    SoundEffectsDesignOutput,
    create_sound_effects_design_chain,
)
from src.text_split_chain import SplitTextOutput, create_split_text_chain
from src.utils import GPTModels, replace_labels, get_audio_from_voice_id, get_character_color


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
        self.html_generator = HTMLGenerator()
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
    def _add_voice_ids_to_tts_params(
        text_split: SplitTextOutput,
        tts_params_list: list[TTSParams],
        character2voice: dict[str, str],
    ) -> list[TTSParams]:
        for character_phrase, params in zip(text_split.phrases, tts_params_list):
            params.voice_id = character2voice[character_phrase.character]
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

    @staticmethod
    def _update_sound_effects_descriptions_with_durations(
        sound_effects_descriptions: list[SoundEffectDescription],
        char2time: TTSTimestampsAlignment,
    ) -> list[SoundEffectDescription]:
        for sed in sound_effects_descriptions:
            ix_start, ix_end = sed.ix_start_orig_text, sed.ix_end_orig_text
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

    async def process_single_task(
        self, semaphore: asyncio.Semaphore, func: Callable, **params
    ) -> Any:
        async with semaphore:
            return await func(**params)

    async def create_emotion_text_task(
        self, semaphore: asyncio.Semaphore, text: str
    ) -> asyncio.Task:
        return asyncio.create_task(
            self.process_single_task(
                semaphore=semaphore,
                func=self.text_tts_processor.run,
                text=text,
            )
        )

    async def create_effect_text_task(
        self, semaphore: asyncio.Semaphore, text: str
    ) -> asyncio.Task:
        return asyncio.create_task(
            self.process_single_task(
                semaphore=semaphore,
                func=self._design_sound_effects,
                text=text,
            )
        )

    async def create_voice_mapping_task(
        self, semaphore: asyncio.Semaphore, text_split: SplitTextOutput
    ) -> asyncio.Task:
        return asyncio.create_task(
            self.process_single_task(
                semaphore=semaphore,
                func=self._map_characters_to_voices,
                text_split=text_split,
            )
        )

    async def prepare_text_for_tts_with_voice_mapping(
        self,
        full_text: str,
        text_split: SplitTextOutput,
        generate_effects: bool,
        use_user_voice: bool,
        voice_id: str = None,
    ):
        semaphore = asyncio.Semaphore(OPENAI_MAX_PARALLEL)

        # Create tasks for emotion processing
        emotion_tasks = [
            self.create_emotion_text_task(semaphore, phrase.text)
            for phrase in text_split.phrases
        ]

        # Create effect task if needed
        effect_task = None
        if generate_effects:
            effect_task = await self.create_effect_text_task(semaphore, full_text)

        # Create voice mapping task if needed
        voice_mapping_task = None
        if not use_user_voice:
            voice_mapping_task = await self.create_voice_mapping_task(
                semaphore=semaphore,
                text_split=text_split,
            )

        # Gather all tasks that need to be executed
        tasks_to_execute = [*emotion_tasks]
        if effect_task:
            tasks_to_execute.append(effect_task)
        if voice_mapping_task:
            tasks_to_execute.append(voice_mapping_task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks_to_execute)

        # Process results
        if generate_effects and not use_user_voice:
            # Both effects and voice mapping are present
            prepared_texts = results[:-2]
            effects = results[-2]
            voice_mapping = results[-1]
        elif generate_effects:
            # Only effects are present
            prepared_texts = results[:-1]
            effects = results[-1]
            voice_mapping = SelectVoiceChainOutput(
                character2props={
                    char: CharacterPropertiesNullable(gender=None, age_group=None)
                    for char in text_split.characters
                },
                character2voice={char: voice_id for char in text_split.characters},
            )
        elif not use_user_voice:
            # Only voice mapping is present
            prepared_texts = results[:-1]
            effects = None
            voice_mapping = results[-1]
        else:
            # Neither effects nor voice mapping
            prepared_texts = results
            effects = None
            voice_mapping = SelectVoiceChainOutput(
                character2props={
                    char: CharacterPropertiesNullable(gender=None, age_group=None)
                    for char in text_split.characters
                },
                character2voice={char: voice_id for char in text_split.characters},
            )

        return prepared_texts, effects, voice_mapping

    def get_effect_params(
        self, text_split: SplitTextOutput, se_design_output: SoundEffectsDesignOutput
    ):
        effects_texts = [effect.prompt for effect in se_design_output.sound_effects_descriptions]
        text_between_effects_texts = [
            effect.text_between_tags for effect in se_design_output.sound_effects_descriptions
        ]

        lines_for_sound_effect = []
        for text in text_between_effects_texts:
            for i, phrase in enumerate(text_split.phrases):
                if text.lower() in phrase.text.lower():
                    lines_for_sound_effect.append(i)

        return effects_texts, text_between_effects_texts, lines_for_sound_effect

    async def run(
        self,
        text: str,
        generate_effects: bool,
        use_user_voice: bool = False,
        voice_id: str = None,
    ):
        now_str = utils.get_utc_now_str()
        uuid_trimmed = str(uuid4()).split('-')[0]
        dir_name = f'{now_str}-{uuid_trimmed}'
        out_dp_root = os.path.join('data', 'audiobooks', dir_name)
        os.makedirs(out_dp_root, exist_ok=False)

        debug_dp = os.path.join(out_dp_root, 'debug')
        os.makedirs(debug_dp)

        if use_user_voice and not voice_id:
            yield None, "", self.html_generator.generate_message_without_voice_id()

        else:
            # zero stage
            yield None, "", self.html_generator.generate_status(
                "Starting Process", [("Splitting text into characters...", False)]
            )

            # first stage
            # with get_openai_callback() as cb:
            text_split = await self._split_text(text=text)
            text_split_dict_list = [item.model_dump() for item in text_split._phrases]
            for item in text_split_dict_list:
                item['character'] = replace_labels(item['character'])

            text_split_html = generate_full_text_split_without_effect_html(text_split_dict_list)
            text_split_result_html = self.html_generator.generate_text_split(text_split_html)
            status_html = create_status_html(
                "Text Analysis Complete",
                [("Text splitting", True), ("Mapping characters to voices...", False)],
            )
            first_stage_result_html = status_html + text_split_result_html
            yield None, "", first_stage_result_html

            # second stage
            (
                tts_params_list,
                se_design_output,
                select_voice_chain_out,
            ) = await self.prepare_text_for_tts_with_voice_mapping(
                text, text_split, generate_effects, use_user_voice, voice_id
            )

            if generate_effects:
                (
                    effects_texts,
                    text_between_effects_texts,
                    lines_for_sound_effect,
                ) = self.get_effect_params(text_split, se_design_output)
                text_split_html = generate_full_text_split_with_effect_html(
                    text_split_dict_list,
                    effects_texts,
                    text_between_effects_texts,
                )

            status_html = create_status_html(
                "Voice Mapping Complete",
                [("Text splitting", True), ("Voice mapping", True), ("Generating audio...", False)],
            )
            text_split_result_html = self.html_generator.generate_text_split(text_split_html)
            if not use_user_voice:
                voice_assignments_html = generate_voice_assignments_html(select_voice_chain_out)
                voice_assignments_result_html = self.html_generator.generate_voice_assignments(
                    voice_assignments_html
                )
            else:
                voice_assignments_result_html = ''
            second_stage_result_html = (
                status_html + text_split_result_html + voice_assignments_result_html + '</div>'
            )
            yield None, "", second_stage_result_html

            # third stage
            tts_params_list = self._add_voice_ids_to_tts_params(
                text_split=text_split,
                tts_params_list=tts_params_list,
                character2voice=select_voice_chain_out.character2voice,
            )

            tts_dp = os.path.join(out_dp_root, 'tts')
            os.makedirs(tts_dp)
            tts_out = await self._generate_tts_audio(tts_params_list=tts_params_list, out_dp=tts_dp)

            self._save_tts_debug_data(
                tts_params_list=tts_params_list, tts_out=tts_out, out_dp=debug_dp
            )

            if generate_effects:
                se_descriptions = se_design_output.sound_effects_descriptions

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

                self._save_sound_effects_debug_data(
                    sound_effect_design_output=se_design_output,
                    sound_effect_descriptions=se_descriptions,
                    out_dp=debug_dp,
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

            status_html = create_status_html(
                "Process Complete ✨",
                [("Text splitting", True), ("Voice mapping", True), ("Audio generation", True)],
            )
            third_stage_result_html = (
                status_html
                + text_split_result_html
                + voice_assignments_result_html
                + self.html_generator.generate_final_message()
                + '</div>'
            )
            yield final_audio_fp, "", third_stage_result_html
        # logger.info(f'end of the process. openai callback stats: {cb}')
