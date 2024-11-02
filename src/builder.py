import asyncio

from langchain_community.callbacks import get_openai_callback

from src.audio_generators import AudioGeneratorWithEffects
from src.config import OPENAI_MAX_PARALLEL
from src.lc_callbacks import LCMessageLoggerAsync
from src.select_voice_chain import SelectVoiceChainOutput, VoiceSelector, CharacterPropertiesNullable
from src.text_split_chain import SplitTextOutput, create_split_text_chain
from src.utils import GPTModels


class AudiobookBuilder:
    def __init__(self):
        self.voice_selector = VoiceSelector()
        self.audio_generator = AudioGeneratorWithEffects()

    async def split_text(self, text: str) -> SplitTextOutput:
        chain = create_split_text_chain(llm_model=GPTModels.GPT_4o)
        with get_openai_callback() as cb:
            chain_out = await chain.ainvoke(
                {"text": text}, config={"callbacks": [LCMessageLoggerAsync()]}
            )
        return chain_out

    async def map_characters_to_voices(
        self, text_split: SplitTextOutput
    ) -> SelectVoiceChainOutput:
        chain = self.voice_selector.create_voice_mapping_chain(
            llm_model=GPTModels.GPT_4o
        )
        with get_openai_callback() as cb:
            chain_out = await chain.ainvoke(
                {
                    "text": text_split.text_annotated,
                    "characters": text_split.characters,
                },
                config={"callbacks": [LCMessageLoggerAsync()]},
            )
        return chain_out

    async def prepare_text_for_tts_with_voice_mapping(
        self, text_split: SplitTextOutput, generate_effects: bool, use_user_voice: bool, voice_id: str = None
    ):
        lines_for_sound_effect = self.audio_generator.select_lines_for_sound_effect(
            text_split, fraction=float(0.2 * generate_effects)
        )

        semaphore = asyncio.Semaphore(OPENAI_MAX_PARALLEL)

        tasks = []
        tasks.extend(
            [
                await self.audio_generator.create_emotion_text_task(
                    semaphore, phrase.text
                )
                for phrase in text_split.phrases
            ]
        )

        tasks.extend(
            [
                await self.audio_generator.create_effect_text_task(
                    semaphore, text_split.phrases[idx].text
                )
                for idx in lines_for_sound_effect
            ]
        )

        if not use_user_voice:
            tasks.append(
                asyncio.create_task(
                    self.audio_generator.process_single_task(
                        semaphore=semaphore,
                        func=self.map_characters_to_voices,
                        text_split=text_split,
                    )
                )
            )

        results = await asyncio.gather(*tasks)

        if not use_user_voice:
            prepared_texts = results[:-1]
            voice_mapping = results[-1]
        else:
            prepared_texts = results
            character2props = {char: CharacterPropertiesNullable(gender=None, age_group=None) for char in text_split.characters}
            character2voice = {char: voice_id for char in text_split.characters}
            voice_mapping = SelectVoiceChainOutput(
                character2props=character2props,
                character2voice=character2voice
            )

        emotion_texts = [x.output for x in prepared_texts if x.task == "add_emotion"]
        effects_texts = [x.output for x in prepared_texts if x.task == "add_effects"]

        return (
            emotion_texts,
            effects_texts,
            voice_mapping,
            lines_for_sound_effect,
        )

    async def run(self, text: str, generate_effects: bool, use_user_voice: bool, voice_id: str = None):
        text_split = await self.split_text(text)

        (
            data_for_tts,
            data_for_sound_effects,
            select_voice_chain_out,
            lines_for_sound_effect,
        ) = await self.prepare_text_for_tts_with_voice_mapping(
            text_split, generate_effects, use_user_voice, voice_id
        )

        out_path = await self.audio_generator.generate_audio(
            text_split=text_split,
            data_for_tts=data_for_tts,
            data_for_sound_effects=data_for_sound_effects,
            character_to_voice=select_voice_chain_out.character2voice,
            lines_for_sound_effect=lines_for_sound_effect,
        )
        return out_path
