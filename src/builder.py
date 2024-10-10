from langchain_community.callbacks import get_openai_callback

from src.audio_generators import AudioGeneratorSimple
from src.lc_callbacks import LCMessageLoggerAsync
from src.select_voice_chain import SelectVoiceChainOutput, VoiceSelector
from src.text_split_chain import SplitTextOutput, create_split_text_chain
from src.utils import GPTModels


class AudiobookBuilder:

    def __init__(self):
        self.voice_selector = VoiceSelector(
            csv_table_fp="data/11labs_available_tts_voices.csv"
        )
        self.audio_generator = AudioGeneratorSimple()

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

    async def run(self, text: str):
        text_split = await self.split_text(text)
        select_voice_chain_out = await self.map_characters_to_voices(
            text_split=text_split
        )
        # TODO: show select_voice_chain_out.character2props on UI
        out_path = await self.audio_generator.generate_audio(
            text_split=text_split,
            character_to_voice=select_voice_chain_out.character2voice,
        )
        return out_path
