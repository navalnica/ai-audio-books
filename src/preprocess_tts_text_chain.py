import json

import openai
from elevenlabs import VoiceSettings

from src.config import OPENAI_API_KEY, logger
from src.schemas import TTSParams
from src.utils import GPTModels, auto_retry

from .prompts import TEXT_MODIFICATION_WITH_SSML


class TTSTextProcessorWithSSML:

    # TODO: refactor to langchain function (?)

    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

    @staticmethod
    def _wrap_results(data: dict, default_text: str) -> TTSParams:
        modified_text = data.get('modified_text', default_text)
        voice_params = data.get('params', {})
        stability = voice_params.get('stability', 0.5)
        similarity_boost = voice_params.get('similarity_boost', 0.5)
        style = voice_params.get('style', 0.5)

        params = TTSParams(
            # NOTE: voice will be set later in the builder pipeline
            voice_id='',
            text=modified_text,
            # reference: https://elevenlabs.io/docs/speech-synthesis/voice-settings
            voice_settings=VoiceSettings(
                stability=stability,
                similarity_boost=similarity_boost,
                style=style,
                use_speaker_boost=False,
            ),
        )
        return params

    @auto_retry
    async def run(self, text: str) -> TTSParams:
        text_prepared = text.strip()

        completion = await self.client.chat.completions.create(
            model=GPTModels.GPT_4o,
            messages=[
                {"role": "system", "content": TEXT_MODIFICATION_WITH_SSML},
                {"role": "user", "content": text_prepared},
            ],
            response_format={"type": "json_object"},
        )
        chatgpt_output = completion.choices[0].message.content
        if chatgpt_output is None:
            raise ValueError(f'received None as openai response content')

        try:
            output_dict = json.loads(chatgpt_output)
            logger.info(f"TTS text processing succeeded: {output_dict}")
        except json.JSONDecodeError as e:
            logger.exception("Error in parsing the modified text")
            raise ValueError(f"error, output_text: {chatgpt_output}") from e

        output_wrapped = self._wrap_results(output_dict, default_text=text_prepared)
        return output_wrapped
