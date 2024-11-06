import json

import openai
from elevenlabs import VoiceSettings

from src.config import (
    DEFAULT_TTS_SIMILARITY_BOOST,
    DEFAULT_TTS_STABILITY,
    DEFAULT_TTS_STABILITY_ACCEPTABLE_RANGE,
    DEFAULT_TTS_STYLE,
    OPENAI_API_KEY,
    logger,
)
from src.prompts import EMOTION_STABILITY_MODIFICATION
from src.schemas import TTSParams
from src.utils import GPTModels, auto_retry


class TTSParamProcessor:

    # TODO: refactor to langchain function (?)

    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

    @staticmethod
    def _wrap_results(data: dict, default_text: str) -> TTSParams:
        stability = data.get('stability', DEFAULT_TTS_STABILITY)
        stability = max(stability, DEFAULT_TTS_STABILITY_ACCEPTABLE_RANGE[0])
        stability = min(stability, DEFAULT_TTS_STABILITY_ACCEPTABLE_RANGE[1])

        similarity_boost = DEFAULT_TTS_SIMILARITY_BOOST
        style = DEFAULT_TTS_STYLE

        params = TTSParams(
            # NOTE: voice will be set later in the builder pipeline
            voice_id='',
            text=default_text,
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
                {"role": "system", "content": EMOTION_STABILITY_MODIFICATION},
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
            logger.exception(f"Error in parsing LLM output: '{chatgpt_output}'")
            raise e

        output_wrapped = self._wrap_results(output_dict, default_text=text_prepared)
        return output_wrapped
