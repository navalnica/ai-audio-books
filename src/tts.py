import typing as t
from copy import deepcopy

from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import AsyncElevenLabs

load_dotenv()

from src.config import ELEVENLABS_API_KEY, logger
from src.schemas import SoundEffectsParams, TTSParams, TTSTimestampsResponse
from src.utils import auto_retry

ELEVEN_CLIENT_ASYNC = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)


async def tts_astream(
    voice_id: str, text: str, params: dict | None = None
) -> t.AsyncIterator[bytes]:
    params_all = dict(voice_id=voice_id, text=text)

    if params is not None:
        params_all["voice_settings"] = VoiceSettings(  # type: ignore
            stability=params.get("stability"),
            similarity_boost=params.get("similarity_boost"),
            style=params.get("style"),
        )

    logger.info(
        f"request to 11labs TTS endpoint with params {params_all} "
        f'for the following text: "{text}"'
    )
    async_iter = ELEVEN_CLIENT_ASYNC.text_to_speech.convert(**params_all)
    async for chunk in async_iter:
        if chunk:
            yield chunk


@auto_retry
async def tts_astream_consumed(voice_id: str, text: str, params: dict | None = None) -> list[bytes]:
    aiterator = tts_astream(voice_id=voice_id, text=text, params=params)
    return [x async for x in aiterator]


@auto_retry
async def tts_w_timestamps(params: TTSParams) -> TTSTimestampsResponse:

    async def _tts_w_timestamps(params: TTSParams) -> TTSTimestampsResponse:
        # NOTE: we need to use special `to_dict()` method to ensure pydantic model is converted
        # to dict with proper aliases
        params_dict = params.to_dict()

        params_no_text = deepcopy(params_dict)
        text = params_no_text.pop('text')
        logger.info(
            f"request to 11labs TTS endpoint with params {params_no_text} "
            f'for the following text: "{text}"'
        )

        response_raw = await ELEVEN_CLIENT_ASYNC.text_to_speech.convert_with_timestamps(
            **params_dict
        )

        response_parsed = TTSTimestampsResponse.model_validate(response_raw)
        return response_parsed

    res = await _tts_w_timestamps(params=params)
    return res


async def sound_generation_astream(params: SoundEffectsParams) -> t.AsyncIterator[bytes]:
    params_no_text = params.model_dump(exclude={"text"})
    logger.info(
        f"request to 11labs sound effect generation with params {params_no_text} "
        f'for the following text: "{params.text}"'
    )

    async_iter = ELEVEN_CLIENT_ASYNC.text_to_sound_effects.convert(
        text=params.text,
        duration_seconds=params.duration_seconds,
        prompt_influence=params.prompt_influence,
    )
    async for chunk in async_iter:
        if chunk:
            yield chunk


@auto_retry
async def sound_generation_consumed(params: SoundEffectsParams):
    aiterator = sound_generation_astream(params=params)
    return [x async for x in aiterator]
