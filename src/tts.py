import typing as t

from dotenv import load_dotenv
from elevenlabs.client import AsyncElevenLabs, ElevenLabs
from elevenlabs import VoiceSettings

load_dotenv()

from src.config import logger, ELEVENLABS_API_KEY
from src.utils import auto_retry

ELEVEN_CLIENT = ElevenLabs(api_key=ELEVENLABS_API_KEY)

ELEVEN_CLIENT_ASYNC = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)


def tts_stream(voice_id: str, text: str) -> t.Iterator[bytes]:
    async_iter = ELEVEN_CLIENT.text_to_speech.convert(voice_id=voice_id, text=text)
    for chunk in async_iter:
        if chunk:
            yield chunk


def tts(voice_id: str, text: str):
    tts_iter = tts_stream(voice_id=voice_id, text=text)
    combined = b"".join(tts_iter)
    return combined


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
async def tts_astream_consumed(
    voice_id: str, text: str, params: dict | None = None
) -> list[bytes]:
    aiterator = tts_astream(voice_id=voice_id, text=text, params=params)
    return [x async for x in aiterator]


async def sound_generation_astream(
    sound_generation_data: dict,
) -> t.AsyncIterator[bytes]:
    text = sound_generation_data.pop("text")
    logger.info(
        f"request to 11labs sound effect generation with params {sound_generation_data} "
        f'for the following text: "{text}"'
    )

    async_iter = ELEVEN_CLIENT_ASYNC.text_to_sound_effects.convert(
        text=text,
        duration_seconds=sound_generation_data["duration_seconds"],
        prompt_influence=sound_generation_data["prompt_influence"],
    )
    async for chunk in async_iter:
        if chunk:
            yield chunk


@auto_retry
async def sound_generation_consumed(sound_generation_data: dict):
    aiterator = sound_generation_astream(sound_generation_data=sound_generation_data)
    return [x async for x in aiterator]
