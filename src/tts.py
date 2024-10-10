import os
import typing as t

from dotenv import load_dotenv
from elevenlabs.client import AsyncElevenLabs, ElevenLabs
from elevenlabs import VoiceSettings

load_dotenv()


ELEVEN_CLIENT = ElevenLabs(api_key=os.getenv("11LABS_API_KEY"))

ELEVEN_CLIENT_ASYNC = AsyncElevenLabs(api_key=os.getenv("11LABS_API_KEY"))


def tts_stream(voice_id: str, text: str) -> t.Iterator[bytes]:
    async_iter = ELEVEN_CLIENT.text_to_speech.convert(voice_id=voice_id, text=text)
    for chunk in async_iter:
        if chunk:
            yield chunk


def tts(voice_id: str, text: str):
    tts_iter = tts_stream(voice_id=voice_id, text=text)
    combined = b"".join(tts_iter)
    return combined


async def tts_astream(voice_id: str, text: str, params: dict) -> t.AsyncIterator[bytes]:
    async_iter = ELEVEN_CLIENT_ASYNC.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        voice_settings=VoiceSettings(
            stability=params['stability'],
            similarity_boost=params['similarity_boost'],
            style=params['style']
        ))
    async for chunk in async_iter:
        if chunk:
            yield chunk


async def sound_generation_astream(sound_generation_data: dict) -> t.AsyncIterator[bytes]:
    async_iter = ELEVEN_CLIENT_ASYNC.text_to_sound_effects.convert(
        text=sound_generation_data['text'],
        duration_seconds=sound_generation_data['duration_seconds'],
        prompt_influence=sound_generation_data['prompt_influence']
    )
    async for chunk in async_iter:
        if chunk:
            yield chunk