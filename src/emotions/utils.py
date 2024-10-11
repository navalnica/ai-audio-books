from pydub import AudioSegment
from pathlib import Path
from elevenlabs import ElevenLabs, AsyncElevenLabs
from elevenlabs import play, save

from src.config import logger


def get_audio_duration(filepath: str) -> float:
    """
    Returns the duration of the audio file in seconds.

    :param filepath: Path to the audio file.
    :return: Duration of the audio file in seconds.
    """
    audio = AudioSegment.from_file(filepath)
    duration_in_seconds = len(audio) / 1000  # Convert milliseconds to seconds
    return round(duration_in_seconds, 1)


def add_overlay_for_audio(
    main_audio_filename: str,
    sound_effect_filename: str,
    output_filename: str = None,
    cycling_effect: bool = True,
    decrease_effect_volume: int = 0,
) -> str:
    try:
        main_audio = AudioSegment.from_file(main_audio_filename)
        effect_audio = AudioSegment.from_file(sound_effect_filename)
    except Exception as e:
        raise RuntimeError(f"Error loading audio files: {e}")

    if cycling_effect:
        while len(effect_audio) < len(main_audio):
            effect_audio += effect_audio

    effect_audio = effect_audio[: len(main_audio)]

    if decrease_effect_volume > 0:
        effect_audio = effect_audio - decrease_effect_volume
    combined_audio = main_audio.overlay(effect_audio)

    if output_filename is None:
        output_filename = (
            f"{Path(main_audio_filename).stem}_{Path(sound_effect_filename).stem}.wav"
        )
    combined_audio.export(output_filename, format="wav")
    return output_filename


def sound_generation(sound_generation_data: dict, output_file: str):
    client = ElevenLabs(
        api_key="YOUR_API_KEY",
    )
    audio = client.text_to_sound_effects.convert(
        text=sound_generation_data["text"],
        duration_seconds=sound_generation_data["duration_seconds"],
        prompt_influence=sound_generation_data["prompt_influence"],
    )
    save(audio, output_file)
    logger.error("Successfully generated sound effect to file: %s", output_file)


async def sound_generation_async(sound_generation_data: dict, output_file: str):
    client = AsyncElevenLabs(
        api_key="YOUR_API_KEY",
    )
    audio = await client.text_to_sound_effects.convert(
        text=sound_generation_data["text"],
        duration_seconds=sound_generation_data["duration_seconds"],
        prompt_influence=sound_generation_data["prompt_influence"],
    )
    save(audio, output_file)
    logger.error("Successfully generated sound effect to file: %s", output_file)
