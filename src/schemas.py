from __future__ import annotations

import base64
import typing as t
from enum import StrEnum

import pandas as pd
from elevenlabs import VoiceSettings
from pydantic import BaseModel, ConfigDict, Field

from src import utils


class AudioOutputFormat(StrEnum):
    MP3_22050_32 = "mp3_22050_32"
    MP3_44100_32 = "mp3_44100_32"
    MP3_44100_64 = "mp3_44100_64"
    MP3_44100_96 = "mp3_44100_96"
    MP3_44100_128 = "mp3_44100_128"
    MP3_44100_192 = "mp3_44100_192"
    PCM_16000 = "pcm_16000"
    PCM_22050 = "pcm_22050"
    PCM_24000 = "pcm_24000"
    PCM_44100 = "pcm_44100"
    ULAW_8000 = "ulaw_8000"


class ExtraForbidModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# use Ellipsis to mark omitted function parameter.
# cast it to Any type to avoid warnings from type checkers
# exact same approach is used in elevenlabs client.
OMIT = t.cast(t.Any, ...)


class TTSParams(ExtraForbidModel):

    # NOTE: pydantic treats Ellipsis as a mark of a required field.
    # in order to set Ellipsis as actual field default value, we need to use workaround
    # and use Field's default_factory

    voice_id: str
    text: str
    # enable_logging: typing.Optional[bool] = None

    # NOTE: we opt for quality over speed - thus don't use this param
    # optimize_streaming_latency: typing.Optional[OptimizeStreamingLatency] = None

    # NOTE: here we set default different from 11labs API
    # output_format: AudioOutputFormat = AudioOutputFormat.MP3_44100_128
    output_format: AudioOutputFormat = AudioOutputFormat.MP3_44100_192

    # NOTE: pydantic has protected "model_" namespace.
    # here we use workaround to pass "model_id" param to 11labs client
    # via serialization_alias
    audio_model_id: t.Optional[str] = Field(
        default_factory=lambda: OMIT, serialization_alias="model_id"
    )

    language_code: t.Optional[str] = Field(default_factory=lambda: OMIT)
    voice_settings: t.Optional[VoiceSettings] = Field(default_factory=lambda: OMIT)
    # pronunciation_dictionary_locators: t.Optional[
    #     t.Sequence[PronunciationDictionaryVersionLocator]
    # ] = Field(default_factory=lambda: OMIT)
    seed: t.Optional[int] = Field(default_factory=lambda: OMIT)
    previous_text: t.Optional[str] = Field(default_factory=lambda: OMIT)
    next_text: t.Optional[str] = Field(default_factory=lambda: OMIT)
    previous_request_ids: t.Optional[t.Sequence[str]] = Field(default_factory=lambda: OMIT)
    next_request_ids: t.Optional[t.Sequence[str]] = Field(default_factory=lambda: OMIT)
    # request_options: t.Optional[RequestOptions] = None

    def to_dict(self):
        """
        dump the pydantic model in the format required by 11labs api.

        NOTE: we need to use `by_alias=True` in order to correctly handle
        alias for `audio_model_id` field,
        since model_id belongs to pydantic protected namespace.

        NOTE: we also ignore all fields with default Ellipsis value,
        since 11labs will assign Ellipses itself,
        and we won't get any warning in logs.
        """
        ellipsis_fields = {field for field, value in self if value is ...}
        res = self.model_dump(by_alias=True, exclude=ellipsis_fields)
        return res


class TTSTimestampsAlignemnt(ExtraForbidModel):
    characters: list[str]
    character_start_times_seconds: list[float]
    character_end_times_seconds: list[float]
    _text_joined: str

    def __init__(self, **data):
        super().__init__(**data)
        self._text_joined = "".join(self.characters)

    @property
    def text_joined(self):
        return self._text_joined

    def to_dataframe(self):
        return pd.DataFrame(
            {
                "char": self.characters,
                "start": self.character_start_times_seconds,
                "end": self.character_end_times_seconds,
            }
        )

    @classmethod
    def combine_alignments(
        cls,
        alignments: list[TTSTimestampsAlignemnt],
        pause_bw_chunks_s: float = 0.2,
    ) -> TTSTimestampsAlignemnt:
        """
        Combine alignemnts created for different TTS phrases in a single aligment for a whole text.

        NOTE: while splitting original text into character phrases,
        we ignore separators between phrases.
        They may be different: single or multiple spaces, newlines, etc.
        To account for them we insert fixed pause and characters between phrases in final alignment.
        This will give use an approximation of a real timestamp mapping
        for voicing a whole original text.

        NOTE: The quality of such approximation seems appropriate,
        considering the amount of time required to implement more accurate mapping.
        """

        chars = []
        starts = []
        ends = []
        prev_chunk_end_time = 0.0
        n_alignments = len(alignments)

        for ix, a in enumerate(alignments):
            cur_starts_absolute = [prev_chunk_end_time + s for s in a.character_start_times_seconds]
            cur_ends_absolute = [prev_chunk_end_time + e for e in a.character_end_times_seconds]

            chars.extend(a.characters)
            starts.extend(cur_starts_absolute)
            ends.extend(cur_ends_absolute)

            if ix < n_alignments - 1:
                chars.append('#')
                placeholder_start = cur_ends_absolute[-1]
                starts.append(placeholder_start)
                ends.append(placeholder_start + pause_bw_chunks_s)

            prev_chunk_end_time = ends[-1]

        return TTSTimestampsAlignemnt(
            characters=chars,
            character_start_times_seconds=starts,
            character_end_times_seconds=ends,
        )

    def get_start_time_by_char_ix(self, char_ix: int):
        return self.character_start_times_seconds[char_ix]

    def get_end_time_by_char_ix(self, char_ix: int):
        return self.character_end_times_seconds[char_ix]


class TTSTimestampsResponse(ExtraForbidModel):
    audio_base64: str
    alignment: TTSTimestampsAlignemnt
    normalized_alignment: TTSTimestampsAlignemnt

    @property
    def audio_bytes(self):
        return base64.b64decode(self.audio_base64)

    def write_audio_to_file(self, filepath_no_ext: str, audio_format: AudioOutputFormat) -> str:
        if audio_format.startswith("pcm_"):
            sr = int(audio_format.removeprefix("pcm_"))
            fp = f"{filepath_no_ext}.wav"
            utils.write_raw_pcm_to_file(
                data=self.audio_bytes,
                fp=fp,
                n_channels=1,  # seems like it's 1 channel always
                bytes_depth=2,  # seems like it's 2 bytes always
                sampling_rate=sr,
            )
            return fp
        elif audio_format.startswith("mp3_"):
            fp = f"{filepath_no_ext}.mp3"
            # received mp3 seems to already contain all required metadata
            # like sampling rate
            # and sample width
            utils.write_bytes(data=self.audio_bytes, fp=fp)
            return fp
        else:
            raise ValueError(f"don't know how to write audio format: {audio_format}")


class SoundEffectsParams(ExtraForbidModel):
    text: str
    duration_seconds: float | None
    prompt_influence: float | None
