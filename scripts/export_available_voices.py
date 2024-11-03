import logging
import os

import click
import pandas as pd
from dotenv import load_dotenv
from elevenlabs import ElevenLabs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s): %(message)s",
)
logger = logging.getLogger("export-available-voices")


load_dotenv()


@click.command()
@click.option("-ak", "--api-key", envvar="ELEVEN_LABS_API_KEY")
@click.option("-o", "--output-csv-path", default="data/11labs_available_tts_voices.csv")
def main(*, api_key: str | None, output_csv_path: str) -> None:
    if api_key is None:
        raise OSError("Who's gonna set the `ELEVEN_LABS_API_KEY` environmental variable?")

    client = ElevenLabs(api_key=api_key)
    response = client.voices.get_all()
    available_voices = pd.DataFrame.from_records(
        [
            voice.model_dump(
                include={
                    "voice_id",
                    "name",
                    "language",
                    "labels",
                    "description",
                    "preview_url",
                },
            )
            for voice in response.voices
        ]
    )
    available_voices = pd.concat(
        (
            available_voices.drop(
                columns=[
                    "labels",
                    "description",
                    "available_for_tiers",
                    "settings",
                    "sharing",
                    "high_quality_base_model_ids",
                    "safety_control",
                    "voice_verification",
                    "category",
                    "samples",
                ]
            ),
            pd.DataFrame.from_records(available_voices["labels"]).rename(
                columns={"use_case": "category"}
            ),
        ),
        axis=1,
    )

    available_voices.drop(columns="fine_tuning").to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    main()
