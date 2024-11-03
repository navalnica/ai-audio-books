import logging

import click
import pandas as pd
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from elevenlabs.core import ApiError
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s): %(message)s",
)
logger = logging.getLogger("add-voices")


load_dotenv()


@click.command()
@click.option("-ak", "--api-key", envvar="ELEVEN_LABS_API_KEY")
@click.option("-i", "--input-csv-path", default="data/11labs_tts_voices.csv")
def main(*, api_key: str | None, input_csv_path: str) -> None:
    if api_key is None:
        raise OSError("Who's gonna set the `ELEVEN_LABS_API_KEY` environmental variable?")

    client = ElevenLabs(api_key=api_key)
    voices_to_import = pd.read_csv(input_csv_path)

    for _, row in tqdm(voices_to_import.iterrows(), total=len(voices_to_import)):
        try:
            client.voices.add_sharing_voice(
                public_user_id=(public_user_id := row["public_owner_id"]),
                voice_id=(voice_id := row["voice_id"]),
                new_name=(name := row["name"]),
            )
        except ApiError:
            logger.error(
                f"Shared voice with `{public_user_id = }`, `{voice_id = }` " "already added."
            )
        else:
            logger.info(
                f"Added shared voice with `{public_user_id = }`, `{voice_id = }`, " f"`{name = }`."
            )


if __name__ == "__main__":
    main()
