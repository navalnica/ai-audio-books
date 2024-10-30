import os
from pathlib import Path
from typing import List

import gradio as gr
from altair.vegalite.v5.theme import theme
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

from src.utils import get_audio_from_voice_id

load_dotenv()

from src.builder import AudiobookBuilder
from src.config import logger, FILE_SIZE_MAX, MAX_TEXT_LEN, GRADIO_THEME, DESCRIPTION_JS
from data import samples_to_split as samples

from enum import StrEnum


class StatusSections(StrEnum):
    TEXT_SPLIT_BY_CHARACTER = "Text Split by Characters"


def get_auth_params() -> tuple[str, str]:
    user = os.environ["AUTH_USER"]
    password = os.environ["AUTH_PASS"]
    return user, password


def parse_pdf(file_path):
    """Parse the PDF file and return the text content."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return "\n".join([doc.page_content for doc in documents])


def load_text_from_file(uploaded_file):
    # Save the uploaded file temporarily to check its size
    temp_file_path = uploaded_file.name

    if os.path.getsize(temp_file_path) > FILE_SIZE_MAX * 1024 * 1024:
        raise ValueError(
            f"The uploaded file exceeds the size limit of {FILE_SIZE_MAX} MB."
        )

    if uploaded_file.name.endswith(".txt"):
        with open(temp_file_path, "r", encoding="utf-8") as file:
            text = file.read()
    elif uploaded_file.name.endswith(".pdf"):
        text = parse_pdf(temp_file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a .txt or .pdf file.")

    return text


async def respond(
    text: str,
    uploaded_file,
    generate_effects: bool,
) -> tuple[Path | None, str, str]:
    if uploaded_file is not None:
        try:
            text = load_text_from_file(uploaded_file=uploaded_file)
        except Exception as e:
            logger.exception(e)
            yield None, str(e), "### Error\nFailed to process file."

    if (text_len := len(text)) > MAX_TEXT_LEN:
        gr.Warning(
            f"Input text length of {text_len} characters "
            f"exceeded current limit of {MAX_TEXT_LEN} characters. "
            "Please input a shorter text."
        )
        yield None, "", "### Error\nText too long. Please input a shorter text."

    # Initial status
    yield None, "", """
### Status: Starting Process
🔄 Splitting text into characters...
"""

    builder = AudiobookBuilder()
    text_split = await builder.split_text(text)
    text_split_dict_list = [item.model_dump() for item in text_split._phrases]

    # Create character list markdown
    text_split_by_character = "\n".join(
        f"- **{item['character'] or 'Unassigned'}**: {item['text']}"
        for item in text_split_dict_list
    )

    yield None, "", f"""
### Status: Text Analysis Complete
✅ Text split into {len(text_split_dict_list)} segments
🔄 Mapping characters to voices...

### {StatusSections.TEXT_SPLIT_BY_CHARACTER}:
{text_split_by_character}
"""

    (
        data_for_tts,
        data_for_sound_effects,
        select_voice_chain_out,
        lines_for_sound_effect,
    ) = await builder.prepare_text_for_tts_with_voice_mapping(
        text_split, generate_effects
    )

    # Create voice mapping markdown
    result_voice_chain_out = {}
    for key in set(select_voice_chain_out.character2props) | set(
        select_voice_chain_out.character2voice
    ):
        result_voice_chain_out[key] = select_voice_chain_out.character2props.get(
            key, []
        ).model_dump()
        result_voice_chain_out[key][
            "voice_id"
        ] = select_voice_chain_out.character2voice.get(key, [])
        result_voice_chain_out[key]["sample_audio_url"] = get_audio_from_voice_id(
            result_voice_chain_out[key]["voice_id"]
        )

    mapping_md = "\n".join(
        f"- **{character}** →"
        f" **Gender**: {voice_properties.get('gender', None)}, "
        f"**Age**: {voice_properties.get('age_group', None)}, "
        f"**Voice ID**: {voice_properties.get('voice_id', None)} "
        f"[<a href='#' class='audio-link' data-audio-url='{voice_properties.get('sample_audio_url', '')}'>Listen Preview 🔊</a>]"
        for character, voice_properties in result_voice_chain_out.items()
    )

    yield None, "", f"""
### Status: Voice Mapping Complete
✅ Text split into {len(text_split_dict_list)} segments
✅ Voice mapping completed
🔄 Generating audio...

### {StatusSections.TEXT_SPLIT_BY_CHARACTER}:
{text_split_by_character}

### Voice Assignments:
{mapping_md}
"""

    out_path = await builder.audio_generator.generate_audio(
        text_split=text_split,
        data_for_tts=data_for_tts,
        data_for_sound_effects=data_for_sound_effects,
        character_to_voice=select_voice_chain_out.character2voice,
        lines_for_sound_effect=lines_for_sound_effect,
    )

    yield out_path, "", f"""
### Status: Process Complete ✨
✅ Text split into {len(text_split_dict_list)} segments
✅ Voice mapping completed
✅ Audio generation complete

### {StatusSections.TEXT_SPLIT_BY_CHARACTER}:
{text_split_by_character}

### Voice Assignments:
{mapping_md}

### 🎉 Your audiobook is ready! Press play to listen.
"""


def refresh():
    return None, None, None, None


with gr.Blocks(js=DESCRIPTION_JS, theme=GRADIO_THEME) as ui:
    with gr.Row(variant="panel"):
        text_input = gr.Textbox(label="Enter the book text here", lines=15)
        file_input = gr.File(
            label="Upload a text file or PDF",
            file_types=[".txt", ".pdf"],
            visible=True,
        )

    examples = gr.Examples(
        examples=[
            [samples.GATSBY_1],
            [samples.GATSBY_2],
            [samples.WONDERFUL_CHRISTMAS_1],
            [samples.WONDERFUL_CHRISTMAS_2],
        ],
        inputs=text_input,
        label="Sample Inputs",
        example_labels=[
            "The Great Gatsby, #1",
            "The Great Gatsby, #2",
            "The Wonderful Christmas in Pumpkin Delight Lane, #1",
            "The Wonderful Christmas in Pumpkin Delight Lane, #2",
        ],
    )

    audio_output = gr.Audio(
        label='Generated audio. Please wait for the waveform to appear, before hitting "Play"',
        type="filepath",
    )

    error_output = gr.Textbox(label="Error Message", interactive=False, visible=False)

    effects_generation_checkbox = gr.Checkbox(
        label="Add background effects",
        value=False,
        info="Select if you want to add occasional sound effect to the audiobook",
    )

    with gr.Row(variant="panel"):
        submit_button = gr.Button("Generate the audiobook", variant="primary")
        refresh_button = gr.Button("Refresh", variant="secondary")

    # status panel
    with gr.Row(variant="panel"):
        status_display = gr.Markdown(
            value="### Status: Waiting to Start\nEnter text or upload a file to begin.",
            label="Generation Status",
        )

    submit_button.click(
        fn=respond,
        inputs=[
            text_input,
            file_input,
            effects_generation_checkbox,
        ],  # Include the uploaded file as an input
        outputs=[
            audio_output,
            error_output,
            status_display,
        ],  # Include the audio output and error message output
    )
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[
            audio_output,
            error_output,
            file_input,
            status_display,
        ],  # Reset audio output, error message, and uploaded file
    )

    # Hide error message dynamically when input is received
    text_input.change(
        fn=lambda _: gr.update(visible=False),  # Hide the error field
        inputs=[text_input],
        outputs=error_output,
    )

    file_input.change(
        fn=lambda _: gr.update(visible=False),  # Hide the error field
        inputs=[file_input],
        outputs=error_output,
    )

    # To clear error field when refreshing
    refresh_button.click(
        fn=lambda _: gr.update(visible=False),  # Hide the error field
        inputs=[],
        outputs=error_output,
    )

# ui.launch(auth=get_auth_params())
ui.launch()
