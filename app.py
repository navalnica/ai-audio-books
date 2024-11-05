import os
from pathlib import Path
from typing import List

import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

from data import samples_to_split as samples
from src.builder import AudiobookBuilder
from src.config import DESCRIPTION, FILE_SIZE_MAX, MAX_TEXT_LEN, logger


def get_auth_params():
    user = os.environ["AUTH_USER"]
    password = os.environ["AUTH_PASS"]
    return (user, password)


def parse_pdf(file_path):
    """Parse the PDF file and return the text content."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return "\n".join([doc.page_content for doc in documents])


def load_text_from_file(uploaded_file):
    # Save the uploaded file temporarily to check its size
    temp_file_path = uploaded_file.name

    if os.path.getsize(temp_file_path) > FILE_SIZE_MAX * 1024 * 1024:
        raise ValueError(f"The uploaded file exceeds the size limit of {FILE_SIZE_MAX} MB.")

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
) -> tuple[Path | None, str]:
    if uploaded_file is not None:
        try:
            text = load_text_from_file(uploaded_file=uploaded_file)
        except Exception as e:
            logger.exception(e)
            return (None, str(e))

    if (text_len := len(text)) > MAX_TEXT_LEN:
        gr.Warning(
            f"Input text length of {text_len} characters "
            f"exceeded current limit of {MAX_TEXT_LEN} characters. "
            "Please input a shorter text."
        )
        return None, ""

    builder = AudiobookBuilder()
    audio_fp = await builder.run(text=text, generate_effects=generate_effects)

    return audio_fp, ""


def refresh():
    return None, None, None  # Reset audio output, error message, and uploaded file


with gr.Blocks(title="Audiobooks Generation") as ui:
    gr.Markdown(DESCRIPTION)

    with gr.Row(variant="panel"):
        text_input = gr.Textbox(label="Enter the book text here", lines=15)
        file_input = gr.File(
            label="Upload a text file or PDF",
            file_types=[".txt", ".pdf"],
            visible=False,
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
    # error output is hidden initially
    error_output = gr.Textbox(label="Error Message", interactive=False, visible=False)

    effects_generation_checkbox = gr.Checkbox(
        label="Add sound effects",
        value=False,
        info="Select if you want to add occasional sound effect to the audiobook",
    )

    with gr.Row(variant="panel"):
        submit_button = gr.Button("Generate the audiobook", variant="primary")
        refresh_button = gr.Button("Refresh", variant="secondary")

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
        ],  # Include the audio output and error message output
    )
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[
            audio_output,
            error_output,
            file_input,
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

ui.launch(auth=get_auth_params())
