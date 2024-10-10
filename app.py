import os
from pathlib import Path
from typing import List

import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

from src.builder import AudiobookBuilder
from src.config import logger, FILE_SIZE_MAX


def get_auth_params() -> List[tuple[str, str]]:
    users = os.environ["AUTH_USERS"].split(",")
    passwords = os.environ["AUTH_PASS"].split(",")

    auth_list = [(u, passwords[0] if len(passwords) == 1 else p) for u, p in zip(users, passwords)]
    return auth_list


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


async def respond(text: str, uploaded_file) -> tuple[Path | None, str]:
    if uploaded_file is not None:
        try:
            text = load_text_from_file(uploaded_file=uploaded_file)
        except Exception as e:
            logger.exception(e)
            return (None, str(e))

    builder = AudiobookBuilder()
    audio_fp = await builder.run(text=text)
    return audio_fp, ""


def refresh():
    return None, None, None  # Reset audio output, error message, and uploaded file


with gr.Blocks(title="Audiobooks Generation") as ui:
    gr.Markdown("# Audiobooks Generation")

    with gr.Row(variant="panel"):
        text_input = gr.Textbox(label="Enter the book text", lines=20)
        # Add a file upload field for .txt and .pdf files
        file_input = gr.File(
            label="Upload a text file or PDF", file_types=[".txt", ".pdf"]
        )

    with gr.Row(variant="panel"):
        audio_output = gr.Audio(label="Generated audio", type="filepath")
        error_output = gr.Textbox(
            label="Error Messages", interactive=False, visible=False
        )  # Initially hidden

    submit_button = gr.Button("Submit")
    submit_button.click(
        fn=respond,
        inputs=[text_input, file_input],  # Include the uploaded file as an input
        outputs=[
            audio_output,
            error_output,
        ],  # Include the audio output and error message output
    )

    refresh_button = gr.Button("Refresh")
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
