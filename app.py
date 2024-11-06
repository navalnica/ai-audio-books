import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

from data import samples_to_split as samples
from src.builder import AudiobookBuilder
from src.config import FILE_SIZE_MAX, MAX_TEXT_LEN, logger
from src.web.utils import create_status_html
from src.web.variables import DESCRIPTION_JS, GRADIO_THEME, STATUS_DISPLAY_HTML, VOICE_UPLOAD_JS


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


async def audiobook_builder(
    text: str,
    uploaded_file,
    generate_effects: bool,
    use_user_voice: bool,
    voice_id: str | None = None,
):
    builder = AudiobookBuilder()

    if uploaded_file is not None:
        try:
            text = load_text_from_file(uploaded_file=uploaded_file)
        except Exception as e:
            logger.exception(e)
            msg = "Failed to load text from the provided document"
            gr.Warning(msg)
            yield None, str(e), builder.html_generator.generate_error(msg)
            return

    if not text:
        logger.info(f"No text was passed. can't generate an audiobook")
        msg = 'Please provide the text to generate audiobook from'
        gr.Warning(msg)
        yield None, "", builder.html_generator.generate_error(msg)
        return

    if (text_len := len(text)) > MAX_TEXT_LEN:
        msg = (
            f"Input text length of {text_len} characters "
            f"exceeded current limit of {MAX_TEXT_LEN} characters. "
            "Please input a shorter text."
        )
        logger.info(msg)
        gr.Warning(msg)
        yield None, "", builder.html_generator.generate_error(msg)
        return

    async for stage in builder.run(text, generate_effects, use_user_voice, voice_id):
        yield stage


def refresh():
    return None, None, None, STATUS_DISPLAY_HTML


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

    error_output = gr.Textbox(label="Error Message", interactive=False, visible=False)

    effects_generation_checkbox = gr.Checkbox(
        label="Add sound effects",
        value=False,
        info="Select if you want to add occasional sound effect to the audiobook",
    )

    use_voice_checkbox = gr.Checkbox(
        label="Use my voice",
        value=False,
        info="Select if you want to use your voice for whole or part of the audiobook (Generations may take longer than usual)",
    )

    submit_button = gr.Button("Generate the audiobook", variant="primary")

    with gr.Row(variant="panel"):
        add_voice_btn = gr.Button("Add my voice", variant="primary")
        refresh_button = gr.Button("Refresh", variant="secondary")

    voice_result = gr.Textbox(visible=False, interactive=False, label="Processed Result")
    status_display = gr.HTML(value=STATUS_DISPLAY_HTML, label="Generation Status")
    audio_output = gr.Audio(
        label='Generated audio. Please wait for the waveform to appear, before hitting "Play"',
        type="filepath",
    )

    # callbacks

    add_voice_btn.click(fn=None, inputs=None, outputs=voice_result, js=VOICE_UPLOAD_JS)
    submit_button.click(
        fn=audiobook_builder,
        inputs=[
            text_input,
            file_input,
            effects_generation_checkbox,
            use_voice_checkbox,
            voice_result,
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
        ],  # Reset audio output, error message, and uploaded file
    )
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
    refresh_button.click(
        fn=lambda _: gr.update(visible=False),  # Hide the error field
        inputs=[],
        outputs=error_output,
    )

# ui.launch(auth=get_auth_params())
ui.launch()
