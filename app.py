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
    def get_character_color(character: str) -> str:
        if not character or character == "Unassigned":
            return "#808080"
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEEAD", "#D4A5A5", "#9B59B6", "#3498DB"]
        hash_val = sum(ord(c) for c in character)
        return colors[hash_val % len(colors)]

    def create_status_html(status: str, steps: list[tuple[str, bool]]) -> str:
        steps_html = "\n".join([
            f'<div class="step-item" style="display: flex; align-items: center; padding: 0.8rem; margin-bottom: 0.5rem; background-color: #3b4c63; border-radius: 6px; font-weight: 600;">'
            f'<span class="step-icon" style="margin-right: 1rem; font-size: 1.3rem;">{("✅" if completed else "🔄")}</span>'
            f'<span class="step-text" style="font-size: 1.1rem; color: #e0e0e0;">{step}</span>'
            f'</div>'
            for step, completed in steps
        ])

        return f'''
        <div class="status-container" style="font-family: system-ui; max-width: 1472px; margin: 0 auto; background-color: #2e3b4e; padding: 1rem; border-radius: 8px; color: #f0f0f0;">
            <div class="status-header" style="background: #3b4c63; padding: 1rem; border-radius: 8px; font-weight: bold;">
                <h3 class="status-title" style="margin: 0; color: #ffffff; font-size: 1.5rem; font-weight: 700;">Status: {status}</h3>
                <p class="status-description" style="margin: 0.5rem 0 0 0; color: #c0c0c0; font-size: 1rem; font-weight: 400;">Processing steps below.</p>
            </div>
            <div class="steps" style="margin-top: 1rem;">
                {steps_html}
        </div>
        '''

    # Error handling for file upload
    if uploaded_file is not None:
        try:
            text = load_text_from_file(uploaded_file=uploaded_file)
        except Exception as e:
            logger.exception(e)
            yield None, str(e), create_status_html("Error",
                                                    []) + '<div class="error-message" style="color: #e53e3e;">Failed to process file.</div></div>'

    # Length check
    if (text_len := len(text)) > MAX_TEXT_LEN:
        gr.Warning(
            f"Input text length of {text_len} characters "
            f"exceeded current limit of {MAX_TEXT_LEN} characters. "
            "Please input a shorter text."
        )
        yield None, "", create_status_html("Error",
                                            []) + '<div class="error-message" style="color: #e53e3e;">Text too long. Please input a shorter text.</div></div>'

    # Initial status
    yield None, "", create_status_html("Starting Process", [
        ("Splitting text into characters...", False)
    ]) + "</div>"

    # Text splitting
    builder = AudiobookBuilder()
    text_split = await builder.split_text(text)
    text_split_dict_list = [item.model_dump() for item in text_split._phrases]

    # Group texts by character
    character_groups = {}
    for item in text_split_dict_list:
        char = item['character'] or 'Unassigned'
        if char not in character_groups:
            character_groups[char] = []
        character_groups[char].append(item['text'])

    # Create character list HTML
    text_split_html = ""
    for character, texts in character_groups.items():
        color = get_character_color(character)
        text_split_html += f'''
        <div class="character-group" style="margin-bottom: 1.5rem;" style="background-color: #3b4c63; padding: 1rem; border-radius: 8px; margin-top: 1rem; color: #e0e0e0;">
            <h4 style="color: {color}; font-weight: 600; margin-bottom: 0.5rem;">
                {character}
            </h4>
            <ul style="list-style-type: disc; margin: 0; padding-left: 1.5rem;">
                {"".join(f'<li style="margin-bottom: 0.25rem;">{text}</li>' for text in texts)}
            </ul>
        </div>
        '''

    yield None, "", create_status_html("Text Analysis Complete", [
        ("Text splitting", True),
        ("Mapping characters to voices...", False)
    ]) + f'''
        <div class="section" style="background-color: #3b4c63; padding: 1rem; border-radius: 8px; margin-top: 1rem; color: #e0e0e0;">
            <h3 style="color: #ffffff; font-size: 1.15rem; margin-bottom: 1rem;">Text Split by Character:</h3>
            {text_split_html}
        </div>
    </div>
    '''

    # Voice mapping
    (
        data_for_tts,
        data_for_sound_effects,
        select_voice_chain_out,
        lines_for_sound_effect,
    ) = await builder.prepare_text_for_tts_with_voice_mapping(
        text_split, generate_effects
    )

    # Create voice mapping HTML
    result_voice_chain_out = {}
    for key in set(select_voice_chain_out.character2props) | set(
            select_voice_chain_out.character2voice
    ):
        result_voice_chain_out[key] = select_voice_chain_out.character2props.get(
            key, []
        ).model_dump()
        result_voice_chain_out[key]["voice_id"] = select_voice_chain_out.character2voice.get(key, [])
        result_voice_chain_out[key]["sample_audio_url"] = get_audio_from_voice_id(
            result_voice_chain_out[key]["voice_id"]
        )

    voice_assignments_html = ""
    for character, voice_properties in result_voice_chain_out.items():
        color = get_character_color(character)
        voice_assignments_html += f'''
        <div class="voice-assignment" style="background-color: #3b4c63; padding: 1rem; border-radius: 8px; margin-top: 1rem; color: #e0e0e0;">
            <span style="color: {color}; font-weight: 600;">{character}</span>
            <span style="margin: 0 0.5rem;">→</span>
            <span style="color: #4a5568;">
                <strong>Gender: {voice_properties.get('gender', 'N/A')}</strong> ,
                <strong>Age: {voice_properties.get('age_group', 'N/A')}</strong> ,
                <strong>Voice ID: {voice_properties.get('voice_id', 'N/A')}</strong> 
            </span>
            <a href="#" 
               class="audio-link" 
               data-audio-url="{voice_properties.get('sample_audio_url', '')}"
               style="margin-left: 0.5rem; color: #4299e1; text-decoration: none;">
               Listen Preview 🔊
            </a>
        </div>
        '''

    yield None, "", create_status_html("Voice Mapping Complete", [
        ("Text splitting", True),
        ("Voice mapping", True),
        ("Generating audio...", False)
    ]) + f'''
        <div class="section" style="background-color: #3b4c63; padding: 1rem; border-radius: 8px; margin-top: 1rem; color: #e0e0e0;">
            <h3 style="color: #ffffff; font-size: 1.15rem; margin-bottom: 1rem;">Text Split by Character:</h3>
            {text_split_html}
        </div>
        <div class="section" style="background-color: #3b4c63; padding: 1rem; border-radius: 8px; margin-top: 1rem; color: #e0e0e0;">
            <h3 style="color: #ffffff; font-size: 1.15rem; margin-bottom: 1rem;">Voice Assignments:</h3>
            {voice_assignments_html}
        </div>
    </div>
    '''

    # Audio generation
    out_path = await builder.audio_generator.generate_audio(
        text_split=text_split,
        data_for_tts=data_for_tts,
        data_for_sound_effects=data_for_sound_effects,
        character_to_voice=select_voice_chain_out.character2voice,
        lines_for_sound_effect=lines_for_sound_effect,
    )

    yield out_path, "", create_status_html("Process Complete ✨", [
        ("Text splitting", True),
        ("Voice mapping", True),
        ("Audio generation", True)
    ]) + f'''
        
        <div class="section" style="background-color: #3b4c63; padding: 1rem; border-radius: 8px; margin-top: 1rem; color: #e0e0e0;">
            <h3 style=style="color: #ffffff; font-size: 1.15rem; margin-bottom: 1rem;">Text Split by Character:</h3>
            {text_split_html}
        </div>
        <div class="section" style="background-color: #3b4c63; padding: 1rem; border-radius: 8px; margin-top: 1rem; color: #e0e0e0;">
            <h3 style=style="color: #ffffff; font-size: 1.15rem; margin-bottom: 1rem;">Voice Assignments:</h3>
            {voice_assignments_html}
        </div>
        <div class="audiobook-ready" style="background-color: #3b4c63; padding: 1rem; border-radius: 8px; margin-top: 1rem; text-align: center;">
                <h3 style="color: #ffffff; font-size: 1.25rem; margin-bottom: 0.5rem;">🎉 Your audiobook is ready!</h3>
                <p style="color: #c0c0c0;">Press play to listen.</p>
            </div>
    </div>
    '''


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
    status_display = gr.HTML(
        value='''
        <style>
          .status-container {
              font-family: system-ui;
              max-width: 1472;
              margin: 0 auto;
              background-color: #2e3b4e; /* Darker background color */
              padding: 1rem;
              border-radius: 8px;
              color: #f0f0f0; /* Light text color */
          }
          .status-header {
              background: #3b4c63; /* Slightly lighter background */
              padding: 1rem;
              border-radius: 8px;
              font-weight: bold; /* Emphasize header */
          }
          .status-title {
              margin: 0;
              color: #ffffff; /* White color for title */
              font-size: 1.5rem; /* Larger title font */
              font-weight: 700; /* Bold title */
          }
          .status-description {
              margin: 0.5rem 0 0 0;
              color: #c0c0c0;
              font-size: 1rem;
              font-weight: 400; /* Regular weight for description */
          }
          .steps {
              margin-top: 1rem;
          }
          .step-item {
              display: flex;
              align-items: center;
              padding: 0.8rem;
              margin-bottom: 0.5rem;
              background-color: #3b4c63; /* Matching background color */
              border-radius: 6px;
              color: #f0f0f0; /* Light text color */
              font-weight: 600; /* Medium weight for steps */
          }
          .step-item:hover {
              background-color: rgba(255, 255, 255, 0.07);
          }
          .step-icon {
              margin-right: 1rem;
              font-size: 1.3rem; /* Slightly larger icon size */
          }
          .step-text {
              font-size: 1.1rem; /* Larger text for step description */
              color: #e0e0e0; /* Lighter text for better readability */
          }
        </style>

        <div class="status-container">
            <div class="status-header">
                <h3 class="status-title">Status: Waiting to Start</h3>
                <p class="status-description">Enter text or upload a file to begin.</p>
            </div>
            <div class="steps">
                <div class="step-item">
                    <span class="step-icon">📚</span>
                    <span class="step-text">Split text into characters</span>
                </div>
                <div class="step-item">
                    <span class="step-icon">🎭</span>
                    <span class="step-text">Assign each character a voice</span>
                </div>
                <!-- Add more steps as needed -->
            </div>
        </div>
        ''',
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
