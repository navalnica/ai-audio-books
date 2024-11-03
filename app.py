import json
import os
from html.parser import HTMLParser
from pathlib import Path
from typing import List
import re

import gradio as gr
import openai
from altair.vegalite.v5.theme import theme
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

from src.utils import get_audio_from_voice_id

load_dotenv()

from data import samples_to_split as samples
from src.builder import AudiobookBuilder
from src.config import FILE_SIZE_MAX, MAX_TEXT_LEN, logger, VOICE_UPLOAD_JS, STATUS_DISPLAY_HTML, \
    GRADIO_THEME, DESCRIPTION_JS, OPENAI_API_KEY

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
        raise ValueError(f"The uploaded file exceeds the size limit of {FILE_SIZE_MAX} MB.")

    if uploaded_file.name.endswith(".txt"):
        with open(temp_file_path, "r", encoding="utf-8") as file:
            text = file.read()
    elif uploaded_file.name.endswith(".pdf"):
        text = parse_pdf(temp_file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a .txt or .pdf file.")

    return text

def get_character_color(character: str) -> str:
    if not character or character == "Unassigned":
        return "#808080"
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEEAD", "#D4A5A5", "#9B59B6", "#3498DB"]
    hash_val = sum(ord(c) for c in character)
    return colors[hash_val % len(colors)]

# Function to replace 'c<number>' with 'Character<number>'
def replace_labels(text):
    # Replace 'c<number>' with 'Character<number>'
    return re.sub(r'\bc(\d+)\b', r'Character\1', text)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return f"{int(hex_color[0:2], 16)},{int(hex_color[2:4], 16)},{int(hex_color[4:6], 16)}"

def create_status_html(status: str, steps: list[tuple[str, bool]]) -> str:
    steps_html = "\n".join([
        f'<div class="step-item" style="display: flex; align-items: center; padding: 0.8rem; margin-bottom: 0.5rem; background-color: #31395294; border-radius: 6px; font-weight: 600;">'
        f'<span class="step-icon" style="margin-right: 1rem; font-size: 1.3rem;">{("✅" if completed else "🔄")}</span>'
        f'<span class="step-text" style="font-size: 1.1rem; color: #e0e0e0;">{step}</span>'
        f'</div>'
        for step, completed in steps
    ])

    return f'''
    <div class="status-container" style="font-family: system-ui; max-width: 1472px; margin: 0 auto; background-color: #31395294; padding: 1rem; border-radius: 8px; color: #f0f0f0;">
        <div class="status-header" style="background: #31395294; padding: 1rem; border-radius: 8px; font-weight: bold;">
            <h3 class="status-title" style="margin: 0; color: rgb(224, 224, 224); font-size: 1.5rem; font-weight: 700;">Status: {status}</h3>
            <p class="status-description" style="margin: 0.5rem 0 0 0; color: #c0c0c0; font-size: 1rem; font-weight: 400;">Processing steps below.</p>
        </div>
        <div class="steps" style="margin-top: 1rem;">
            {steps_html}
    </div>
    '''

def generate_legend_for_text_split_html(text_split_dict_list: list[dict], add_effect_legend: bool = False) -> str:
    legend_html = "<div style='margin-bottom: 1rem;'>"
    legend_html += "<div style='font-size: 1.35em; font-weight: bold;'>Legend:</div>"

    unique_characters = set(item['character'] or 'Unassigned' for item in text_split_dict_list)

    for character in unique_characters:
        color = get_character_color(character)
        # Set a slightly smaller font size for each character name
        legend_html += f"<div style='color: {color}; font-size: 1.1em; margin-bottom: 0.25rem;'>{character}</div>"
    if add_effect_legend:
        legend_html += f"<div style='color: #BBB951F7; font-size: 1.1em; margin-bottom: 0.25rem;'>Effects</div>"
    legend_html += "</div>"
    return legend_html

def generate_text_split_without_effect_html(text_split_dict_list: list[dict]) -> str:
    text_split_html = "<div style='font-size: 1.2em; line-height: 1.6;'>"

    for item in text_split_dict_list:
        character = item['character'] or 'Unassigned'
        text = item['text']
        color = get_character_color(character)
        rgba_color = f"rgba({hex_to_rgb(color)}, 0.3)"
        # Add each phrase inline with character color
        text_split_html += f"<span style='background-color: {rgba_color}; padding: 0.2em; border-radius: 0.2em;'>{text}</span> "

    text_split_html += "</div>"
    return text_split_html

def generate_full_text_split_without_effect_html(text_split_dict_list: list[dict]) -> str:
    legend_html = generate_legend_for_text_split_html(text_split_dict_list)
    text_split_html = generate_text_split_without_effect_html(text_split_dict_list)
    return legend_html + text_split_html

def generate_full_text_split_with_effect_html(
        text_split_dict_list: list[dict],
        text_with_effects: list[str],
        text_between_effects_texts: list[str],
) -> str:
    css_styles = """
    <style>
            .text-effect-container {
            font-size: 1.2em;
            line-height: 1.6;
        }

        .character-segment {
            padding: 0.2em;
            border-radius: 0.2em;
        }

        .effect-container {
            position: relative;
            display: inline-block;
        }

        .effect-text {
            background-color: rgba(187, 185, 81, 0.97);
            padding: 2px 4px;
            border-radius: 3px;
            border-bottom: 2px dashed rgba(0, 0, 0, 0.83);
            cursor: help;
        }

        .effect-tooltip {
            visibility: hidden;
            background-color: #333;
            color: white;
            text-align: center;
            padding: 5px 10px;
            border-radius: 6px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            white-space: nowrap;
            opacity: 0;
            transition: opacity 0.3s;
        }

        .effect-tooltip::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #333 transparent transparent transparent;
        }

        .effect-container:hover .effect-tooltip {
            visibility: visible;
            opacity: 1;
        }
    </style>
    """

    def create_effect_span(text: str, effect_description: str, bg_color: str) -> str:
        """Create an HTML span with effect tooltip."""
        return f"""
            <span class="character-segment" style="background-color: {bg_color}">
                <span class="effect-container">
                    <span class="effect-text">{text}</span>
                    <span class="effect-tooltip">Effect: {effect_description}</span>
                </span>
            </span>"""

    def create_regular_span(text: str, bg_color: str) -> str:
        """Create a regular HTML span with background color."""
        return f'<span class="character-segment" style="background-color: {bg_color}">{text}</span>'

    html_parts = []
    effect_index = 0

    for split_item in text_split_dict_list:
        character = split_item['character'] or 'Unassigned'
        text = split_item['text']
        color = get_character_color(character)
        rgba_color = f"rgba({hex_to_rgb(color)}, 0.3)"

        if effect_index >= len(text_between_effects_texts) or text_between_effects_texts[
            effect_index].lower() not in text.lower():
            html_parts.append(create_regular_span(text, rgba_color))
        else:
            prev_end = 0
            while effect_index < len(text_between_effects_texts) and text_between_effects_texts[
                effect_index].lower() in text.lower():
                effect_text = text_with_effects[effect_index]
                text_between_effect_description = text_between_effects_texts[effect_index]

                effect_start = text.lower().find(text_between_effect_description.lower())
                effect_end = effect_start + len(text_between_effect_description)

                html_parts.append(create_regular_span(
                    text[prev_end:effect_start],
                    rgba_color,
                ))
                html_parts.append(create_effect_span(
                    text_between_effect_description,
                    effect_text,
                    rgba_color,
                ))
                effect_index += 1
                prev_end = effect_end

    legend_html = generate_legend_for_text_split_html(text_split_dict_list, add_effect_legend=True)

    content_html = f"""
    {css_styles}
    <div class="text-effect-container">
        {''.join(html_parts)}
    </div>
    """
    return legend_html + content_html


async def respond(
        text: str,
        uploaded_file,
        generate_effects: bool,
        use_user_voice: bool,
        voice_id: str = None,
) -> tuple[Path | None, str, str]:

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
    builder = AudiobookBuilder()
    if use_user_voice:
        if voice_id:
            out_path = await builder.run(text=text, generate_effects=generate_effects, use_user_voice=use_user_voice, voice_id=voice_id)
            yield out_path, "", """<div class="audiobook-ready" style="background-color: #31395294; padding: 1rem; border-radius: 8px; margin-top: 1rem; text-align: center;">
                    <h3 style="color: rgb(224, 224, 224); font-size: 1.5em; margin-bottom: 1rem;">🎉 Your audiobook is ready!</h3>
                    <p style="color: #4299e1; cursor: pointer;" onclick="document.querySelector('.play-pause-button.icon.svelte-ije4bl').click();">🔊 Press play to listen 🔊</p>
                </div>"""
        else:
            yield None, "", """<div class="audiobook-ready" style="background-color: #31395294; padding: 1rem; border-radius: 8px; margin-top: 1rem; text-align: center;">
                    <h3 style="color: rgb(224, 224, 224); font-size: 1.5em; margin-bottom: 1rem;">🫤 At first you should add your voice</h3>
                </div>"""

    else:
        # Initial status
        yield None, "", create_status_html("Starting Process", [
            ("Splitting text into characters...", False)
        ]) + "</div>"

        text_split = await builder.split_text(text)
        text_split_dict_list = [item.model_dump() for item in text_split._phrases]
        # Replace 'c<number>' with 'Character<number>' in the character labels
        for item in text_split_dict_list:
            item['character'] = replace_labels(item['character'])

        text_split_html = generate_full_text_split_without_effect_html(text_split_dict_list)

        yield None, "", create_status_html("Text Analysis Complete", [
            ("Text splitting", True),
            ("Mapping characters to voices...", False)
        ]) + f'''
            <div class="section" style="background-color: #31395294; padding: 1rem; border-radius: 8px; margin-top: 1rem; color: #e0e0e0;">
                <h3 style="color: rgb(224, 224, 224); font-size: 1.5em; margin-bottom: 1rem;">Text Split by Character:</h3>
                {text_split_html}
            </div>
        </div>
        '''

        # Voice mapping
        (
            data_for_tts,
            data_for_sound_effects,
            text_between_effects_texts,
            select_voice_chain_out,
            lines_for_sound_effect,
        ) = await builder.prepare_text_for_tts_with_voice_mapping(
            text_split, generate_effects, use_user_voice
        )

        if generate_effects:
            text_split_html = generate_full_text_split_with_effect_html(
                text_split_dict_list,
                data_for_sound_effects,
                text_between_effects_texts,
            )

        # Create voice mapping HTML
        result_voice_chain_out = {}
        for key in set(select_voice_chain_out.character2props) | set(
                select_voice_chain_out.character2voice
        ):
            character_props = select_voice_chain_out.character2props.get(key, []).model_dump()
            # Add voice_id and sample audio URL
            character_props["voice_id"] = select_voice_chain_out.character2voice.get(key, [])
            character_props["sample_audio_url"] = get_audio_from_voice_id(character_props["voice_id"])

            result_voice_chain_out[replace_labels(key)] = character_props

        voice_assignments_html = ""
        for character, voice_properties in result_voice_chain_out.items():
            color = get_character_color(character)
            voice_assignments_html += f'''
            <div class="voice-assignment" style="background-color: #31395294; padding: 1rem; border-radius: 8px; margin-top: 1rem; color: #e0e0e0;">
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
            <div class="section" style="background-color: #31395294; padding: 1rem; border-radius: 8px; margin-top: 1rem; color: #e0e0e0;">
                <h3 style="color: rgb(224, 224, 224); font-size: 1.5em; margin-bottom: 1rem;">Text Split by Character:</h3>
                {text_split_html}
            </div>
            <div class="section" style="background-color: #31395294; padding: 1rem; border-radius: 8px; margin-top: 1rem; color: #e0e0e0;">
                <h3 style="color: rgb(224, 224, 224); font-size: 1.5em; margin-bottom: 1rem;">Voice Assignments:</h3>
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
            
            <div class="section" style="background-color: #31395294; padding: 1rem; border-radius: 8px; margin-top: 1rem; color: #e0e0e0;">
                <h3 style="color: rgb(224, 224, 224); font-size: 1.5em; margin-bottom: 1rem;">Text Split by Character:</h3>
                {text_split_html}
            </div>
            <div class="section" style="background-color: #31395294; padding: 1rem; border-radius: 8px; margin-top: 1rem; color: #e0e0e0;">
                <h3 style="color: rgb(224, 224, 224); font-size: 1.5em; margin-bottom: 1rem;">Voice Assignments:</h3>
                {voice_assignments_html}
            </div>
            <div class="audiobook-ready" style="background-color: #31395294; padding: 1rem; border-radius: 8px; margin-top: 1rem; text-align: center;">
                    <h3 style="color: rgb(224, 224, 224); font-size: 1.5em; margin-bottom: 1rem;">🎉 Your audiobook is ready!</h3>
                    <p style="color: #4299e1; cursor: pointer;" onclick="document.querySelector('.play-pause-button.icon.svelte-ije4bl').click();">🔊 Press play to listen 🔊</p>
                </div>
        </div>
        '''


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

    use_voice_checkbox = gr.Checkbox(
        label="Use my voice",
        value=False,
        info="Select if you want to use your voice for whole or part of the audiobook (Generations may take longer than usual)",
    )

    submit_button = gr.Button("Generate the audiobook", variant="primary")

    with gr.Row(variant="panel"):
        add_voice_btn = gr.Button(
            "Add my voice",
            variant="primary"
        )
        refresh_button = gr.Button("Refresh", variant="secondary")

    voice_result = gr.Textbox(visible=False, interactive=False, label="Processed Result")
    status_display = gr.HTML(
        value=STATUS_DISPLAY_HTML,
        label="Generation Status"
    )

    add_voice_btn.click(fn=None, inputs=None, outputs=voice_result, js=VOICE_UPLOAD_JS)
    submit_button.click(
        fn=respond,
        inputs=[
            text_input,
            file_input,
            effects_generation_checkbox,
            use_voice_checkbox,
            voice_result
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
