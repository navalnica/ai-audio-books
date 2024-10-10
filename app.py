import os

import gradio as gr
import librosa
from dotenv import load_dotenv

load_dotenv()

from src.builder import AudiobookBuilder

api_key = os.getenv("AIML_API_KEY")


def respond(text):
    builder = AudiobookBuilder()
    builder.run(text=text)

    audio, sr = librosa.load("audiobook.mp3", sr=None)
    return (sr, audio)


with gr.Blocks(title="Audiobooks Generation") as ui:
    gr.Markdown("# Audiobooks Generation")

    with gr.Row(variant="panel"):
        text_input = gr.Textbox(label="Enter the book text", lines=20)

    with gr.Row(variant="panel"):
        audio_output = gr.Audio(label="Generated audio")

    submit_button = gr.Button("Submit")
    submit_button.click(
        fn=respond,
        inputs=[text_input],
        outputs=[audio_output],
    )


ui.launch()
