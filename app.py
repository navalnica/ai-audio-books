import json
import os
import re

import librosa
import requests
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


api_key = os.getenv("AIML_API_KEY")


CHARACTER_CLASSIFICATION_PROMPT = """
**Task:**  
Analyze the provided story text and classify each character in the given list \
by their gender. Use `"M"` for Male and `"F"` for Female. Classify the \
characters based on contextual clues such as names, pronouns, descriptions, \
roles, and interactions within the story.

**Output Format:**  
Provide the classification in a JSON object where each key is a character's \
name, and the value is `"M"` or `"F"`.

**Example Input:**
```
### Story
Once upon a time Alice met Bob and Charlie.

### Characters
["alice", "bob", "charlie"]
```

**Example Output:**
```json
{
  "alice": "F",
  "bob": "M",
  "charlie": "M"
}
"""


TEXT_ANNOTATION_PROMPT = """\
**Task:**  
Analyze the provided text and annotate each segment by indicating whether it is \
part of the narration or spoken by a specific character. Use "Narrator" for \
narration and the character's name for dialogues. Format the annotated text in a \
clear and consistent manner, suitable for subsequent text-to-speech processing.

**Formatting Guidelines:**

- Narration: Prefix with `[Narrator]`
- Character Dialogue: Prefix with `[Character Name]`
- Multiple Characters Speaking: Prefix with `[Character Name 1] [Character Name 2] ... [Character Name N]`
- Consistent Line Breaks: Ensure each labeled segment starts on a new line for clarity.
"""


VOICES = pd.read_csv("data/11labs_tts_voices.csv").query("language == 'en'")


class AudiobookBuilder:
    def __init__(
            self,
            *,
            aiml_api_key: str | None = None,
            aiml_base_url: str = "https://api.aimlapi.com/v1",
            eleven_api_key: str | None = None,
    ) -> None:
        self._aiml_api_key = aiml_api_key or os.environ["AIML_API_KEY"]
        self._aiml_base_url = aiml_base_url
        self._aiml_client = OpenAI(api_key=api_key, base_url=self._aiml_base_url)
        self._default_narrator_voice = "ALY2WaJPY0oBJlqpQbfW"
        self._eleven_api_key = eleven_api_key or os.environ["ELEVEN_API_KEY"]

    def annotate_text(self, text: str) -> str:
        response = self._send_request_to_llm(messages=[
            {
                "role": "system",
                "content": TEXT_ANNOTATION_PROMPT,
            },
            {
                "role": "user",
                "content": text,
            }
        ])
        return response["choices"][0]["message"]["content"]
    
    def classify_characters(self, annotated_text: str, unique_characters: list[str]) -> dict:
        response = self._send_request_to_llm(
            messages=[
                {
                    "role": "system",
                    "content": CHARACTER_CLASSIFICATION_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"### Story\n\n{annotated_text}\n\n### Characters\n\n{unique_characters}",
                },
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response["choices"][0]["message"]["content"])
    
    def generate_audio(
            self,
            annotated_text: str,
            character_to_voice: dict[str, str],
            *,
            chunk_size: int = 1024,
    ) -> None:
        current_character = "narrator"
        with open("audiobook.mp3", "wb") as ab:
            for line in annotated_text.splitlines():
                cleaned_line = line.strip().lower()
                if not cleaned_line:
                    continue
                try:
                    current_character = re.findall(r"\[[\w\s]+\]", cleaned_line)[0][1:-1]
                except:
                    pass
                voice_id = character_to_voice[current_character]
                character_text = cleaned_line[cleaned_line.rfind("]")+1:].lstrip()
                fragment = self._send_request_to_tts(voice_id=voice_id, text=character_text)
                for chunk in fragment.iter_content(chunk_size=chunk_size):
                    if chunk:
                        ab.write(chunk)           

    @staticmethod
    def get_unique_characters(annotated_text: str) -> list[str]:
        characters = set[str]()
        for line in annotated_text.splitlines():
            cleaned_line = line.strip().lower()
            if not cleaned_line.startswith("["):
                continue
            line_characters = re.findall(r"\[[\w\s]+\]", cleaned_line)
            characters = characters.union(ch[1:-1] for ch in line_characters)
        return list(characters - {"narrator"})
    
    def map_characters_to_voices(self, character_to_gender: dict[str, str]) -> dict[str, str]:
        character_to_voice = {"narrator": self._default_narrator_voice}
        
        # Damy vperyod!
        f_characters = [character for character, gender in character_to_gender.items() if gender.strip().lower() == "f"]
        if f_characters:
            f_voices = VOICES.query("gender == 'female'").iloc[:len(f_characters)].copy()        
            f_voices["character"] = f_characters
            character_to_voice |= f_voices.set_index("character")["voice_id"].to_dict()

        m_characters = [character for character, gender in character_to_gender.items() if gender.strip().lower() == "m"]
        if m_characters:
            m_voices = VOICES.query("gender == 'male'").iloc[:len(m_characters)].copy()
            m_voices["character"] = m_characters
            character_to_voice |= m_voices.set_index("character")["voice_id"].to_dict()
        
        return character_to_voice
    
    def _send_request_to_llm(self, messages: list[dict], **kwargs) -> dict:
        response = requests.post(
            url=f"{self._aiml_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._aiml_api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "gpt-4o",
                "temperature": 0.0,
                "messages": messages,
                "stream": False,
                "max_tokens": 16_384,
                **kwargs,
            }),
        )
        response.raise_for_status()
        return response.json()
    
    def _send_request_to_tts(self, voice_id: str, text: str):
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self._eleven_api_key,
        }
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response


def respond(text):
    builder = AudiobookBuilder()
    
    annotated_text = builder.annotate_text(text)
    unique_characters = builder.get_unique_characters(annotated_text)
    character_to_gender = builder.classify_characters(text, unique_characters)
    character_to_voice = builder.map_characters_to_voices(character_to_gender)
    builder.generate_audio(annotated_text, character_to_voice)
    
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
