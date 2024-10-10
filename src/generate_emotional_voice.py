from openai import OpenAI
import json
import requests

client = OpenAI(api_key = '')
PROMT = """
You should help me to make an audiobook with realistic emotion-based voice using TTS.
You are tasked with adjusting the emotional tone of a given text
by modifying the text with special characters such as "!", "...", "-", "~",
and uppercase words to add emphasis or convey emotion. For adding more emotion u can
duplicate special characters for example "!!!".
Do not remove or add any different words.
Only alter the presentation of the existing words.
After modifying the text, adjust the "stability", "similarity_boost" and "style" parameters
according to the level of emotional intensity in the modified text.
Higher emotional intensity should lower the "stability" and raise the "similarity_boost". 
 Your output should be in the following JSON format:
 {
  "modified_text": "Modified text with emotional adjustments.",
  "params": {
    "stability": 0.7,
    "similarity_boost": 0.5,
    "style": 0.3
  }
}

The "stability" parameter should range from 0 to 1,
with lower values indicating a more expressive, less stable voice.
The "similarity_boost" parameter should also range from 0 to 1,
with higher values indicating more emphasis on the voice similarity.
The "style" parameter should also range from 0 to 1,
where lower values indicate a neutral tone and higher values reflect more stylized or emotional delivery.
Adjust both according to the emotional intensity of the text.

Example of text that could be passed:

Text: "I can't believe this is happening."
"""

text_to_modified = """
The moment she closed the door behind her, he felt the emptiness crash down on him like an invisible weight. The room suddenly seemed foreign, lifeless. Every corner, every object reminded him of her presence, her laughter, those moments that once felt like they would last forever.

He sat down on the couch, his hand tracing the empty space beside him where she used to sit. His heart pounded too loudly, thoughts swirling like autumn leaves in the wind. 'Why?' — the question burned in his mind, gnawing at him from the inside. It felt as though everything had collapsed at once, leaving only silence and pain.

He knew he would never see her smile again, never hear her voice and that was unbearable. Yet, he couldn’t reconcile himself with the fact that she was truly gone. 'How do I go on?' — he wondered, but there was no answer.
"""

def generate_modified_text(text: str) -> dict:
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": PROMT},
                  {"role": "user", "content": text}],
        response_format={"type": "json_object"}
    )
    chatgpt_output = completion.choices[0].message.content
    try:
        output_dict = json.loads(chatgpt_output)
        return output_dict
    except json.JSONDecodeError:
        raise f"error, output_text: {chatgpt_output}"


def generate_audio(text: str, params: dict, output_file: str):
    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/pMsXgVXv3BLzUgSXRplE"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ""
    }

    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": params
    }

    response = requests.post(url, json=data, headers=headers)
    with open(f'{output_file}.mp3', 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)

if __name__ == "__main__":
    default_param = {
        "stability": 0.5,
        "similarity_boost": 0.5,
        "style": 0.5
    }
    generate_audio(text_to_modified, default_param, "text_without_prompt")
    modified_text_with_params = generate_modified_text(text_to_modified)
    print(modified_text_with_params)
    generate_audio(modified_text_with_params['modified_text'], modified_text_with_params['params'], "text_with_prompt")