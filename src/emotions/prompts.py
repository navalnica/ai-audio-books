SOUND_EFFECT_GENERATION = """
You should help me to make an audiobook with realistic emotion sound using TTS.
You are tasked with generating a description of sound effects
that matches the atmosphere, actions, and tone of a given sentence or text from a book.
The description should be tailored to create a sound effect using ElevenLabs'sound generation API.
The generated sound description must evoke the scene
or emotions from the text (e.g., footsteps, wind, tense silence, etc.),
and it should be succinct and fit the mood of the text.

Additionally, you should include the following parameters in your response:

    Text: A generated description of the sound that matches the text provided.
        Keep the description simple and effective to capture the soundscape.
        This text will be converted into a sound effect. 
    Duration_seconds: The appropriate duration of the sound effect,
        which should be calculated based on the length and nature of the scene.
        Cap this duration at 22 seconds. But be carefully, for very long text in input make a long sound effect,
         for small make a small one. And the duration should be similar to duration of input text
    Prompt_influence: A value between 0 and 1, where a higher value makes the sound generation closely
        follow the sound description. For general sound effects (e.g., footsteps, background ambiance),
        use a value around 0.3. For more specific or detailed sound scenes
        (e.g., thunderstorm, battle sounds), use a higher value like 0.5 to 0.7.

Your output should be in the following JSON format:

{
  "text": "A soft breeze rustling through leaves, distant birds chirping.",
  "duration_seconds": 4.0,
  "prompt_influence": 0.4
}

"""

SOUND_EFFECT_GENERATION_WITHOUT_DURATION_PREDICTION = """
You should help me to make an audiobook with realistic emotion sound using TTS.
You are tasked with generating a description of sound effects
that matches the atmosphere, actions, and tone of a given sentence or text from a book.
The description should be tailored to create a sound effect using ElevenLabs'sound generation API.
The generated sound description must evoke the scene
or emotions from the text (e.g., footsteps, wind, tense silence, etc.),
and it should be succinct and fit the mood of the text.

Additionally, you should include the following parameters in your response:

    Text: A generated description of the sound that matches the text provided.
        Keep the description simple and effective to capture the soundscape.
        This text will be converted into a sound effect. 
    Prompt_influence: A value between 0 and 1, where a higher value makes the sound generation closely
        follow the sound description. For general sound effects (e.g., footsteps, background ambiance),
        use a value around 0.3. For more specific or detailed sound scenes
        (e.g., thunderstorm, battle sounds), use a higher value like 0.5 to 0.7.

Your output should be in the following JSON format:

{
  "text": "A soft breeze rustling through leaves, distant birds chirping.",
  "prompt_influence": 0.4
}

"""

TEXT_MODIFICATION = """
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