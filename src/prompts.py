class SplitTextPrompt:
    SYSTEM = """\
you are provided with the book sample.
please rewrite it and insert xml tags indicating character to whom current phrase belongs.
for example: <narrator>I looked at her</narrator><Jill>What are you looking at?</Jill>

Notes:
- sometimes narrator is one of characters taking part in the action.
in this case use narrator's name (if available) instead of "narrator"
- if it's impossible to identify character name from the text provided, use codes "c1", "c2", etc,
where "c" prefix means character and number is used to enumerate unknown characters
- all quotes of direct speech must be attributed to characters, for example:
<Tom>“She’s a nice girl,”</Tom><narrator>said Tom after a moment.</narrator>
mind that sometimes narrator could also be a character.
- use ALL available context to determine the character.
sometimes the character name becomes clear from the following phrases
- DO NOT include in your response anything except for the original text with character xml tags!!!
"""

    USER = """\
Here is the book sample:
---
{text}"""


class ModifyTextPrompt:
    SYSTEM = """\
You are provided with the book sample.
You should help me to make an audiobook with exaggerated emotion-based voice using Text-to-Speech models.
Your task is to adjust the emotional tone of a given text by modifying the text in the following ways:
- add special characters: "!" (adds emphasis), "?" (enhances question intonation), "..." (adds pause)
- write words in uppercase - to add emphasis or convey emotion

For example:
Text: "I can't believe this is happening. Who would expect it?"
Output text: "I CAN'T believe this is happening... Who would expect it??"

Notes:
- Do not remove or add any words!
- You are allowed ONLY to add "!", "?", "..." symbols and re-write existing words in uppercase!
- To add more emotions, you can duplicate exclamation or question marks, for example: "!!!" or "???"
- DO NOT place "!" or "?" INSIDE existing sentences, since it breaks the sentence in parts
- Be generous on pauses between sentences or between the obviously different parts of the same sentence.
Reason is TTS model tends to dub with fast speed. 
- But don't add too many pauses within one sentence! Add them only where needed.
- Remember: sentences must sound naturally, in the way a profession voice actor would read it!
- DO NOT add pauses in the very end of the given text!
"""

    USER = """\
Here is the book sample:
---
{text}"""


class CharacterVoicePropertiesPrompt:
    SYSTEM = """\
You are a helpful assistant proficient in literature and psychology.
Our goal is to create an audio book from the given text.
For that we need to hire voice actors.
Please help us to find the right actor for each character present in the text.

You are provided with the text split by the characters
to whom text parts belong to.

Your task is to assign available properties to each character provided.
List of available properties:
- gender: {available_genders}
- age_group: {available_age_groups}

NOTES:
- assign EXACTLY ONE property value for each property
- select properties values ONLY from the list of AVAILABLE property values
- fill properties for ALL characters from the list provided
- DO NOT include any characters absent in the list provided

{format_instructions}
"""

    # You MUST answer with the following JSON:
    # {{
    #     "character2props":
    #     {{
    #         <character_name>:
    #         {{
    #             "gender": <value>,
    #             "age_group": <value>
    #         }}
    #     }}
    # }}

    USER = """\
<text>
{text}
</text>

<characters>
{characters}
</characters>
"""


class SoundEffectsPrompt:
    SYSTEM = """\
You are an expert in directing audiobooks creation.
Your task is to design sound effects (by writing their text description) layed over the voice actors narration.
Sound effects descriptions are going to be passed to text-to-sound-effect AI model.
Sound effects must enhance storytelling and evoke immersive experience in listeners.

You are provided with the audiobook text chunk -
you must insert XML tags containing prompts for AI model describing sound effects.

XML effect tags must have following structure:
<effect prompt="prompt to be passed to text-to-sound-effect AI model">original line from the text</effect>

WRITE PROMPTS TO BE VERY RICH IN DETAILS, precisely describing the effect!
Your prompts MUST BE SPECIFIC, AVOID ABSTRACT sounds like "sound of a cozy room".

Generated sound effect will be overlayed over the text between the opening and the closing effect XML tag.
Use tags position to control start time of the effect and its duration.

Additional requirements:
- In the very beginning, analyze the whole text chunk provided in order to understand events and atmosphere.
- Aim for episodical sound effects, highlighting atmosphere and characters' actions.
For example, cracking of stairs, wind blowing, car honks, sound of a falling book, ticking clock
- NEVER generate background music
- NEVER generate ambient sounds, for example people's voices, sound of the crowd
- NEVER generate sounds for gestures, for example for a hand raised in the air.
- NEVER generate effects for sounds people may produce: laughing, giggling, sobbing, crying, talking, singing, screaming.
- NEVER generate silence, since it's a too abstract effect
- The text-to-sound-effects model is able to generate only short audio files, up to 5 seconds long
- Aim to position sound effects at the most intuitive points for a natural, immersive experience.
For example, instead of placing the sound effect only on a single word or object (like "stairs"),
tag a broader phrase making the effect feel part of the action or dialogue.
- It's allowed to add no sound effects

Examples of bad prompts:
1. "brief silence, creating a moment of tension" - it's too short, not specific and is an ambient sound.
2. "brief, tense silence, filled with unspoken words and a slight undercurrent of tension" - very abstract, and breaks the rule of not generating silence
3. "sudden burst of bright light filling a room, creating a warm and inviting atmosphere" - abstract
4. "sudden, commanding gesture of a hand being lifted, creating a brief pause in conversation" - abstract
5. "exaggerated, humorous emphasis on the age, suggesting an ancient, creaky presence"

Examples of good prompts:
1. "soft rustling of paper as a page is turned, delicate and gentle"
2. "light thud of a magazine landing on a wooden table, slightly echoing in the quiet room"
3. "Old wooden staircase creaking under slow footsteps, each step producing uneven crackles, groans, and occasional sharp snaps, emphasizing age and fragility in a quiet, echoing space" - it's specific and rich in details

Response with the original text with selected phrases wrapped inside emotion XML tags.
Do not modify original text!
Do not include anythin else in your answer.
"""

    USER = """\
{text}
"""


# TODO: this prompt is not used
PREFIX = """\
You should help me to make an audiobook with realistic emotion sound using TTS.
You are tasked with generating a description of sound effects
that matches the atmosphere, actions, and tone of a given sentence or text from a book.
The description should be tailored to create a sound effect using ElevenLabs'sound generation API.
The generated sound description must evoke the scene
or emotions from the text (e.g., footsteps, wind, tense silence, etc.),
and it should be succinct and fit the mood of the text."""

# TODO: this prompt is not used
SOUND_EFFECT_GENERATION = f"""
{PREFIX}

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

{{
  "text": "A soft breeze rustling through leaves, distant birds chirping.",
  "duration_seconds": 4.0,
  "prompt_influence": 0.4
}}

NOTES:
- NEVER add any speech or voices in your instructions!
- NEVER add any music in your instructions!
- NEVER add city sounds, car honks in your instructions!
- make your text descriptions VERY SPECIFIC, AVOID vague instructions.
If it's necessary, you can use couple sentences to formulate the instruction.
But remember to use keep instructions simple.
- aim to create specific sounds, like crackling fireplace, footsteps, wind, etc...
"""

# TODO: this prompt is not used
SOUND_EFFECT_GENERATION_WITHOUT_DURATION_PREDICTION = f"""
{PREFIX}

Additionally, you should include the following parameters in your response:

    Text: A generated description of the sound that matches the text provided.
        Keep the description simple and effective to capture the soundscape.
        This text will be converted into a sound effect. 
    Prompt_influence: A value between 0 and 1, where a higher value makes the sound generation closely
        follow the sound description. For general sound effects (e.g., footsteps, background ambiance),
        use a value around 0.3. For more specific or detailed sound scenes
        (e.g., thunderstorm, battle sounds), use a higher value like 0.5 to 0.7.

Your output should be in the following JSON format:

{{
  "text": "A soft breeze rustling through leaves, distant birds chirping.",
  "prompt_influence": 0.4
}}"""


EMOTION_STABILITY_MODIFICATION = """
You should help me to make an audiobook with exaggerated emotion-based voice using Text-to-Speech.
Your single task it to select "stability" TTS parameter value,
based on the emotional intensity level in the provided text chunk.

Provided text was previously modified by uppercasing some words and adding "!", "?", "..." symbols.
The more there are uppercase words or "!", "?", "..." symbols, the higher emotional intensity level is.
Higher emotional intensity must be associated with lower values of "stability" parameter,
and lower emotional intensity must be associated with higher "stability" values.
Low "stability" makes TTS to generate more expressive, less stable speech - better suited to convey emotional range.

Available range for "stability" values is [0.3; 0.8].

You MUST answer with the following JSON,
containing a SINGLE "stability" parameter with selected value:
{"stability": float}
DO NOT INCLUDE ANYTHING ELSE in your response.

Example:
Input: "I CAN'T believe this is happening... Who would expect it??"
Expected output: {"stability": 0.4}
"""

# TODO: this prompt is not used
TEXT_MODIFICATION_WITH_SSML = """
You should help me to make an audiobook with overabundant emotion-based voice using TTS.
You are tasked with transforming the text provided into a sophisticated SSML script 
that is optimized for emotionally, dramatically and breathtaking rich audiobook narration. 
Analyze the text for underlying emotions, detect nuances in intonation, and discern the intended impact. 
Apply suitable SSML enhancements to ensure that the final TTS output delivers 
a powerful, engaging, dramatic and breathtaking listening experience appropriate for an audiobook context 
(more effects/emotions are better than less)."

Please, use only provided SSML tags and don't generate any other tags.
Key SSML Tags to Utilize:
<speak>: This is the root element. All SSML content to be synthesized must be enclosed within this tag.
<prosody>: Manipulates pitch, rate, and volume to convey various emotions and emphases. Use this tag to adjust the voice to match the mood and tone of different parts of the narrative.
<break>: Inserts pauses of specified durations. Use this to create natural breaks in speech, aiding in dramatic effect and better comprehension for listeners.
<emphasis>: Adds stress to words or phrases to highlight key points or emotions, similar to vocal emphasis in natural speech.
<p> and <s>: Structural tags that denote paragraphs and sentences, respectively. They help to manage the flow and pacing of the narrative appropriately.

Input Text Example: "He stood there, gazing into the endless horizon. As the sun slowly sank, painting the sky with hues of orange and red, he felt a sense of deep melancholy mixed with awe."

Modified text should be in the XML format. Expected SSML-enriched Output:

<speak>
    <p>
        <s>
            He stood there, <prosody rate="slow" volume="soft">gazing into the endless horizon.</prosody>
        </s>
        <s>
            As the sun slowly <prosody rate="medium" pitch="-2st">sank,</prosody> 
            <prosody volume="medium" pitch="+1st">painting the sky with hues of orange and red,</prosody> 
            he felt a sense of deep <prosody volume="soft" pitch="-1st">melancholy</prosody> mixed with <emphasis level="moderate">awe.</emphasis>
        </s>
    </p>
</speak>

After modifying the text, adjust the "stability", "similarity_boost" and "style" parameters
according to the level of emotional intensity in the modified text.
Higher emotional intensity should lower the "stability" and raise the "similarity_boost". 
Your output should be in the following JSON format:
 {
  "modified_text": "Modified text in xml format with SSML tags.",
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
"""
