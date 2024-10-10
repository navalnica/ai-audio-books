class SplitTextPromptV1:
    SYSTEM = """\
You are a helpful assistant proficient in literature and language.
Imagine you are helping to prepare the provided text for narration to create the audio book.
We need to understand how many voice actors we need to hire and how to split the text between them.

Your task is to help with this process, namely:
1. Identify all book characters occuring in the text, including "narrator".
We will hire individual voice actor for each one of them.
2. Split the text provided by characters. Let's refer to each split as "part".
Order of parts MUST be the same as in the original text.

Details:
- First, analyze the whole text to extract the list of characters.
Put found characters to corresponding output field.
- Then, analyze the text top-down and as you proceed fill the "parts" field
- Each part must be attributed to a single character.
Character must belong to the "characters" list
- Use "narrator" character for any descriptive or narrative text, 
such as actions ("He shook his head"), narrative parts ("I thought")
thoughts, or descriptions that aren't part of spoken dialogue
- In some books narrator is one of the main characters, having its own name and phrases.
In this case, use regualar character name instead of "narrator" role
- If it's impossible to identify character name from the text provided, use codes "c1", "c2", etc,
where "c" prefix means character and number is used to enumerate unknown characters

Format your answer as a following JSON:
{{
    "characters": [list of unique character names that are found in the text provided],
    "parts":
    [
        {{
            "character": <character name>, "text": <the part's text>
        }}
    ]
}}

Ensure the order of the parts in the JSON output matches the original order of the text.

Examples of text split by characters, already in the target format.

Example 1.
{{
    "characters": ["Mr. Gatz", "narrator"],
    "parts":
    [
        {{"character": "Mr. Gatz", "text": "“Gatz is my name.”"}},
        {{"character": "narrator", "text": "“—Mr. Gatz. I thought you might want to take the body West.” He shook his head."}},
        {{"character": "Mr. Gatz", "text": "“Jimmy always liked it better down East. He rose up to his position in the East. Were you a friend of my boy’s, Mr.—?”"}},
        {{"character": "narrator", "text": "“We were close friends.”"}},
        {{"character": "Mr. Gatz", "text": "“He had a big future before him, you know. He was only a young man, but he had a lot of brain power here.”"}},
        {{"character": "narrator", "text": "He touched his head impressively, and I nodded."}},
        {{"character": "Mr. Gatz", "text": "“If he’d of lived, he’d of been a great man. A man like James J. Hill. He’d of helped build up the country.”"}},
        {{"character": "narrator", "text": "“That’s true,” I said, uncomfortably."}},
        {{"character": "Mr. Gatz", "text": "He fumbled at the embroidered coverlet, trying to take it from the bed, and lay down stiffly—was instantly asleep."}},
    ]
}}

Example 2.
{{
    'characters': [
        'narrator',
        'Mr. Carraway',
        'Daisy',
        'Miss Baker',
        'Tom',
        'Nick'
    ],
    'parts': [
        {{'character': 'narrator', 'text': '“If you’ll get up.”'}},
        {{'character': 'Mr. Carraway', 'text': '“I will. Good night, Mr. Carraway. See you anon.”'}},
        {{'character': 'Daisy', 'text': '“Of course you will,” confirmed Daisy. “In fact I think I’ll arrange a marriage. Come over often, Nick, and I’ll sort of—oh—fling you together. You know—lock you up accidentally in linen closets and push you out to sea in a boat, and all that sort of thing—”'}},
        {{'character': 'Miss Baker', 'text': '“Good night,” called Miss Baker from the stairs. “I haven’t heard a word.”'}},
        {{'character': 'Tom', 'text': '“She’s a nice girl,” said Tom after a moment. “They oughtn’t to let her run around the country this way.”'}},
        {{'character': 'Daisy', 'text': '“Who oughtn’t to?” inquired Daisy coldly.'}},
        {{'character': 'narrator', 'text': '“Her family.”'}},
        {{'character': 'narrator', 'text': '“Her family is one aunt about a thousand years old. Besides, Nick’s going to look after her, aren’t you, Nick? She’s going to spend lots of weekends out here this summer. I think the home influence will be very good for her.”'}},
        {{'character': 'narrator', 'text': 'Daisy and Tom looked at each other for a moment in silence.'}}
    ]
}}
"""

    USER = """\
Here is the book sample:
---
{text}"""


class SplitTextPromptV2:
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
