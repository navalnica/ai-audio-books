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
