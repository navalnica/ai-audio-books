from src.sound_effects_design import SoundEffectDescription
from src.text_split_chain import CharacterPhrase
from src.utils import (
    get_audio_from_voice_id,
    get_character_color,
    get_collection_safe_index,
    hex_to_rgb,
    prettify_unknown_character_label,
)
from src.web.variables import EFFECT_CSS


def create_status_html(status: str, steps: list[tuple[str, bool]], error_text: str = '') -> str:
    # CSS for the spinner animation
    spinner_css = """
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .spinner {
            width: 20px;
            height: 20px;
            border: 3px solid #e0e0e0;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            display: inline-block;
        }
    """

    spinner_div = "<div class='spinner'></div>"
    steps_html = "\n".join(
        [
            f'<div class="step-item" style="display: flex; align-items: center; padding: 0.8rem; margin-bottom: 0.5rem; background-color: #31395294; border-radius: 6px; font-weight: 600;">'
            f'<span class="step-icon" style="margin-right: 1rem; font-size: 1.3rem;">'
            f'{"✅" if completed else spinner_div}'
            f'</span>'
            f'<span class="step-text" style="font-size: 1.1rem; color: #e0e0e0;">{step}</span>'
            f'</div>'
            for step, completed in steps
        ]
    )

    # status_description = '<p class="status-description" style="margin: 0.5rem 0 0 0; color: #c0c0c0; font-size: 1rem; font-weight: 400;">Processing steps below.</p>'
    status_description = ''

    if error_text:
        error_html = f'<div class="error-message" style="color: #e53e3e; font-size: 1.2em;">{error_text}</div></div>'
    else:
        error_html = ''

    return f'''
    <div class="status-container" style="font-family: system-ui; max-width: 1472px; margin: 0 auto; background-color: #31395294; padding: 1rem; border-radius: 8px; color: #f0f0f0;">
        <style>{spinner_css}</style>
        <div class="status-header" style="background: #31395294; padding: 1rem; border-radius: 8px; font-weight: bold;">
            <h3 class="status-title" style="margin: 0; color: rgb(224, 224, 224); font-size: 1.5rem; font-weight: 700;">Status: {status}</h3>
            {status_description}
            {error_html}
        </div>
        <div class="steps" style="margin-top: 1rem;">
            {steps_html}
        </div>
    </div>
    '''


def create_effect_span_prefix_postfix(effect_description: str):
    """Create an HTML span with effect tooltip."""
    # NOTE: it's important not to use multiline python string in order not to add whitespaces
    prefix = (
        '<span class="character-segment">'
        '<span class="effect-container">'
        '<span class="effect-text">'
    )

    postfix = (
        '</span>'
        f'<span class="effect-tooltip">Effect: {effect_description}</span>'
        '</span>'
        '</span>'
    )

    return prefix, postfix


def create_effect_span(text: str, effect_description: str) -> str:
    prefix, postfix = create_effect_span_prefix_postfix(effect_description=effect_description)
    res = f"{prefix}{text}{postfix}"
    return res


def create_regular_span(text: str, bg_color: str) -> str:
    """Create a regular HTML span with background color."""
    return f'<span class="character-segment" style="background-color: {bg_color}">{text}</span>'


def _generate_legend_for_text_split_html(
    character_phrases: list[CharacterPhrase], add_effect_legend: bool = False
) -> str:
    html = (
        "<div style='margin-bottom: 1rem;'>"
        "<div style='font-size: 1.35em; font-weight: bold;'>Legend:</div>"
    )

    unique_characters = set(phrase.character or 'Unassigned' for phrase in character_phrases)
    characters_sorted = sorted(unique_characters, key=lambda c: c.lower())

    for character in characters_sorted:
        color = get_character_color(character)
        html += f"<div style='color: {color}; font-size: 1.1em; margin-bottom: 0.25rem;'>{character}</div>"

    if add_effect_legend:
        html += (
            '<div style="font-size: 1.1em; margin-bottom: 0.25rem;">'
            '<span class="effect-text">🎵 #1</span>'
            ' - sound effect start position (hover to see the prompt)'
            '</div>'
        )

    html += "</div>"
    return html


def _generate_text_split_html(
    character_phrases: list[CharacterPhrase],
) -> tuple[str, dict[int, int]]:
    html_items = ["<div style='font-size: 1.2em; line-height: 1.6;'>"]

    index_mapping = {}  # Mapping from original index to HTML index
    orig_index = 0  # Index in the original text
    html_index = len(html_items[0])  # Index in the HTML output

    for phrase in character_phrases:
        character = phrase.character or 'Unassigned'
        text = phrase.text
        color = get_character_color(character)
        rgba_color = f"rgba({hex_to_rgb(color)}, 0.5)"

        prefix = f"<span style='background-color: {rgba_color}; border-radius: 0.2em;'>"
        suffix = '</span>'

        # Append the HTML for this phrase
        html_items.append(f"{prefix}{text}{suffix}")

        # Map each character index from the original text to the HTML text
        html_index += len(prefix)
        for i in range(len(text)):
            index_mapping[orig_index + i] = html_index + i
        # Update indices
        orig_index += len(text)
        html_index += len(text) + len(suffix)

    html_items.append("</div>")

    html = ''.join(html_items)
    return html, index_mapping


def generate_text_split_inner_html_no_effect(character_phrases: list[CharacterPhrase]) -> str:
    legend_html = _generate_legend_for_text_split_html(
        character_phrases=character_phrases, add_effect_legend=False
    )
    text_split_html, char_ix_orig_2_html = _generate_text_split_html(
        character_phrases=character_phrases
    )
    return legend_html + text_split_html


def generate_text_split_inner_html_with_effects(
    character_phrases: list[CharacterPhrase],
    sound_effects_descriptions: list[SoundEffectDescription],
) -> str:
    legend_html = _generate_legend_for_text_split_html(
        character_phrases=character_phrases, add_effect_legend=True
    )
    text_split_html, char_ix_orig_2_html = _generate_text_split_html(
        character_phrases=character_phrases
    )

    if not sound_effects_descriptions:
        return legend_html + text_split_html

    prev_end = 0
    content_html_parts = []
    for ix, sed in enumerate(sound_effects_descriptions, start=1):
        # NOTE: 'sed' contains approximate indices from the original text.
        # that's why we use safe conversion before accessing char mapping
        ix_start = get_collection_safe_index(
            ix=sed.ix_start_orig_text, collection=char_ix_orig_2_html
        )
        # ix_end = get_collection_safe_index(ix=sed.ix_end_orig_text, collection=char_ix_orig_2_html)

        html_start_ix = char_ix_orig_2_html[ix_start]
        # html_end_ix = char_ix_orig_2_html[ix_end]  # NOTE: this is incorrect
        # BUG: here we take exact same number of characters as in text between sound effect tags.
        # This introduces the bug: HTML text could be included in 'text_under_effect',
        # due to inaccuracies in 'sed' indices.
        # html_end_ix = html_start_ix + ix_end - ix_start  # NOTE: this is correct
        # NOTE: reason is that html may exist between original text characters

        prefix = text_split_html[prev_end:html_start_ix]
        if prefix:
            content_html_parts.append(prefix)

        # text_under_effect = text_split_html[html_start_ix:html_end_ix]
        text_under_effect = f'🎵 #{ix}'
        if text_under_effect:
            effect_prefix, effect_postfix = create_effect_span_prefix_postfix(
                effect_description=sed.prompt
            )
            text_under_effect_wrapped = f'{effect_prefix}{text_under_effect}{effect_postfix}'
            content_html_parts.append(text_under_effect_wrapped)

        # prev_end = html_end_ix
        prev_end = html_start_ix

    last = text_split_html[prev_end:]
    if last:
        content_html_parts.append(last)

    content_html = ''.join(content_html_parts)
    content_html = f'{EFFECT_CSS}<div class="text-effect-container">{content_html}</div>'
    html = legend_html + content_html
    return html


def generate_voice_mapping_inner_html(select_voice_chain_out):
    character2props = {}
    html = AUDIO_PLAYER_CSS

    for key in set(select_voice_chain_out.character2props) | set(
        select_voice_chain_out.character2voice
    ):
        character_props = select_voice_chain_out.character2props.get(key, []).model_dump()
        character_props["voice_id"] = select_voice_chain_out.character2voice.get(key, [])
        character_props["sample_audio_url"] = get_audio_from_voice_id(character_props["voice_id"])

        character2props[prettify_unknown_character_label(key)] = character_props

    for character, voice_properties in sorted(character2props.items(), key=lambda x: x[0].lower()):
        color = get_character_color(character)
        audio_url = voice_properties.get('sample_audio_url', '')

        html += f'''
                <div class="voice-assignment">
                    <div class="voice-details">
                        <span class="character-name" style="color: {color};">{character}</span>
                        <span>→</span>
                        <span class="voice-props">
                            Gender: {voice_properties.get('gender', 'N/A')}, 
                            Age: {voice_properties.get('age_group', 'N/A')}, 
                            Voice ID: {voice_properties.get('voice_id', 'N/A')}
                        </span>
                    </div>
                    <div class="custom-audio-player">
                        <audio controls preload="none">
                            <source src="{audio_url}" type="audio/mpeg">
                            Your browser does not support the audio element.
                        </audio>
                    </div>
                </div>
            '''

    return html


AUDIO_PLAYER_CSS = """\
<style>
    .custom-audio-player {
        display: inline-block;
        width: 250px;
        --bg-color: #ff79c6;
        --highlight-color: #4299e100;
        --text-color: #e0e0e0;
        --border-radius: 0px;
    }

    .custom-audio-player audio {
        width: 100%;
        height: 36px;
        border-radius: var(--border-radius);
        background-color: #3f2a2a00;
        outline: none;
    }

    .custom-audio-player audio::-webkit-media-controls-panel {
        background-color: var(--bg-color);
    }

    .custom-audio-player audio::-webkit-media-controls-current-time-display,
    .custom-audio-player audio::-webkit-media-controls-time-remaining-display {
        color: var(--text-color);
    }

    .custom-audio-player audio::-webkit-media-controls-play-button {
        background-color: var(--highlight-color);
        border-radius: 50%;
        height: 30px;
        width: 30px;
    }

    .custom-audio-player audio::-webkit-media-controls-timeline {
        background-color: var(--bg-color);
        height: 6px;
        border-radius: 3px;
    }

    /* Container styles for voice assignment display */
    .voice-assignment {
        background-color: rgba(49, 57, 82, 0.8);
        padding: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 0.2rem;
        padding-bottom: 0.2rem;
        border-radius: var(--border-radius);
        margin-top: 0.5rem;
        color: var(--text-color);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
        border-radius: 7px;
    }

    .voice-assignment span {
        font-weight: 600;
    }

    .voice-details {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .character-name {
        color: var(--highlight-color);
        font-weight: bold;
    }

    .voice-props {
        color: #4a5568;
    }
</style>
"""
