from src.utils import hex_to_rgb, get_character_color, get_audio_from_voice_id, replace_labels
from src.web.variables import EFFECT_CSS


def create_status_html(status: str, steps: list[tuple[str, bool]]) -> str:
    steps_html = "\n".join(
        [
            f'<div class="step-item" style="display: flex; align-items: center; padding: 0.8rem; margin-bottom: 0.5rem; background-color: #31395294; border-radius: 6px; font-weight: 600;">'
            f'<span class="step-icon" style="margin-right: 1rem; font-size: 1.3rem;">{("✅" if completed else "🔄")}</span>'
            f'<span class="step-text" style="font-size: 1.1rem; color: #e0e0e0;">{step}</span>'
            f'</div>'
            for step, completed in steps
        ]
    )

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


def generate_legend_for_text_split_html(
    text_split_dict_list: list[dict], add_effect_legend: bool = False
) -> str:
    legend_html = "<div style='margin-bottom: 1rem;'>"
    legend_html += "<div style='font-size: 1.35em; font-weight: bold;'>Legend:</div>"

    unique_characters = set(item['character'] or 'Unassigned' for item in text_split_dict_list)

    for character in unique_characters:
        color = get_character_color(character)
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

    html_parts = []
    effect_index = 0

    for split_item in text_split_dict_list:
        character = split_item['character'] or 'Unassigned'
        text = split_item['text']
        color = get_character_color(character)
        rgba_color = f"rgba({hex_to_rgb(color)}, 0.3)"

        if (
            effect_index >= len(text_between_effects_texts)
            or text_between_effects_texts[effect_index].lower() not in text.lower()
        ):
            html_parts.append(create_regular_span(text, rgba_color))
        else:
            prev_end = 0
            while (
                effect_index < len(text_between_effects_texts)
                and text_between_effects_texts[effect_index].lower() in text.lower()
            ):
                effect_text = text_with_effects[effect_index]
                text_between_effect_description = text_between_effects_texts[effect_index]

                effect_start = text.lower().find(text_between_effect_description.lower())
                effect_end = effect_start + len(text_between_effect_description)

                html_parts.append(
                    create_regular_span(
                        text[prev_end:effect_start],
                        rgba_color,
                    )
                )
                html_parts.append(
                    create_effect_span(
                        text_between_effect_description,
                        effect_text,
                        rgba_color,
                    )
                )
                effect_index += 1
                prev_end = effect_end

    legend_html = generate_legend_for_text_split_html(text_split_dict_list, add_effect_legend=True)

    content_html = f"""
    {EFFECT_CSS}
    <div class="text-effect-container">
        {''.join(html_parts)}
    </div>
    """
    return legend_html + content_html


def generate_voice_assignments_html(select_voice_chain_out):
    result_voice_chain_out = {}
    for key in set(select_voice_chain_out.character2props) | set(
            select_voice_chain_out.character2voice
    ):
        character_props = select_voice_chain_out.character2props.get(key, []).model_dump()
        character_props["voice_id"] = select_voice_chain_out.character2voice.get(key, [])
        character_props["sample_audio_url"] = get_audio_from_voice_id(
            character_props["voice_id"]
        )

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

        return voice_assignments_html