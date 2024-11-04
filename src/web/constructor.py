from src.web.utils import create_status_html

class HTMLGenerator:
    @staticmethod
    def generate_error(text: str) -> str:
        return create_status_html("Error", []) + f'<div class="error-message" style="color: #e53e3e;">{text}</div></div>'

    @staticmethod
    def generate_status(stage_title: str, steps: list[tuple[str, bool]]) -> str:
        return create_status_html(stage_title, steps) + "</div>"

    @staticmethod
    def generate_text_split(text_split_html: str) -> str:
        return f'''
            <div class="section" style="background-color: #31395294; padding: 1rem; border-radius: 8px; margin-top: 1rem; color: #e0e0e0;">
                <h3 style="color: rgb(224, 224, 224); font-size: 1.5em; margin-bottom: 1rem;">Text Split by Character:</h3>
                {text_split_html}
            </div>
        '''

    @staticmethod
    def generate_voice_assignments(voice_assignments_html: str) -> str:
        return f'''
            <div class="section" style="background-color: #31395294; padding: 1rem; border-radius: 8px; margin-top: 1rem; color: #e0e0e0;">
                <h3 style="color: rgb(224, 224, 224); font-size: 1.5em; margin-bottom: 1rem;">Voice Assignments:</h3>
                {voice_assignments_html}
            </div>
        '''

    @staticmethod
    def generate_message_without_voice_id() -> str:
        return '''
                <div class="audiobook-ready" style="background-color: #31395294; padding: 1rem; border-radius: 8px; margin-top: 1rem; text-align: center;">
                    <h3 style="color: rgb(224, 224, 224); font-size: 1.5em; margin-bottom: 1rem;">🫤 At first you should add your voice</h3>
                </div>
            '''

    @staticmethod
    def generate_final_message() -> str:
        return '''
            <div class="audiobook-ready" style="background-color: #31395294; padding: 1rem; border-radius: 8px; margin-top: 1rem; text-align: center;">
                <h3 style="color: rgb(224, 224, 224); font-size: 1.5em; margin-bottom: 1rem;">🎉 Your audiobook is ready!</h3>
                <p style="color: #4299e1; cursor: pointer;" onclick="document.querySelector('.play-pause-button.icon.svelte-ije4bl').click();">🔊 Press play to listen 🔊</p>
            </div>
        '''