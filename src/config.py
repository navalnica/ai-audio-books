import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s): %(message)s",
)
logger = logging.getLogger("audio-books")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ELEVENLABS_API_KEY = os.environ["ELEVEN_LABS_API_KEY"]

FILE_SIZE_MAX = 0.5  # in mb

OPENAI_MAX_PARALLEL = 8  # empirically set
ELEVENLABS_MAX_PARALLEL = 15  # current limitation of available subscription

# VOICES_CSV_FP = "data/11labs_available_tts_voices.csv"
VOICES_CSV_FP = "data/11labs_available_tts_voices.reviewed.csv"

MAX_TEXT_LEN = 5000

DESCRIPTION = """\
# AI Audiobooks Generator

Create an audiobook from the input text automatically, using Gen-AI!

All you need to do - is to input the book text or select it from the provided Sample Inputs.

AI will do the rest:
- Split text into characters
- Assign each character a voice
- Enhance text to convey emotions and intonations during Text-to-Speech
- Generate audiobook using Text-to-Speech model
- Generate sound effects to create immersive atmosphere (optional)
"""

DESCRIPTION_JS = """function createGradioAnimation() {
    // Create main container
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.padding = '2rem';
    container.style.background = 'transparent';
    container.style.borderRadius = '12px';
    container.style.margin = '0 0 2rem 0';
    container.style.maxWidth = '100%';
    container.style.transition = 'all 0.3s ease';

    // Create header section
    var header = document.createElement('div');
    header.style.textAlign = 'center';
    header.style.marginBottom = '2rem';
    container.appendChild(header);

    // Title with spaces
    var titleText = 'AI   Audio   Books';
    var title = document.createElement('h1');
    title.style.fontSize = '2.5rem';
    title.style.fontWeight = '700';
    title.style.color = '#f1f1f1';
    title.style.marginBottom = '1.5rem';
    title.style.opacity = '0'; // Start with opacity 0
    title.style.transition = 'opacity 0.5s ease'; // Add transition
    title.innerText = titleText;
    header.appendChild(title);

    // Add description
    var description = document.createElement('p');
    description.innerHTML = `
        <div style="font-size: 1.1rem; color: #c0c0c0; margin-bottom: 2rem; line-height: 1.6;">
            Create an audiobook from the input text automatically, using Gen-AI!<br>
            All you need to do - is to input the book text or select it from the provided Sample Inputs.
        </div>
    `;
    description.style.opacity = '0';
    description.style.transition = 'opacity 0.5s ease';
    header.appendChild(description);

    // Create process section
    var processSection = document.createElement('div');
    processSection.style.backgroundColor = 'rgba(255, 255, 255, 0.05)';
    processSection.style.padding = '1.5rem';
    processSection.style.borderRadius = '8px';
    processSection.style.marginTop = '1rem';
    container.appendChild(processSection);

    // Add "AI will do the rest:" header
    var processHeader = document.createElement('div');
    processHeader.style.fontSize = '1.2rem';
    processHeader.style.fontWeight = '600';
    processHeader.style.color = '#e0e0e0';
    processHeader.style.marginBottom = '1rem';
    processHeader.innerHTML = 'AI will do the rest:';
    processHeader.style.opacity = '0';
    processHeader.style.transition = 'opacity 0.5s ease';
    processSection.appendChild(processHeader);

    // Define steps with icons
    var steps = [
        { text: 'Split text into characters', icon: '📚' },
        { text: 'Assign each character a voice', icon: '🎭' },
        { text: 'Enhance text to convey emotions and intonations during Text-to-Speech', icon: '😊' },
        { text: 'Generate audiobook using Text-to-Speech model', icon: '🎧' },
        { text: 'Generate sound effects to create immersive atmosphere (optional)', icon: '🎵' },
    ];

    // Create steps list
    var stepsList = document.createElement('div');
    stepsList.style.opacity = '0';
    stepsList.style.transition = 'opacity 0.5s ease';
    processSection.appendChild(stepsList);

    steps.forEach(function(step, index) {
        var stepElement = document.createElement('div');
        stepElement.style.display = 'flex';
        stepElement.style.alignItems = 'center';
        stepElement.style.padding = '0.8rem';
        stepElement.style.marginBottom = '0.5rem';
        stepElement.style.backgroundColor = 'rgba(255, 255, 255, 0.03)';
        stepElement.style.borderRadius = '6px';
        stepElement.style.transform = 'translateX(-20px)';
        stepElement.style.opacity = '0';
        stepElement.style.transition = 'all 0.3s ease';

        // Add hover effect
        stepElement.onmouseover = function() {
            this.style.backgroundColor = 'rgba(255, 255, 255, 0.07)';
        };
        stepElement.onmouseout = function() {
            this.style.backgroundColor = 'rgba(255, 255, 255, 0.03)';
        };

        var icon = document.createElement('span');
        icon.style.marginRight = '1rem';
        icon.style.fontSize = '1.2rem';
        icon.innerText = step.icon;
        stepElement.appendChild(icon);

        var text = document.createElement('span');
        text.style.color = '#c0c0c0';
        text.style.fontSize = '1rem';
        text.innerText = step.text;
        stepElement.appendChild(text);

        stepsList.appendChild(stepElement);
    });

    // Insert into Gradio container
    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    // New timing for animations
    setTimeout(function() {
        title.style.opacity = '1';
    }, 250);
    
    // Show description after 1 second
    setTimeout(function() {
        description.style.opacity = '1';
        processHeader.style.opacity = '1';
    }, 700);

    // Show steps after 2 seconds
    setTimeout(function() {
        stepsList.style.opacity = '1';
        stepsList.querySelectorAll('div').forEach(function(step, index) {
            setTimeout(function() {
                step.style.transform = 'translateX(0)';
                step.style.opacity = '1';
            }, index * 100);
        });
    }, 1100);

    async function playAudio(url) {
        try {
            const audio = new Audio(url);
            await audio.play();
        } catch (error) {
            console.error('Error playing audio:', error);
        }
    }
    
    // Add click handler to all audio links
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('audio-link')) {
            e.preventDefault();
            playAudio(e.target.getAttribute('data-audio-url'));
        }
    });

    return 'Animation created';
}"""

GRADIO_THEME = "freddyaboulton/dracula_revamped"
