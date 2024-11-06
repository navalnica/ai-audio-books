from src.config import ELEVENLABS_API_KEY

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

STATUS_DISPLAY_HTML = '''
        <style>
          .status-container {
              font-family: system-ui;
              max-width: 1472;
              margin: 0 auto;
              background-color: #31395294; /* Darker background color */
              padding: 1rem;
              border-radius: 8px;
              color: #f0f0f0; /* Light text color */
          }
          .status-header {
              background: #31395294; /* Slightly lighter background */
              padding: 1rem;
              border-radius: 8px;
              font-weight: bold; /* Emphasize header */
          }
          .status-title {
              margin: 0;
              color: rgb(224, 224, 224); /* White color for title */
              font-size: 1.5rem; /* Larger title font */
              font-weight: 700; /* Bold title */
          }
          .status-description {
              margin: 0.5rem 0 0 0;
              color: #c0c0c0;
              font-size: 1rem;
              font-weight: 400; /* Regular weight for description */
          }
          .steps {
              margin-top: 1rem;
          }
          .step-item {
              display: flex;
              align-items: center;
              padding: 0.8rem;
              margin-bottom: 0.5rem;
              background-color: #31395294; /* Matching background color */
              border-radius: 6px;
              color: #f0f0f0; /* Light text color */
              font-weight: 600; /* Medium weight for steps */
          }
          .step-item:hover {
              background-color: rgba(255, 255, 255, 0.07);
          }
          .step-icon {
              margin-right: 1rem;
              font-size: 1.3rem; /* Slightly larger icon size */
          }
          .step-text {
              font-size: 1.1rem; /* Larger text for step description */
              color: #e0e0e0; /* Lighter text for better readability */
          }
        </style>

        <div class="status-container">
            <div class="status-header">
                <h2 class="status-title">Status: Waiting to Start</h2>
                <p class="status-description">Enter text or upload a file to begin.</p>
            </div>
            <div class="steps">
                <div class="step-item">
                    <span class="step-icon">📚</span>
                    <span class="step-text">Split text into characters</span>
                </div>
                <div class="step-item">
                    <span class="step-icon">🎭</span>
                    <span class="step-text">Assign each character a voice</span>
                </div>
                <!-- Add more steps as needed -->
            </div>
        </div>
        '''
GRADIO_THEME = "freddyaboulton/dracula_revamped"

VOICE_UPLOAD_JS = f"""
async function createVoiceUploadPopup() {{
    try {{
        let savedVoiceId = null;
        const result = await new Promise((resolve, reject) => {{
            // Create overlay with soft animation
            const overlay = document.createElement('div');
            Object.assign(overlay.style, {{
                position: 'fixed',
                top: '0',
                left: '0',
                width: '100%',
                height: '100%',
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                zIndex: '1000',
                opacity: '0',
                transition: 'opacity 0.3s ease-in-out'
            }});

            overlay.offsetHeight; // Trigger reflow for transition
            overlay.style.opacity = '1';

            // Create popup container with modern design
            const popup = document.createElement('div');
            Object.assign(popup.style, {{
                backgroundColor: '#3b4c63',
                padding: '30px',
                borderRadius: '12px',
                width: '450px',
                maxWidth: '95%',
                position: 'relative',
                boxShadow: '0 10px 25px rgba(0, 0, 0, 0.3)',
                transform: 'scale(0.9)',
                transition: 'transform 0.3s ease-out',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center'
            }});

            popup.offsetHeight; // Trigger reflow
            popup.style.transform = 'scale(1)';

            // Create close button
            const closeBtn = document.createElement('button');
            Object.assign(closeBtn.style, {{
                position: 'absolute',
                right: '15px',
                top: '15px',
                border: 'none',
                background: 'none',
                fontSize: '24px',
                cursor: 'pointer',
                color: '#d3d3d3',
                transition: 'color 0.2s ease'
            }});
            closeBtn.innerHTML = '✕';
            closeBtn.onmouseover = () => closeBtn.style.color = '#ffffff';
            closeBtn.onmouseout = () => closeBtn.style.color = '#d3d3d3';

            // Create content
            const content = document.createElement('div');
            content.innerHTML = `
                <div style="text-align: center; margin-bottom: 25px;">
                    <h2 style="color: #ffffff; margin: 0; font-size: 22px;">Upload Voice Sample</h2>
                    <p style="color: #b0b0b0; margin-top: 10px; font-size: 14px;">
                        Select an audio file to create audiobook with your unique voice.
                    </p>
                </div>
                <div style="margin-bottom: 20px; display: flex; flex-direction: column; align-items: center; width: 100%;">
                    <label for="voiceFile" style="
                        display: block; 
                        margin-bottom: 10px; 
                        color: #c0c0c0; 
                        font-weight: 600;
                        text-align: center;">
                        Choose Audio File (MP3, WAV, OGG):
                    </label>
                    <input type="file" id="voiceFile" accept="audio/*" 
                           style="
                               width: 100%; 
                               padding: 12px; 
                               border: 2px dashed #4a6f91; 
                               border-radius: 8px; 
                               background-color: #2a3a50;
                               color: #ffffff;
                               text-align: center;
                               transition: border-color 0.3s ease;
                           ">
                </div>
                <div id="uploadStatus" style="
                    margin-bottom: 15px; 
                    text-align: center; 
                    min-height: 25px; 
                    color: #d3d3d3;">
                </div>
                <button id="uploadBtn" style="
                    background-color: #4a6f91; 
                    color: #ffffff; 
                    padding: 12px 20px; 
                    border: none; 
                    border-radius: 8px; 
                    cursor: pointer; 
                    width: 100%;
                    font-weight: 600; 
                    transition: background-color 0.3s ease, transform 0.1s ease;
                ">
                    Upload Voice
                </button>
            `;

            // Add elements to DOM
            popup.appendChild(closeBtn);
            popup.appendChild(content);
            overlay.appendChild(popup);
            document.body.appendChild(overlay);

            // Button effects
            const uploadBtn = popup.querySelector('#uploadBtn');
            uploadBtn.onmouseover = () => uploadBtn.style.backgroundColor = '#3b5c77';
            uploadBtn.onmouseout = () => uploadBtn.style.backgroundColor = '#4a6f91';
            uploadBtn.onmousedown = () => uploadBtn.style.transform = 'scale(0.98)';
            uploadBtn.onmouseup = () => uploadBtn.style.transform = 'scale(1)';

            // Handle close
            const handleClose = () => {{
                overlay.style.opacity = '0';
                setTimeout(() => {{
                    overlay.remove();
                    resolve(savedVoiceId);
                }}, 300);
            }};

            closeBtn.onclick = handleClose;
            overlay.onclick = (e) => {{
                if (e.target === overlay) {{
                    handleClose();
                }}
            }};

            // Handle file upload
            const statusDiv = popup.querySelector('#uploadStatus');
            const fileInput = popup.querySelector('#voiceFile');

            uploadBtn.onclick = async () => {{
                const file = fileInput.files[0];
                if (!file) {{
                    statusDiv.textContent = 'Please select a file first.';
                    statusDiv.style.color = '#e74c3c';
                    return;
                }}

                const API_KEY = "{ELEVENLABS_API_KEY}";

                statusDiv.textContent = 'Uploading...';
                statusDiv.style.color = '#4a6f91';
                uploadBtn.disabled = true;
                uploadBtn.style.backgroundColor = '#6c8091';

                const formData = new FormData();
                formData.append('files', file);
                formData.append('name', `voice_${{Date.now()}}`);

                try {{
                    const response = await fetch('https://api.elevenlabs.io/v1/voices/add', {{
                        method: 'POST',
                        headers: {{
                            'Accept': 'application/json',
                            'xi-api-key': API_KEY
                        }},
                        body: formData
                    }});

                    const result = await response.json();

                    if (response.ok) {{
                        savedVoiceId = result.voice_id
                        statusDiv.innerHTML = `
                            <div style="
                                background-color: #2e3e50; 
                                color: #00b894; 
                                padding: 10px; 
                                border-radius: 6px; 
                                font-weight: 600;
                            ">
                                Voice uploaded successfully! 
                                <br>Your Voice ID: <span style="color: #0984e3;">${{result.voice_id}}</span>
                            </div>
                        `;

                        // Update the visible HTML panel
                        const voiceIdPanel = document.querySelector('#voice_id_panel');
                        if (voiceIdPanel) {{
                            voiceIdPanel.innerHTML = `<strong>Your voice_id from uploaded audio is </strong> <span style="color: #0984e3;">${{result.voice_id}}</span>`;
                        }}

                        setTimeout(() => {{
                            overlay.style.opacity = '0';
                            setTimeout(() => {{
                                overlay.remove();
                                resolve(result.voice_id);  // Resolve with the voice ID
                            }}, 300);
                        }}, 3000);
                    }} else {{
                        throw new Error(result.detail?.message || 'Upload failed');
                    }}
                }} catch (error) {{
                    statusDiv.innerHTML = `
                        <div style="
                            background-color: #3b4c63; 
                            color: #d63031; 
                            padding: 10px; 
                            border-radius: 6px; 
                            font-weight: 600;
                        ">
                            Error: ${{error.message}}
                        </div>
                    `;
                    uploadBtn.disabled = false;
                    uploadBtn.style.backgroundColor = '#4a6f91';
                }}
            }};
        }});

        return result;  // Return the voice ID from the Promise
    }} catch (error) {{
        console.error('Error in createVoiceUploadPopup:', error);
        return null;
    }}
}}
"""

EFFECT_CSS = """\
<style>
    .text-effect-container {
        line-height: 1.6;
    }

    .character-segment {
        border-radius: 0.2em;
    }

    .effect-container {
        position: relative;
        display: inline-block;
    }

    .effect-text {
        padding: 2px 4px;
        border-radius: 0px;
        border-bottom: 3px solid rgba(187, 185, 81, 0.97);
        cursor: help;
    }

    .effect-tooltip {
        visibility: hidden;
        background-color: #333;
        color: white;
        text-align: center;
        padding: 5px 10px;
        border-radius: 6px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        white-space: nowrap;
        opacity: 0;
        transition: opacity 0.3s;
    }

    .effect-tooltip::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #333 transparent transparent transparent;
    }

    .effect-container:hover .effect-tooltip {
        visibility: visible;
        opacity: 1;
    }
</style>
"""
