---
title: ai-audio-books
emoji: 📕👨‍💻🎧
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
python_version: 3.11
---

## Description

Automatically generate audiobooks from the text input. Automatically detect characters and map them to appropriate voices. Use text-to-speech models combined with text-to-audio-effect models to create an immersive listening experience

This project focuses on the automatic generation of audiobooks from text input, offering an immersive experience. The system intelligently detects characters in the text and assigns them distinct, appropriate voices using Large Language Model. To enhance the auditory experience, the project incorporates text-to-audio-effect models, adding relevant background sounds and audio effects that match the context of the narrative. The combination of natural-sounding speech synthesis and environmental sound design creates a rich, engaging audiobook experience that adapts seamlessly to different genres and styles of writing, making the storytelling more vivid and captivating for listeners.

### Action Items / Ideas

- intonations
    - add context
- audio effects
    - add context
    - filter, apply only for long phrases
- improve UI
    - show character parts
- testing
    - eval current execution time
- optimizations
    - combine sequential phrases of same character in single phrase
    - support large texts. use batching. problem: how to ensure same characters?
    - can detect characters in first prompt, then split text in each batch into character phrases
        - probably split large phrases into smaller ones
        - identify unknown characters
        - use LLM to recognize characters for a given text and provide descriptions detailed enough to select appropriate voice

