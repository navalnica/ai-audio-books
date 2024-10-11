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

### Action items

- voices
    - filter to use only best voices
- intonations
    - add context 
- audio effects
    - add context
    - filter, apply only for long phrases
    - only for narrator?
    - checkbox! make effects great again (no) optional
- stability
    - add limit on input text size (5000 chars)
- improve UI
    - add error box
    - add samples
    - show character parts
    - remove file upload pane
    - labels on how long to wait
    - labels describing components
    - header and description
- prepare slides / story
- testing
    - eval current execution time
    - test on different text inputs
- optimizations
    - generate audio effects asynchronously
    - combine sequential phrases of same character in single phrase
    - support large texts. use batching. problem: how to ensure same characters?
    - can detect characters in first prompt, then split text in each batch into character phrases
        - probably split large phrases into smaller ones
        - identify unknown characters
        - use LLM to recognize characters for a given text and provide descriptions detailed enough to select appropriate voice

