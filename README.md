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

