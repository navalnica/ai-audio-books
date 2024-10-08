### TODO

- [ ] prepare text for TTS
    - [x] prepare prompt to split text into character phrases
    - [ ] split large text in batches, process each batch separatelly, concat batches
    - [ ] try to identify unknown characters
- [ ] select voices for TTS
    - [ ] map characters to available voices
    - [ ] use LLM to recognize characters for a given text and provide descriptions
detailed enough to select appropriate voice
- [ ] run TTS to create narration
- [ ] add effects. mix them with created narration