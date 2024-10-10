### Action items
- [ ] move speaker split to new pipeline
- [ ] env template
- [ ] move from AI/ML api to langchain
- [ ] bugfix w/ 11labs api
- [ ] async synthesis
- [ ] map characters to voices
- [] emotion enrichment: add intonation markers, auto-set TTS params
- [x] generate good enough sound effects for background
- [ ] mix effects with narrration
- [x] allow files uplaod (.txt)

### Backlog
- [ ] prepare text for TTS
    - [x] prepare prompt to split text into character phrases
    - [ ] split large text in batches, process each batch separatelly, concat batches
    - [ ] try to identify unknown characters
- [ ] select voices for TTS
    - [ ] map characters to available voices
    - [ ] use LLM to recognize characters for a given text and provide descriptions
detailed enough to select appropriate voice
- [ ] preprocess text phrases for TTS: add intonation markers, auto-set TTS params
- [ ] run TTS to create narration
- [ ] add effects. mix them with created narration

