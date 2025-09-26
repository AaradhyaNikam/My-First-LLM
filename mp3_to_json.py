import whisper
import os
import json
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = whisper.load_model("large-v2", device=device)
print("Model loaded successfully.")

audios = os.listdir('Audios')
for audio in audios:
    tutorial_number = audio.split(' - ')[0]
    file_name = audio.split(' - ')[1].split('.mp3')[0]
    
    result = model.transcribe(audio=f"Audios/{audio}",
                            language='hi',
                            task='translate',
                            word_timestamps=False)
    
    chunks = []
    for segment in result['segments']:
        chunks.append({"number" : tutorial_number,
                    "file_name" : file_name,
                    "start" : segment['start'],
                    "end" : segment['end'],
                    "text" : segment['text']})
        
    chunks_with_context = {"chunks" : chunks, "text": result["text"]}
    with open(f"jsons/{audio}.json", "w", encoding='utf-8') as f:

        json.dump(chunks_with_context, f)