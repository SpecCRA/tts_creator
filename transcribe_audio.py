# Imports
import os
import json
from pathlib import Path
from dotenv import load_dotenv

import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


# Folders
IN_DIR = Path("data/cache")
OUT_FILEPATH = Path("data/temp/transcriptions.json")

load_dotenv()
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu" # Disable when not streaming
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"

# Load model
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=8,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device
)

# Create data file if it does not exist
if OUT_FILEPATH.exists():
    with open(OUT_FILEPATH, "r") as f:
        existing_data = json.load(f)

else:
    existing_data = {}

for filename in tqdm(os.listdir(IN_DIR)):
    print(f"Processing {filename}...")
    transcriptions = pipe(
        str(IN_DIR / filename)
    )

    data = {filename: []}

    for chunk in transcriptions["chunks"]:
        data[filename].append(
            {"start": chunk["timestamp"][0], "end": chunk["timestamp"][1], "text": chunk["text"]}
        )
    existing_data.update(data)
    
    with open(OUT_FILEPATH, "w") as f:
        json.dump(existing_data, f)
