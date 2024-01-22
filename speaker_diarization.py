# Import
import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from tqdm import tqdm

load_dotenv()

# If data file does not exist, create it
OUT_FILEPATH = Path("data/temp/diarization_data.json")
IN_DIR = Path("data/cache")

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
diarization_pipe = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.environ.get("HUGGINGFACE_API_KEY"),
)
diarization_pipe.to(torch.device(device))
diarization_pipe.embeddings_batch_size = 2 # Reduce batch size to avoid CUDA out of memory error, limited to 8 gb

for filename in tqdm(os.listdir(IN_DIR)):
    print(f"Processing {filename}...")
    with ProgressHook() as hook:
        diarization = diarization_pipe((IN_DIR / filename), max_speakers=2, hook=hook)

    data = {filename: []}

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        data[filename].append(
            {"start": turn.start, "end": turn.end, "speaker": speaker}
        )

    if OUT_FILEPATH.exists():
        with open(OUT_FILEPATH, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    existing_data.update(data)

    with open(OUT_FILEPATH, "w") as f:
        json.dump(existing_data, f, indent=4)
