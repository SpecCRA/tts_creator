import os
from pathlib import Path
from src.data.process_videos import convert_video

RAW_DATA_DIR = Path("data/raw")
CACHE_DATA_DIR = Path("data/cache")

RAW_FILES = [file for file in os.listdir(RAW_DATA_DIR) if file.endswith(".mkv")]
OUTPUT_FILES_PRESENT = [file for file in os.listdir(CACHE_DATA_DIR) if file.endswith(".mp3")]

for file in RAW_FILES:
    # if target file is not in output directory
    if f"{file.split('.')[0]}.mp3" not in OUTPUT_FILES_PRESENT:
        convert_video(
            video_filepath=str((RAW_DATA_DIR) / file),
            output_dir=str(CACHE_DATA_DIR),
            output_filename=file.split(".")[0],
            bitrate="32k", # Reduce bitrate to reduce file size
            video_present=False,
        )
