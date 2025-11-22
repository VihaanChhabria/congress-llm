import whisper
import os

# Set the environment variable for ffmpeg path
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin"

model = whisper.load_model("tiny.en")
result = model.transcribe("audio.m4a", verbose=True)

# Each segment has start, end, and text
for segment in result['segments']:
    print(segment['start'], segment['end'], segment['text'])