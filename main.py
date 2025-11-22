import os
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin"

# Community-1 open-source speaker diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token="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")

pipeline.to(torch.device("cpu"))

print("Processing audio.m4a")

# apply pretrained pipeline (with optional progress hook)``
with ProgressHook() as hook:
    output = pipeline("audio.m4a", hook=hook)  # runs locally

# print the result
for turn, speaker in output.speaker_diarization:
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")