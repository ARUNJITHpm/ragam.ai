# Malayalam TTS Dataset Builder

Production-grade dataset preparation pipeline for Malayalam TTS fine-tuning from YouTube video URLs.

Target CSV format:

```csv
utt_id,video_id,source_url,audio_path,start_sec,end_sec,duration_sec,transcript
```

Pipeline:

```text
video URLs
→ download full WAV
→ trim 60s to 150s
→ validate audio
→ Whisper transcription
→ speech-aware utterance segmentation
→ dataset.csv / metadata.csv
→ train/val/test split
```
