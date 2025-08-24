# Audio Transcription with Utterance-Level Timestamps

A Python script that generates transcriptions with precise utterance-level timestamps using Hugging Face speech recognition models.

## Features

- **Utterance-level timestamps**: Get precise start and end times for each transcribed segment
- **Multiple output formats**: TXT, JSON, and SRT subtitle formats
- **Configurable chunking**: Adjustable audio chunk length for optimal transcription accuracy
- **Hugging Face integration**: Uses state-of-the-art speech recognition models
- **GPU acceleration**: Automatically detects and uses CUDA if available
- **Fallback support**: Gracefully handles different model types

## Installation

1. Install the required dependencies:
```bash
pip install -e .
```

Or install manually:
```bash
pip install transformers torch torchaudio librosa numpy soundfile
```

## Usage

### Command Line Interface

The main script can be used from the command line:

```bash
# Basic usage with default settings
python transcribe.py audio_file.wav

# Specify output file and format
python transcribe.py audio_file.wav --output my_transcript.txt --format txt

# Use a different model
python transcribe.py audio_file.wav --model "facebook/wav2vec2-base-960h"

# Adjust chunk length for better timestamp accuracy
python transcribe.py audio_file.wav --chunk-length 15.0

# Generate SRT subtitle format
python transcribe.py audio_file.wav --format srt --output subtitles.srt
```

### Command Line Options

- `audio_path`: Path to the audio file (required)
- `--model`: Hugging Face model name (default: Khalsuu/filipino-wav2vec2-l-xls-r-300m-official)
- `--output, -o`: Output file path (default: input_name_transcription.txt)
- `--format`: Output format: txt, json, or srt (default: txt)
- `--chunk-length`: Audio chunk length in seconds (default: 30.0)

### Python API

You can also use the `AudioTranscriber` class in your own Python code:

```python
from transcribe import AudioTranscriber

# Initialize transcriber
transcriber = AudioTranscriber("your-model-name")

# Transcribe with timestamps
transcriptions = transcriber.transcribe_with_timestamps("audio.wav", chunk_length_s=20.0)

# Save in different formats
transcriber.save_transcription(transcriptions, "output.txt", "txt")
transcriber.save_transcription(transcriptions, "output.json", "json")
transcriber.save_transcription(transcriptions, "output.srt", "srt")
```

### Example Script

Run the included example script:

```bash
python example.py
```

Make sure to update the `audio_file` path in `example.py` to point to your actual audio file.

## Output Formats

### TXT Format
Human-readable format with timestamps:
```
Transcription of audio file
Model: Khalsuu/filipino-wav2vec2-l-xls-r-300m-official
Total segments: 5
==================================================

[00:00.000 - 00:20.000] First segment of transcribed text
[00:20.000 - 00:40.000] Second segment of transcribed text
...
```

### JSON Format
Structured data for programmatic use:
```json
[
  {
    "start_time": 0.0,
    "end_time": 20.0,
    "duration": 20.0,
    "text": "First segment of transcribed text",
    "chunk_id": 0
  }
]
```

### SRT Format
Standard subtitle format for video players:
```
1
00:00.000 --> 00:20.000
First segment of transcribed text

2
00:20.000 --> 00:40.000
Second segment of transcribed text
```

## Supported Audio Formats

The script supports various audio formats including:
- WAV
- MP3
- FLAC
- OGG
- And other formats supported by librosa

## Model Selection

You can use any Hugging Face speech recognition model. Some popular options:

- **Wav2Vec2 models**: `facebook/wav2vec2-base-960h`, `facebook/wav2vec2-large-960h-lv60-self`
- **Whisper models**: `openai/whisper-base`, `openai/whisper-small`, `openai/whisper-medium`
- **Custom fine-tuned models**: Like the Filipino model used in the default configuration

## Performance Tips

1. **GPU acceleration**: The script automatically uses CUDA if available for faster processing
2. **Chunk length**: Smaller chunks (15-20 seconds) provide more precise timestamps but may be slower
3. **Model size**: Larger models generally provide better accuracy but require more memory and processing time

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce chunk length or use a smaller model
2. **Audio loading errors**: Ensure the audio file is not corrupted and is in a supported format
3. **Model loading errors**: Check internet connection for downloading models, or use a different model

### Dependencies

Make sure you have the required packages installed:
```bash
pip install transformers torch torchaudio librosa numpy soundfile
```

## License

This project is open source. Feel free to modify and distribute according to your needs.
