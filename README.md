# Audio Transcription and Speaker Diarization CLI

A command-line tool that combines OpenAI Whisper Large for transcription with pyannote.audio for speaker diarization, optimized for Mac M1 with Metal Performance Shaders (MPS) acceleration.

## Features

- **High-quality transcription** using OpenAI Whisper Large model
- **Speaker diarization** using pyannote.audio to identify "who speaks when"
- **Mac M1 optimization** with Metal Performance Shaders (MPS) support
- **Multiple output formats**: JSON, SRT subtitles, CSV
- **Flexible speaker detection** with automatic or manual speaker count
- **CLI interface** for easy automation and scripting

## Installation

### Prerequisites

1. **Python 3.9+** (recommended for best M1 compatibility; Python 3.11+ uses the built-in `tomllib` module while earlier versions rely on the `toml` package)
2. **uv** package manager (recommended):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **ffmpeg** for audio processing:
   ```bash
   brew install ffmpeg
   ```
4. **HuggingFace account** with accepted terms for pyannote models

### Setup with uv (Recommended)

1. Clone or download this repository
2. Run the setup:
   ```bash
   python run.py --setup
   ```
3. Create configuration file:
   ```bash
   python run.py --create-config
   ```
4. Edit the generated `config.toml` file and add your HuggingFace token

### Alternative Setup (pip)

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up HuggingFace token:
   - Go to [huggingface.co](https://huggingface.co) and create an account
   - Accept terms for [pyannote/segmentation](https://huggingface.co/pyannote/segmentation) and [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - Create a token in Settings → Tokens
   - Either set environment variable: `export HF_TOKEN="your_token_here"`
   - Or add it to `config.toml` file

## Usage

### Basic Usage with uv

```bash
# Transcribe and diarize audio file (using config.toml)
python run.py audio.wav

# Save results to JSON file
python run.py audio.wav -o results.json

# Generate SRT subtitles
python run.py conversation.mp3 --format srt -o subtitles.srt

# CSV output with known number of speakers
python run.py meeting.wav --format csv --speakers 3
```

### Direct Usage (without uv)

```bash
# Transcribe and diarize audio file
python transcribe_diarize.py audio.wav

# Save results to JSON file
python transcribe_diarize.py audio.wav -o results.json

# Generate SRT subtitles
python transcribe_diarize.py conversation.mp3 --format srt -o subtitles.srt

# CSV output with known number of speakers
python transcribe_diarize.py meeting.wav --format csv --speakers 3
```

### Command Line Options

```
positional arguments:
  audio_path            Path to the audio file to process

options:
  -h, --help            Show help message
  -o, --output OUTPUT   Output file path (prints to stdout if not specified)
  -f, --format {json,srt,csv}
                        Output format (default: json)
  --model {tiny,base,small,medium,large}
                        Whisper model size (default: large)
  --speakers SPEAKERS   Number of speakers (improves accuracy if known)
  --hf-token HF_TOKEN   HuggingFace token (or set HF_TOKEN env variable)
  --config CONFIG       Path to configuration file
  --create-config       Create a default configuration file
```

### Configuration File Examples

```bash
# Create default config file
python run.py --create-config

# Use custom config file
python run.py audio.wav --config /path/to/custom-config.toml

# Override config defaults with command line
python run.py audio.wav --model medium --speakers 2
```

### Processing Examples

```bash
# Process interview with 2 speakers, output as SRT (uv)
python run.py interview.mp3 --speakers 2 --format srt -o interview.srt

# Use smaller model for faster processing (direct)
python transcribe_diarize.py audio.wav --model medium --format json

# Process and view results directly
python run.py meeting.wav --format csv
```

## Output Formats

### JSON
```json
[
  {
    "speaker": "SPEAKER_0",
    "start": 0.0,
    "end": 5.3,
    "text": "Hello, thank you all for joining."
  },
  {
    "speaker": "SPEAKER_1", 
    "start": 5.3,
    "end": 7.2,
    "text": "Hi, it's great to be here."
  }
]
```

### SRT Subtitles
```
1
00:00:00,000 --> 00:00:05,300
[SPEAKER_0] Hello, thank you all for joining.

2
00:00:05,300 --> 00:00:07,200
[SPEAKER_1] Hi, it's great to be here.
```

### CSV
```csv
speaker,start,end,text
SPEAKER_0,0.0,5.3,"Hello, thank you all for joining."
SPEAKER_1,5.3,7.2,"Hi, it's great to be here."
```

## Performance Notes

### Mac M1 Optimization
- Whisper model automatically uses MPS (Metal Performance Shaders) when available
- Diarization currently runs on CPU (pyannote doesn't fully support MPS yet)
- Large model requires ~10-12GB RAM total

### Model Size Recommendations
- **Large**: Best accuracy, ~10GB RAM, slower processing
- **Medium**: Good balance, ~5GB RAM, faster processing  
- **Small/Base**: Fast processing, lower accuracy, <2GB RAM

### Supported Audio Formats
- WAV, MP3, FLAC, M4A, and other formats supported by ffmpeg
- Automatic conversion to mono 16kHz (required for diarization)

## Configuration File

The tool uses a `config.toml` file for default settings. Create one with:

```bash
python run.py --create-config
```

Example configuration:

```toml
[huggingface]
token = "hf_your_token_here"

[whisper]
default_model = "large"  # tiny, base, small, medium, large

[output]
default_format = "json"  # json, srt, csv

[diarization]
max_speakers = null  # or set a number to limit speakers

[performance]
use_mps = true  # Use Mac M1/M2/M3 GPU acceleration

[cache]
directory = "~/.cache/whisper-diarize"
```

## Troubleshooting

### Common Issues

1. **HuggingFace token errors**:
   - Ensure you've accepted terms for pyannote models
   - Verify token is correctly set in config.toml or environment variable
   - Run `python run.py --create-config` to create template

2. **Memory issues on 8GB Macs**:
   - Use smaller Whisper model: `--model medium` or `--model small`
   - Close other memory-intensive applications

3. **uv not found**:
   - Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Restart terminal or run `source ~/.bashrc`

4. **Slow processing**:
   - Try smaller model size
   - Ensure ffmpeg is installed for audio preprocessing

5. **Import errors**:
   - With uv: `python run.py --setup`
   - With pip: `pip install -r requirements.txt`
   - Check Python version compatibility (3.9+ recommended)

### Getting Help

For issues with:
- **Whisper**: Check [OpenAI Whisper repository](https://github.com/openai/whisper)
- **pyannote.audio**: Check [pyannote documentation](https://github.com/pyannote/pyannote-audio)
- **This tool**: Create an issue with error details and system info

## Technical Details

The tool performs these steps:
1. Loads Whisper Large model (with MPS acceleration if available)
2. Loads pyannote diarization pipeline
3. Transcribes entire audio file to get text with timestamps
4. Performs speaker diarization to identify speaker segments
5. Combines results by matching timestamps
6. Outputs in requested format

The combination algorithm uses overlap detection to assign transcribed text to speaker segments, ensuring accurate speaker attribution even when segment boundaries don't perfectly align.