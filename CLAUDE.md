# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an audio transcription and speaker diarization CLI tool that combines OpenAI Whisper Large for transcription with pyannote.audio for speaker diarization. The project is optimized for Mac M1 with Metal Performance Shaders (MPS) acceleration.

## Common Commands

### Using uv (Recommended)
```bash
# Setup environment and dependencies
python run.py --setup

# Create configuration file
python run.py --create-config

# Basic transcription and diarization
python run.py audio.wav

# Generate SRT subtitles
python run.py audio.wav --format srt -o output.srt

# CSV output with known speaker count
python run.py audio.wav --format csv --speakers 2

# Use smaller model for faster processing
python run.py audio.wav --model medium
```

### Direct Usage (pip)
```bash
# Install dependencies
pip install -r requirements.txt

# Basic transcription and diarization
python transcribe_diarize.py audio.wav

# Create config file
python transcribe_diarize.py --create-config

# Use custom config
python transcribe_diarize.py audio.wav --config custom-config.toml
```

## Required Environment Setup

```bash
# Install uv (recommended package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ffmpeg (required for audio processing)
brew install ffmpeg

# For configuration-based setup:
# 1. Create config file: python run.py --create-config
# 2. Edit config.toml and add your HF token

# For environment variable setup:
export HF_TOKEN="your_hf_token_here"

# Accept terms for pyannote models on HuggingFace:
# - https://huggingface.co/pyannote/segmentation
# - https://huggingface.co/pyannote/speaker-diarization-3.1
```

## Architecture

### Core Components
- `TranscriptionDiarizer` - Main processing class that orchestrates Whisper and pyannote
- `OutputFormatter` - Handles JSON, SRT, and CSV output formatting
- `Config` - Configuration management (config.py) for settings and HF token
- `run.py` - UV runner script for dependency management
- `main()` - CLI argument parsing and execution flow

### Key Features
- **Mac M1 Optimization**: Automatically detects and uses MPS when available
- **Smart Result Combination**: Overlaps transcription segments with speaker segments using temporal matching
- **Multiple Output Formats**: JSON for programmatic use, SRT for subtitles, CSV for analysis
- **Flexible Model Selection**: Supports all Whisper model sizes (tiny to large)

### Processing Pipeline
1. Load Whisper model (with MPS acceleration if available)
2. Load pyannote diarization pipeline
3. Transcribe entire audio file → segments with timestamps
4. Perform speaker diarization → speaker segments with timestamps  
5. Combine results using overlap detection algorithm
6. Format and output results

### Performance Notes
- Whisper Large requires ~10-12GB RAM total
- Diarization runs on CPU (pyannote doesn't fully support MPS)
- Use smaller models (medium/small) for memory-constrained environments
- Processing time scales with audio length and model size