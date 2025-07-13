#!/usr/bin/env python3
"""
UV Runner script for Whisper Diarization CLI Tool
Handles dependency management and execution using uv.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_uv_installed():
    """Check if uv is installed and available."""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Using uv: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("Error: uv is not installed or not in PATH")
    print("Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
    return False


def setup_environment():
    """Set up the virtual environment and install dependencies."""
    project_dir = Path(__file__).parent
    
    print("Setting up environment with uv...")
    
    # Sync dependencies using uv
    result = subprocess.run(
        ["uv", "sync"],
        cwd=project_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error setting up environment: {result.stderr}")
        return False
    
    print("Environment setup completed successfully")
    return True


def run_transcription(args):
    """Run the transcription tool with uv."""
    project_dir = Path(__file__).parent
    
    # Prepare the command
    cmd = ["uv", "run", "python", "transcribe_diarize.py"] + args
    
    print(f"Running: {' '.join(cmd)}")
    
    # Run the command
    result = subprocess.run(cmd, cwd=project_dir)
    return result.returncode


def main():
    """Main runner function."""
    if not check_uv_installed():
        sys.exit(1)
    
    # Check if we need to set up the environment
    project_dir = Path(__file__).parent
    venv_dir = project_dir / ".venv"
    
    if not venv_dir.exists() or "--setup" in sys.argv:
        if not setup_environment():
            sys.exit(1)
        
        # Remove --setup from args if present
        if "--setup" in sys.argv:
            sys.argv.remove("--setup")
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        print("Whisper Diarization CLI Tool - UV Runner")
        print("\nUsage:")
        print("  python run.py [OPTIONS] AUDIO_FILE")
        print("  python run.py --setup  # Force environment setup")
        print("\nExamples:")
        print("  python run.py audio.wav")
        print("  python run.py audio.wav --format srt --speakers 2")
        print("  python run.py audio.wav --model medium --output result.json")
        print("\nFor full options, run:")
        print("  python run.py --help")
        sys.exit(0)
    
    # Pass remaining arguments to the transcription tool
    exit_code = run_transcription(sys.argv[1:])
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
