#!/usr/bin/env python3
"""
Setup script for the Audio Transcription and Speaker Diarization CLI Tool
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="whisper-diarization-cli",
    version="1.0.0",
    author="Audio Processing CLI",
    description="CLI tool for audio transcription and speaker diarization using Whisper and pyannote",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/whisper-diarization-cli",
    py_modules=["transcribe_diarize"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "transcribe-diarize=transcribe_diarize:main",
        ],
    },
)