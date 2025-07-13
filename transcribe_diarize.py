#!/usr/bin/env python3
"""
Audio Transcription and Speaker Diarization CLI Tool
Uses OpenAI Whisper Large for transcription and pyannote.audio for speaker diarization.
Optimized for Mac M1 with Metal Performance Shaders (MPS) acceleration.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torchaudio
import whisper
from pyannote.audio import Pipeline

from config import create_config_if_missing, load_config

# Suppress noisy third-party warnings to keep CLI output readable
warnings.filterwarnings(
    "ignore",
    message="torchaudio._backend.*deprecated",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Module 'speechbrain.pretrained' was deprecated",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="`torchaudio.backend.common.AudioMetaData` has been moved",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The MPEG_LAYER_III subtype is unknown to TorchAudio",
    category=UserWarning,
)


class TranscriptionDiarizer:
    def __init__(
        self,
        whisper_model: Optional[str] = None,
        hf_token: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        # Load configuration
        self.config = load_config(config_path)

        # Set model and token with config defaults
        self.whisper_model_name = whisper_model or self.config.get_whisper_model()
        self.hf_token = hf_token or self.config.get_hf_token()
        self.use_mps = self.config.should_use_mps()

        self.whisper_model = None
        self.diarization_pipeline = None

        # Check for Mac M1 acceleration
        self.device = self._get_device()
        print(f"Using device: {self.device}")

    def _get_device(self) -> str:
        """Determine the best device for computation."""
        if self.use_mps and torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def preprocess_audio(self, audio_path: str) -> str:
        """Preprocess audio with FFmpeg for better transcription quality."""
        print("Preprocessing audio with FFmpeg...")

        # Create temporary file for processed audio
        temp_dir = tempfile.mkdtemp()
        processed_path = os.path.join(temp_dir, "processed_audio.wav")

        # FFmpeg command for audio normalization and enhancement
        ffmpeg_command = [
            "ffmpeg",
            "-i",
            audio_path,
            "-ac",
            "1",  # Convert to mono
            "-ar",
            "16000",  # Resample to 16kHz (optimal for Whisper)
            "-f",
            "wav",
            "-filter_complex",
            "volume=3dB, dynaudnorm=p=0.9:m=20:s=20",  # Volume boost and dynamic normalization
            "-hide_banner",
            "-loglevel",
            "quiet",
            "-y",  # Overwrite output files
            processed_path,
        ]

        print(f"FFmpeg command: {' '.join(ffmpeg_command)}")

        try:
            result = subprocess.run(
                ffmpeg_command, check=True, capture_output=True, text=True
            )
            print("Audio preprocessing completed successfully")
            return processed_path
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg preprocessing failed: {e}")
            print(f"stderr: {e.stderr}")
            # Return original path if preprocessing fails
            return audio_path
        except FileNotFoundError:
            print("FFmpeg not found - using original audio file")
            return audio_path

    def load_models(self):
        """Load Whisper and pyannote models."""
        print("Loading Whisper model...")

        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(models_dir, exist_ok=True)

        # Set Whisper cache directory
        whisper_cache_dir = os.path.join(models_dir, "whisper")
        os.makedirs(whisper_cache_dir, exist_ok=True)

        # Load Whisper model with custom download directory
        self.whisper_model = whisper.load_model(
            self.whisper_model_name, download_root=whisper_cache_dir
        )

        # Try to move Whisper to MPS if available
        if self.device == "mps":
            try:
                self.whisper_model = self.whisper_model.to("mps")
                print("Whisper model moved to MPS (Apple GPU)")
            except Exception:
                print("Could not move Whisper to MPS. Falling back to CPU.")
                self.device = "cpu"

        print("Loading diarization pipeline...")
        if not self.hf_token:
            raise ValueError(
                "HuggingFace token required. Set HF_TOKEN environment variable or pass --hf-token"
            )

        # Set HuggingFace cache directory for pyannote models
        pyannote_cache_dir = os.path.join(models_dir, "pyannote")
        os.makedirs(pyannote_cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = pyannote_cache_dir

        try:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token,
                cache_dir=pyannote_cache_dir,
            )
            print("Diarization pipeline loaded successfully")
        except Exception as e:
            print(f"Error loading diarization pipeline: {e}")
            print(
                "Make sure you have accepted the terms for pyannote models on HuggingFace"
            )
            raise

    def diarize_audio(self, audio_path: str, num_speakers: Optional[int] = None) -> Any:
        """Perform speaker diarization using pyannote."""
        print("Performing speaker diarization...")

        # Check diarization pipeline device
        if (
            self.diarization_pipeline is not None
            and hasattr(self.diarization_pipeline, "_segmentation")
            and hasattr(self.diarization_pipeline._segmentation, "model")
        ):
            seg_device = next(
                self.diarization_pipeline._segmentation.model.parameters()
            ).device
            print(f"Diarization segmentation model device: {seg_device}")

        # Use config max_speakers if not provided
        if num_speakers is None:
            num_speakers = self.config.get_max_speakers()

        # Measure diarization time
        start_time = time.time()

        # Set up diarization parameters
        if self.diarization_pipeline is None:
            raise RuntimeError("Diarization pipeline not loaded")

        if num_speakers:
            diarization = self.diarization_pipeline(
                audio_path, num_speakers=num_speakers
            )
        else:
            diarization = self.diarization_pipeline(audio_path)

        diarization_time = time.time() - start_time

        # Count unique speakers
        speakers = set()
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)

        # Get audio duration for speed calculation
        waveform, sample_rate = torchaudio.load(audio_path)
        audio_duration = waveform.shape[1] / sample_rate
        diarization_ratio = audio_duration / diarization_time

        print(
            f"Diarization completed. Found {len(speakers)} speakers: {sorted(speakers)}"
        )
        print(
            f"Diarization time: {diarization_time:.1f}s, Audio: {audio_duration:.1f}s, Speed: {diarization_ratio:.2f}x real-time"
        )

        return diarization

    def merge_speaker_segments(
        self, diarization: Any, audio_duration: float
    ) -> pd.DataFrame:
        """Convert diarization to DataFrame and merge adjacent segments from same speaker."""
        print("Processing diarization segments...")

        # Convert diarization to DataFrame
        segments_data = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments_data.append(
                {"start": segment.start, "end": segment.end, "speaker": speaker}
            )

        df = pd.DataFrame(segments_data)
        if df.empty:
            return df

        # Sort by start time
        df = df.sort_values("start").reset_index(drop=True)

        # Ensure first segment starts at 0
        if not df.empty and df.iloc[0]["start"] > 0:
            df.at[df.index[0], "start"] = 0.0

        # Merge adjacent segments from the same speaker
        merged_segments = []
        current_segment = df.iloc[0].copy()

        for i in range(1, len(df)):
            next_segment = df.iloc[i]
            gap = next_segment["start"] - current_segment["end"]

            # If same speaker, always merge regardless of gap
            if next_segment["speaker"] == current_segment["speaker"]:
                print(
                    f"  Merging {current_segment['speaker']}: {current_segment['end']:.1f}s → {next_segment['start']:.1f}s (gap: {gap:.1f}s)"
                )
                # Extend current segment
                current_segment["end"] = next_segment["end"]
            else:
                # Save current segment and start new one
                merged_segments.append(current_segment.to_dict())
                current_segment = next_segment.copy()

        # Add last segment
        merged_segments.append(current_segment.to_dict())

        merged_df = pd.DataFrame(merged_segments)

        # Ensure last segment ends at audio duration
        if not merged_df.empty and merged_df.iloc[-1]["end"] < audio_duration:
            merged_df.at[merged_df.index[-1], "end"] = audio_duration

        print(f"Merged {len(df)} segments into {len(merged_df)} speaker blocks")

        return merged_df

    def extract_audio_segment(
        self, audio_path: str, start_time: float, end_time: float, temp_dir: str
    ) -> str:
        """Extract audio segment and save to temporary file."""
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Calculate sample indices
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # Extract segment
        segment_waveform = waveform[:, start_sample:end_sample]

        # Save to temporary file
        temp_file = os.path.join(
            temp_dir, f"segment_{start_time:.2f}_{end_time:.2f}.wav"
        )
        torchaudio.save(temp_file, segment_waveform, sample_rate)

        return temp_file

    def transcribe_segments(
        self, audio_path: str, speaker_segments: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Transcribe each speaker segment separately."""
        print(f"Transcribing {len(speaker_segments)} speaker segments...")

        results = []
        total_transcription_time = 0
        total_audio_duration = 0

        # Create temporary directory for audio segments
        with tempfile.TemporaryDirectory() as temp_dir:
            for idx, segment in speaker_segments.iterrows():
                start_time = segment["start"]
                end_time = segment["end"]
                speaker = segment["speaker"]
                segment_duration = end_time - start_time

                segment_num = idx + 1 if isinstance(idx, int) else 1
                print(
                    f"Transcribing segment {segment_num}/{len(speaker_segments)}: {speaker} ({start_time:.1f}s-{end_time:.1f}s, {segment_duration:.1f}s)"
                )

                # Extract audio segment
                temp_audio_file = self.extract_audio_segment(
                    audio_path, start_time, end_time, temp_dir
                )

                # Transcribe segment with timing
                try:
                    transcription_start = time.time()
                    if self.whisper_model is not None:
                        segment_transcription = self.whisper_model.transcribe(
                            temp_audio_file
                        )
                    else:
                        raise RuntimeError("Whisper model not loaded")
                    transcription_time = time.time() - transcription_start

                    total_transcription_time += transcription_time
                    total_audio_duration += segment_duration

                    segment_ratio = segment_duration / transcription_time

                    text = str(segment_transcription["text"]).strip()

                    if text:  # Only add if transcription produced text
                        results.append(
                            {
                                "speaker": speaker,
                                "start": round(start_time, 2),
                                "end": round(end_time, 2),
                                "text": text,
                            }
                        )
                        print(
                            f"  → Text: {text[:100]}{'...' if len(text) > 100 else ''}"
                        )
                        print(
                            f"  → Speed: {segment_ratio:.2f}x real-time ({transcription_time:.1f}s for {segment_duration:.1f}s audio)"
                        )
                    else:
                        print(f"  → No text detected (speed: {segment_ratio:.2f}x)")

                except Exception as e:
                    print(f"  → Transcription failed: {e}")
                    continue

                # Clean up temporary file
                if os.path.exists(temp_audio_file):
                    os.remove(temp_audio_file)

        # Overall transcription stats
        if total_transcription_time > 0:
            overall_ratio = total_audio_duration / total_transcription_time
            print(
                f"Overall transcription speed: {overall_ratio:.2f}x real-time ({total_transcription_time:.1f}s for {total_audio_duration:.1f}s audio)"
            )

        print(f"Successfully transcribed {len(results)} segments")
        return results

    def process_audio(
        self, audio_path: str, num_speakers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Main processing pipeline - new approach: diarization first, then per-segment transcription."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load models if not already loaded
        if self.whisper_model is None or self.diarization_pipeline is None:
            self.load_models()

        print("=== Diarization First, Then Per-Segment Transcription ===")

        # Preprocess audio with FFmpeg
        processed_audio_path = self.preprocess_audio(audio_path)

        # Get audio duration for segment boundary fixing
        waveform, sample_rate = torchaudio.load(processed_audio_path)
        audio_duration = waveform.shape[1] / sample_rate

        # Step 1: Perform diarization to identify speaker segments (use processed audio)
        diarization = self.diarize_audio(processed_audio_path, num_speakers)

        # Step 2: Merge adjacent segments from the same speaker
        speaker_segments = self.merge_speaker_segments(diarization, audio_duration)

        if speaker_segments.empty:
            print("Warning: No speaker segments found")
            return []

        # Step 3: Transcribe each merged speaker segment separately (use processed audio)
        results = self.transcribe_segments(processed_audio_path, speaker_segments)

        print(f"=== Processing Complete: {len(results)} final segments ===")
        return results


class OutputFormatter:
    @staticmethod
    def format_time_srt(seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        msec = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{msec:03d}"

    @staticmethod
    def to_json(results: List[Dict[str, Any]], indent: int = 2) -> str:
        """Format results as JSON."""
        return json.dumps(results, ensure_ascii=False, indent=indent)

    @staticmethod
    def to_srt(results: List[Dict[str, Any]]) -> str:
        """Format results as SRT subtitles."""
        srt_content = ""
        for idx, segment in enumerate(results, start=1):
            start_time = OutputFormatter.format_time_srt(segment["start"])
            end_time = OutputFormatter.format_time_srt(segment["end"])
            speaker = segment["speaker"]
            text = segment["text"]

            srt_content += f"{idx}\n"
            srt_content += f"{start_time} --> {end_time}\n"
            srt_content += f"[{speaker}] {text}\n\n"

        return srt_content

    @staticmethod
    def to_csv(results: List[Dict[str, Any]]) -> str:
        """Format results as CSV."""
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(["speaker", "start", "end", "text"])

        # Write data
        for segment in results:
            writer.writerow(
                [segment["speaker"], segment["start"], segment["end"], segment["text"]]
            )

        return output.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description="Audio Transcription and Speaker Diarization CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.wav -o result.json
  %(prog)s conversation.mp3 --format srt --speakers 2
  %(prog)s meeting.wav --format csv --hf-token YOUR_TOKEN
        """,
    )

    parser.add_argument(
        "audio_path", nargs="?", help="Path to the audio file to process"
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (if not specified, prints to stdout)",
        default=None,
    )

    parser.add_argument(
        "-f",
        "--format",
        choices=["json", "srt", "csv"],
        default="json",
        help="Output format (default: json)",
    )

    parser.add_argument(
        "--model",
        choices=[
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v2",
            "large-v3",
            "large-v3-turbo",
        ],
        default="large",
        help="Whisper model size (default: large)",
    )

    parser.add_argument(
        "--speakers",
        type=int,
        help="Number of speakers (if known, improves diarization accuracy)",
    )

    parser.add_argument(
        "--hf-token", help="HuggingFace token (or set HF_TOKEN environment variable)"
    )

    parser.add_argument(
        "--config",
        help="Path to configuration file (default: config.toml in current directory)",
    )

    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create a default configuration file and exit",
    )

    args = parser.parse_args()

    # Handle config creation
    if args.create_config:
        config = create_config_if_missing(args.config)
        print(
            "Configuration file created. Please edit it to add your HuggingFace token."
        )
        sys.exit(0)

    # Check if audio_path is provided when not creating config
    if not args.audio_path:
        parser.error("audio_path is required unless --create-config is used")

    try:
        # Load config and get defaults
        config = load_config(args.config)

        # Use command line args or config defaults
        model = args.model if args.model != "large" else config.get_whisper_model()
        format_type = (
            args.format if args.format != "json" else config.get_output_format()
        )

        # Initialize transcriber
        transcriber = TranscriptionDiarizer(
            whisper_model=model, hf_token=args.hf_token, config_path=args.config
        )

        # Process audio
        results = transcriber.process_audio(args.audio_path, args.speakers)

        # Format output
        if format_type == "json":
            output_content = OutputFormatter.to_json(results)
        elif format_type == "srt":
            output_content = OutputFormatter.to_srt(results)
        elif format_type == "csv":
            output_content = OutputFormatter.to_csv(results)

        # Write or print output
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_content)
            print(f"Results saved to: {args.output}")
        else:
            print(output_content)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
