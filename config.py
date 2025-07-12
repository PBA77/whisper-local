"""
Configuration management for the transcription and diarization tool.
"""

import os
import toml
from pathlib import Path
from typing import Optional, Dict, Any


class Config:
    """Configuration manager that handles settings from file and environment variables."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = self._get_config_path(config_path)
        self.config_data = self._load_config()
    
    def _get_config_path(self, config_path: Optional[str] = None) -> Path:
        """Determine the configuration file path."""
        if config_path:
            return Path(config_path)
        
        # Try multiple locations in order of preference
        possible_paths = [
            Path.cwd() / "config.toml",  # Current directory
            Path.home() / ".config" / "whisper-diarize" / "config.toml",  # User config dir
            Path(__file__).parent / "config.toml",  # Same directory as this script
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Return default path (current directory) even if it doesn't exist
        return possible_paths[0]
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from TOML file."""
        if not self.config_path.exists():
            return {}
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return toml.load(f)
        except (toml.TomlDecodeError, IOError) as e:
            print(f"Warning: Could not load config file {self.config_path}: {e}")
            return {}
    
    def get_hf_token(self) -> Optional[str]:
        """Get HuggingFace token from config file or environment variable."""
        # Priority: config file -> environment variable
        token = self.config_data.get("huggingface", {}).get("token")
        if token:
            return token
        
        return os.getenv("HF_TOKEN")
    
    def get_whisper_model(self) -> str:
        """Get default Whisper model from config."""
        return self.config_data.get("whisper", {}).get("default_model", "large")
    
    def get_output_format(self) -> str:
        """Get default output format from config."""
        return self.config_data.get("output", {}).get("default_format", "json")
    
    def get_cache_dir(self) -> Path:
        """Get cache directory for models."""
        cache_dir = self.config_data.get("cache", {}).get("directory")
        if cache_dir:
            return Path(cache_dir).expanduser()
        
        # Default cache location
        return Path.home() / ".cache" / "whisper-diarize"
    
    def should_use_mps(self) -> bool:
        """Check if MPS acceleration should be used."""
        return self.config_data.get("performance", {}).get("use_mps", True)
    
    def get_max_speakers(self) -> Optional[int]:
        """Get maximum number of speakers for diarization."""
        max_speakers = self.config_data.get("diarization", {}).get("max_speakers")
        if max_speakers == "null" or max_speakers == "None":
            return None
        return max_speakers
    
    def create_default_config(self) -> None:
        """Create a default configuration file."""
        default_config = {
            "huggingface": {
                "token": "your_hf_token_here",
                "# Note": "Get your token from https://huggingface.co/settings/tokens"
            },
            "whisper": {
                "default_model": "large",
                "# Available models": "tiny, base, small, medium, large"
            },
            "output": {
                "default_format": "json",
                "# Available formats": "json, srt, csv"
            },
            "diarization": {
                "max_speakers": None,
                "# Note": "Set to number if you want to limit speaker detection"
            },
            "performance": {
                "use_mps": True,
                "# Note": "Use Metal Performance Shaders on Mac M1/M2/M3"
            },
            "cache": {
                "directory": "~/.cache/whisper-diarize",
                "# Note": "Where to store downloaded models"
            }
        }
        
        # Create directory if it doesn't exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            toml.dump(default_config, f)
        
        print(f"Created default configuration file at: {self.config_path}")
        print("Please edit the file to add your HuggingFace token.")


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration with optional path override."""
    return Config(config_path)


def create_config_if_missing(config_path: Optional[str] = None) -> Config:
    """Load config or create default if missing."""
    config = Config(config_path)
    
    if not config.config_path.exists():
        config.create_default_config()
    
    return config