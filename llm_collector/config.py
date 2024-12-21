from pathlib import Path
from typing import Any
import os
import json
from dotenv import load_dotenv

class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Load non-sensitive configuration from config.json
        config_path = Path("config.json")
        if config_path.exists():
            self.config = json.loads(config_path.read_text())
        else:
            raise FileNotFoundError("config.json file not found. Please create it with the necessary configurations.")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value from config.json or environment variables."""
        # Check environment variables first
        env_value = os.getenv(key)
        if env_value is not None:
            return env_value
        
        # Fallback to config.json
        parts = key.split(".")
        value = self.config
        for part in parts:
            value = value.get(part, default)
            if value == default:
                break
        return value