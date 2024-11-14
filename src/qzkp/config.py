"""Configuration settings for QZKP."""

from typing import Dict, Any
import yaml
import os

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    default_config = {
        "dimensions": 8,
        "security_level": 128,
        "batch_size": 1000,
        "max_cache_size": 10000,
        "thread_count": min(32, (os.cpu_count() or 1) * 4),
    }

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            default_config.update(user_config)

    return default_config
