# src/utils/config_loader.py
import yaml
from pathlib import Path

def load_config():
    """Loads the application configuration from the root config.yaml file."""
    config_path = Path(__file__).parent.parent.parent / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
        