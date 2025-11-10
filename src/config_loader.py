"""Configuration loader for YAML config files."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary with configuration values
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., "training.models")
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def merge_configs(cli_args: Dict[str, Any], config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Merge CLI arguments with configuration file.
    CLI arguments take precedence over config file values.
    
    Args:
        cli_args: Dictionary of CLI arguments
        config_file: Path to configuration file (optional)
        
    Returns:
        Merged configuration dictionary
    """
    if config_file and Path(config_file).exists():
        file_config = load_config(config_file)
    else:
        file_config = {}
    
    # Merge: CLI args override config file
    merged = {**file_config, **cli_args}
    
    # Remove None values (not provided in CLI)
    merged = {k: v for k, v in merged.items() if v is not None}
    
    return merged


def validate_config(config: Dict[str, Any], required_keys: list) -> None:
    """
    Validate that required configuration keys are present.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required key paths (dot notation)
        
    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = []
    
    for key_path in required_keys:
        value = get_config_value(config, key_path)
        if value is None:
            missing_keys.append(key_path)
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")

