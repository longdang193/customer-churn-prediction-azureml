"""Azure ML configuration loading utilities."""

from typing import Dict, Optional

from .env_loader import get_env_var, load_env_file


def load_azure_config(config_path: Optional[str] = None) -> Dict[str, str]:
    """Load Azure ML configuration from environment variables.
    
    Args:
        config_path: Path to config.env file (default: "config.env" in project root)
        
    Returns:
        Dictionary containing subscription_id, resource_group, and workspace_name
        
    Raises:
        ValueError: If required configuration is missing
    """
    load_env_file(config_path)
    
    config = {
        "subscription_id": get_env_var("AZURE_SUBSCRIPTION_ID", required=True),
        "resource_group": get_env_var("AZURE_RESOURCE_GROUP", required=True),
        "workspace_name": get_env_var("AZURE_WORKSPACE_NAME", required=True),
    }
    
    return config


def get_data_asset_config(config_path: Optional[str] = None) -> Dict[str, str]:
    """Get data asset configuration from environment variables.
    
    Args:
        config_path: Path to config.env file (default: "config.env" in project root)
        
    Returns:
        Dictionary containing data_asset_name and data_asset_version
        
    Raises:
        ValueError: If DATA_ASSET_FULL is not set
    """
    load_env_file(config_path)
    
    data_asset_name = get_env_var("DATA_ASSET_FULL", default="churn-data")
    data_asset_version = get_env_var("DATA_VERSION", default="1")
    
    return {
        "data_asset_name": data_asset_name,
        "data_asset_version": data_asset_version,
    }

