"""
Utility to load configuration from config.env file
"""
import os
from pathlib import Path


def load_env_file(env_file="config.env"):
    """Load environment variables from a .env file"""
    env_path = Path(__file__).parent / env_file

    if not env_path.exists():
        print(f"Warning: {env_file} not found at {env_path}")
        return

    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                # Parse KEY="VALUE" or KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value


def get_config():
    """Get Azure ML configuration as a dictionary"""
    load_env_file()

    config = {
        # Azure basics
        'subscription_id': os.getenv('AZURE_SUBSCRIPTION_ID'),
        'tenant_id': os.getenv('AZURE_TENANT_ID'),
        'resource_group': os.getenv('AZURE_RESOURCE_GROUP'),
        'location': os.getenv('AZURE_LOCATION'),

        # Workspace
        'workspace_name': os.getenv('AZURE_WORKSPACE_NAME'),

        # Compute
        'compute_instance': os.getenv('AZURE_COMPUTE_INSTANCE_NAME'),
        'compute_cluster': os.getenv('AZURE_COMPUTE_CLUSTER_NAME'),

        # Storage
        'storage_account': os.getenv('AZURE_STORAGE_ACCOUNT'),
        'storage_container': os.getenv('AZURE_STORAGE_CONTAINER'),

        # Data
        'data_asset_full': os.getenv('DATA_ASSET_FULL'),
        'data_asset_sample': os.getenv('DATA_ASSET_SAMPLE'),
        'data_version': os.getenv('DATA_VERSION'),

        # Model
        'model_name': os.getenv('MODEL_NAME'),
        'experiment_name': os.getenv('EXPERIMENT_NAME'),

        # Deployment
        'endpoint_name': os.getenv('ENDPOINT_NAME'),
        'deployment_name': os.getenv('DEPLOYMENT_NAME'),
    }

    return config


if __name__ == "__main__":
    # Test loading
    config = get_config()
    print("Configuration loaded:")
    for key, value in config.items():
        if value:
            # Mask sensitive values
            if 'id' in key.lower() or 'key' in key.lower():
                display_value = value[:8] + "..." if len(value) > 8 else "***"
            else:
                display_value = value
            print(f"  {key}: {display_value}")
        else:
            print(f"  {key}: NOT SET")
