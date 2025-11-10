#!/usr/bin/env python3
"""Verify Azure ML Data Asset exists and show its details."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from load_config import get_config


def main():
    """Verify the data asset exists."""
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
    except ImportError:
        print("ERROR: Azure ML SDK not installed.")
        print("Install with: pip install azure-ai-ml azure-identity")
        sys.exit(1)
    
    config = get_config()
    
    print("Connecting to Azure ML...")
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=config['subscription_id'],
        resource_group_name=config['resource_group'],
        workspace_name=config['workspace_name']
    )
    
    print(f"\nRetrieving data asset: {config['data_asset_full']} v{config['data_version']}")
    
    try:
        data_asset = ml_client.data.get(
            name=config['data_asset_full'],
            version=config['data_version']
        )
        
        print("\n" + "="*60)
        print("✓ DATA ASSET VERIFIED")
        print("="*60)
        print(f"Name: {data_asset.name}")
        print(f"Version: {data_asset.version}")
        print(f"Type: {data_asset.type}")
        print(f"Path: {data_asset.path}")
        print(f"Description: {data_asset.description}")
        print("\nReference in pipelines:")
        print(f"  azureml:{data_asset.name}:{data_asset.version}")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Data asset not found: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

