#!/usr/bin/env python3
"""Create Azure ML Data Asset from data in Azure Blob Storage."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from load_config import get_config


def create_data_asset(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    storage_account: str,
    container: str,
    data_asset_name: str,
    data_version: str,
    data_path: str = "churn.csv",
    description: str = "Bank Customer Churn Dataset"
):
    """Create an Azure ML Data Asset from blob storage."""
    try:
        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import Data
        from azure.ai.ml.constants import AssetTypes
        from azure.identity import DefaultAzureCredential
    except ImportError:
        print("ERROR: Azure ML SDK not installed.")
        print("Install with: pip install azure-ai-ml azure-identity")
        sys.exit(1)
    
    print("=" * 60)
    print("Creating Azure ML Data Asset")
    print("=" * 60)
    
    print(f"\n[1/4] Authenticating to Azure...")
    print(f"  Subscription: {subscription_id}")
    print(f"  Resource Group: {resource_group}")
    print(f"  Workspace: {workspace_name}")
    
    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        print("  ✓ Authentication successful")
    except Exception as e:
        print(f"  ✗ Authentication failed: {e}")
        print("\nTry running: az login")
        sys.exit(1)
    
    print(f"\n[2/4] Preparing data asset configuration...")
    
    try:
        datastores = ml_client.datastores.list()
        datastore_names = [ds.name for ds in datastores]
        print(f"  Available datastores: {', '.join(datastore_names)}")
        
        default_datastore = None
        for ds in datastores:
            if hasattr(ds, 'account_name') and ds.account_name == storage_account:
                default_datastore = ds.name
                print(f"  Found matching datastore: {default_datastore}")
                break
        
        if not default_datastore:
            default_datastore = "workspaceblobstore"
            print(f"  Using default datastore: {default_datastore}")
        
        data_uri = f"azureml://datastores/{default_datastore}/paths/{data_path}"
        
    except Exception as e:
        print(f"  Warning: Could not list datastores: {e}")
        data_uri = f"https://{storage_account}.blob.core.windows.net/{container}/{data_path}"
    
    print(f"  Data URI: {data_uri}")
    print(f"  Asset Name: {data_asset_name}")
    print(f"  Version: {data_version}")
    
    print(f"\n[3/4] Creating data asset...")
    
    data_asset = Data(
        name=data_asset_name,
        version=data_version,
        description=description,
        type=AssetTypes.URI_FILE,
        path=data_uri
    )
    
    try:
        ml_client.data.create_or_update(data_asset)
        print(f"  ✓ Data asset created successfully!")
    except Exception as e:
        print(f"  ✗ Failed to create data asset: {e}")
        sys.exit(1)
    
    print(f"\n[4/4] Verifying data asset...")
    try:
        retrieved_asset = ml_client.data.get(name=data_asset_name, version=data_version)
        print(f"  ✓ Asset verified")
        print(f"\n  Name: {retrieved_asset.name}")
        print(f"  Version: {retrieved_asset.version}")
        print(f"  Path: {retrieved_asset.path}")
        print(f"  Type: {retrieved_asset.type}")
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
    
    print("\n" + "=" * 60)
    print("SUCCESS! Data asset created.")
    print("=" * 60)
    print(f"\nReference in pipelines: azureml:{data_asset_name}:{data_version}")
    print(f"View in Azure ML Studio: https://ml.azure.com")
    print("=" * 60)


def main():
    """Load config and create data asset."""
    print("Loading configuration from config.env...")
    config = get_config()
    
    required_keys = {
        'subscription_id': 'AZURE_SUBSCRIPTION_ID',
        'resource_group': 'AZURE_RESOURCE_GROUP',
        'workspace_name': 'AZURE_WORKSPACE_NAME',
        'storage_account': 'AZURE_STORAGE_ACCOUNT',
        'storage_container': 'AZURE_STORAGE_CONTAINER',
        'data_asset_full': 'DATA_ASSET_FULL',
        'data_version': 'DATA_VERSION'
    }
    
    missing_vars = [env_var for key, env_var in required_keys.items() if not config.get(key)]
    
    if missing_vars:
        print(f"\nERROR: Missing required configuration variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print(f"\nPlease update your config.env file.")
        sys.exit(1)
    
    print("✓ Configuration loaded\n")
    
    create_data_asset(
        subscription_id=config['subscription_id'],
        resource_group=config['resource_group'],
        workspace_name=config['workspace_name'],
        storage_account=config['storage_account'],
        container=config['storage_container'],
        data_asset_name=config['data_asset_full'],
        data_version=config['data_version'],
        data_path="churn.csv",
        description="Bank Customer Churn Dataset - Full dataset (10k rows)"
    )


if __name__ == "__main__":
    main()

