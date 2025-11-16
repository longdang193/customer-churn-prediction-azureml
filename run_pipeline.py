"""Training pipeline with fixed hyperparameters from config."""

from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, dsl, load_component, Input

from src.utils import load_azure_config, get_data_asset_config


def load_pipeline_components(components_dir: Path) -> Dict[str, Any]:
    """Load AML components from the specified directory."""
    components = {
        "data_prep": load_component(source=str(components_dir / "data_prep.yaml")),
        "train": load_component(source=str(components_dir / "train.yaml")),
    }
    return components


def define_pipeline(components: Dict[str, Any]):
    """Define the Azure ML pipeline using the loaded components."""

    @dsl.pipeline(
        compute="cpu-cluster",
        description="Training pipeline for bank churn prediction",
    )
    def churn_prediction_pipeline(pipeline_job_input_data):
        """Training pipeline with fixed hyperparameters."""
        data_prep_job = components["data_prep"](raw_data=pipeline_job_input_data)
        train_job = components["train"](processed_data=data_prep_job.outputs.processed_data)
        return {
            "model_output": train_job.outputs.model_output,
            "parent_run_id": train_job.outputs.parent_run_id,
        }

    return churn_prediction_pipeline


def main():
    """Main function to define and run the Azure ML pipeline."""
    # Load environment variables from config.env
    load_dotenv("config.env")
    
    config = load_azure_config()

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace_name"],
    )
    components = load_pipeline_components(Path("aml/components"))
    pipeline = define_pipeline(components)

    # Get data asset configuration
    data_asset_config = get_data_asset_config()
    data_asset_name = data_asset_config["data_asset_name"]
    data_asset_version = data_asset_config["data_asset_version"]
    pipeline_input = Input(
        type="uri_folder",
        path=f"azureml:{data_asset_name}:{data_asset_version}",
        mode="mount",
    )
    pipeline_job = pipeline(pipeline_job_input_data=pipeline_input)
    returned_job = ml_client.jobs.create_or_update(pipeline_job)
    
    print(f"✓ Job submitted: {returned_job.name}")
    print(f"  View in Azure ML Studio: {returned_job.studio_url}")


if __name__ == "__main__":
    main()
