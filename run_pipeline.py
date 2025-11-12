# run_pipeline.py

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, dsl, load_component


def load_azure_config() -> Dict[str, Any]:
    """Load Azure ML configuration from environment variables."""
    load_dotenv()

    config = {
        "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
        "resource_group": os.getenv("AZURE_RESOURCE_GROUP"),
        "workspace_name": os.getenv("AZURE_WORKSPACE_NAME"),
    }

    if not all(config.values()):
        raise ValueError("Azure ML configuration is missing in .env file.")

    print("Configuration loaded.")
    return config


def load_pipeline_components(components_dir: Path) -> Dict[str, Any]:
    """Load all AML components from the specified directory."""
    print("\nLoading components...")
    components = {
        "data_prep": load_component(source=str(components_dir / "data_prep.yaml")),
        "train": load_component(source=str(components_dir / "train.yaml")),
        "evaluate": load_component(source=str(components_dir / "evaluate.yaml")),
    }
    print("✓ Components loaded.")
    return components


def define_pipeline(components: Dict[str, Any]):
    """Define the Azure ML pipeline using the loaded components."""

    @dsl.pipeline(
        compute="cpu-cluster",
        description="E2E pipeline for bank churn prediction",
    )
    def churn_prediction_pipeline(
        pipeline_job_input_data,
    ):
        """The end-to-end training pipeline."""
        data_prep_job = components["data_prep"](raw_data=pipeline_job_input_data)
        train_job = components["train"](processed_data=data_prep_job.outputs.processed_data)
        evaluate_job = components["evaluate"](
            test_data=data_prep_job.outputs.processed_data,
            parent_run_id=train_job.outputs.parent_run_id,
        )
        return {
            "pipeline_job_evaluation_results": evaluate_job.outputs.evaluation_output,
            "pipeline_job_model_artifacts": train_job.outputs.model_output,
            "pipeline_job_parent_run_id": train_job.outputs.parent_run_id,
        }

    print("\nPipeline function defined.")
    return churn_prediction_pipeline


def main():
    """Main function to define and run the Azure ML pipeline."""
    config = load_azure_config()

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace_name"],
    )
    print(f"✓ Connected to workspace: {ml_client.workspace_name}")

    components = load_pipeline_components(Path("aml/components"))
    pipeline = define_pipeline(components)

    print("\nCreating and submitting pipeline job...")
    pipeline_input = ml_client.data.get("bank-churn-raw", version="1")
    pipeline_job = pipeline(pipeline_job_input_data=pipeline_input)

    returned_job = ml_client.jobs.create_or_update(pipeline_job)
    print(f"✓ Job submitted. Run name: {returned_job.name}")
    print(f"  View in Azure ML Studio: {returned_job.studio_url}")


if __name__ == "__main__":
    main()
