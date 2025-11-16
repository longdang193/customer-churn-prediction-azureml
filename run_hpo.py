"""HPO pipeline for hyperparameter optimization."""

from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from azure.ai.ml import MLClient, dsl, Input
from azure.ai.ml.entities import Job
from azure.ai.ml.sweep import MedianStoppingPolicy
from azure.identity import DefaultAzureCredential

from hpo_utils import build_parameter_space, load_hpo_config
from src.utils import load_azure_config, get_data_asset_config
from azure.ai.ml import load_component

DEFAULT_COMPUTE = "cpu-cluster"


def create_hpo_pipeline(components: Dict[str, Any], hpo_cfg: Dict[str, Any]):
    """Create HPO pipeline with sweep configuration."""
    metric_name = hpo_cfg.get("metric", "f1")
    # Use generic metric name since each trial trains only one model
    # The train component logs both model-specific (e.g., rf_f1) and generic (f1) metrics
    primary_metric = metric_name  # e.g., "f1" instead of "best_model_f1"
    goal = "maximize" if hpo_cfg.get("mode", "max").lower() == "max" else "minimize"
    max_trials = int(hpo_cfg.get("budget", {}).get("max_trials", 10))
    max_concurrent = int(hpo_cfg.get("budget", {}).get("max_concurrent", min(4, max_trials)))

    parameter_space = build_parameter_space(hpo_cfg.get("search_space", {}))
    early_cfg = hpo_cfg.get("early_stopping", {})

    @dsl.pipeline(compute=DEFAULT_COMPUTE, description="HPO pipeline for churn prediction")
    def hpo_pipeline(pipeline_job_input_data):
        data_prep_job = components["data_prep"](raw_data=pipeline_job_input_data)

        base_train_job = components["train"](processed_data=data_prep_job.outputs.processed_data)
        sweep_job = base_train_job.sweep(
            primary_metric=primary_metric,
            goal=goal,
            sampling_algorithm=hpo_cfg.get("sampling_algorithm", "random"),
        )
        sweep_job.compute = DEFAULT_COMPUTE
        sweep_job.search_space = parameter_space
        sweep_job.set_limits(max_total_trials=max_trials, max_concurrent_trials=max_concurrent)

        if early_cfg.get("enabled", False):
            patience = int(early_cfg.get("patience", 2))
            sweep_job.early_termination = MedianStoppingPolicy(delay_evaluation=patience, evaluation_interval=1)

        return {
            "sweep_model_output": sweep_job.outputs.model_output,
            "sweep_parent_run_ids": sweep_job.outputs.parent_run_id,
        }

    return hpo_pipeline


def main() -> None:
    """Main entry point for HPO pipeline."""
    # Load environment variables from config.env
    load_dotenv("config.env")
    
    config = load_azure_config()
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace_name"],
    )
    components_dir = Path("aml/components")
    # Load components: data_prep from regular pipeline, train from HPO component
    components = {
        "data_prep": load_component(source=str(components_dir / "data_prep.yaml")),
        "train": load_component(source=str(components_dir / "hpo.yaml")),
    }
    hpo_cfg = load_hpo_config()
    pipeline = create_hpo_pipeline(components, hpo_cfg)

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
    returned_job: Job = ml_client.jobs.create_or_update(pipeline_job)
    
    print(f"✓ HPO sweep job submitted: {returned_job.name}")
    print(f"  View in Azure ML Studio: {returned_job.studio_url}")


if __name__ == "__main__":
    main()
