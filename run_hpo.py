# run_hpo.py

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from azure.ai.ml import MLClient, dsl
from azure.ai.ml.entities import Job
from azure.ai.ml.sweep import Choice, MedianStoppingPolicy

from run_pipeline import load_azure_config, load_pipeline_components

CONFIG_PATH = Path("configs/train.yaml")
DEFAULT_COMPUTE = "cpu-cluster"
DEFAULT_DATA_ASSET = "bank-churn-raw"


def load_hpo_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"HPO config not found at {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    return config.get("hpo", {})


def build_parameter_space(search_space: Dict[str, Any]) -> Dict[str, Choice]:
    rf_space = search_space.get("rf", {})
    parameter_space: Dict[str, Choice] = {}

    def to_choice(values):
        cleaned = [None if v is None else v for v in values]
        return Choice(*cleaned)

    if "n_estimators" in rf_space:
        parameter_space["rf_n_estimators"] = to_choice(rf_space["n_estimators"])
    if "max_depth" in rf_space:
        parameter_space["rf_max_depth"] = to_choice(rf_space["max_depth"])
    if "min_samples_split" in rf_space:
        parameter_space["rf_min_samples_split"] = to_choice(rf_space["min_samples_split"])
    if "min_samples_leaf" in rf_space:
        parameter_space["rf_min_samples_leaf"] = to_choice(rf_space["min_samples_leaf"])

    if not parameter_space:
        raise ValueError("HPO search space for random forest is empty. Check configs/train.yaml")
    return parameter_space


def create_hpo_pipeline(components: Dict[str, Any], hpo_cfg: Dict[str, Any]):
    metric_name = hpo_cfg.get("metric", "f1")
    primary_metric = f"best_model_{metric_name}"
    goal = "maximize" if hpo_cfg.get("mode", "max").lower() == "max" else "minimize"
    max_trials = int(hpo_cfg.get("budget", {}).get("max_trials", 10))
    max_concurrent = int(hpo_cfg.get("budget", {}).get("max_concurrent", min(4, max_trials)))

    parameter_space = build_parameter_space(hpo_cfg.get("search_space", {}))
    early_cfg = hpo_cfg.get("early_stopping", {})

    @dsl.pipeline(compute=DEFAULT_COMPUTE, description="HPO pipeline for churn prediction")
    def hpo_pipeline(pipeline_job_input_data):
        data_prep_job = components["data_prep"](raw_data=pipeline_job_input_data)

        sweep_job = components["train"].sweep(
            primary_metric=primary_metric,
            goal=goal,
            sampling_algorithm=hpo_cfg.get("sampling_algorithm", "random"),
        )
        sweep_job.compute = DEFAULT_COMPUTE
        sweep_job.inputs.processed_data = data_prep_job.outputs.processed_data
        sweep_job.parameter_space = parameter_space
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
    config = load_azure_config()
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace_name"],
    )
    print(f"✓ Connected to workspace: {ml_client.workspace_name}")

    components = load_pipeline_components(Path("aml/components"))
    hpo_cfg = load_hpo_config()
    pipeline = create_hpo_pipeline(components, hpo_cfg)

    data_asset_name = os.getenv("AZURE_RAW_DATA_ASSET", DEFAULT_DATA_ASSET)
    pipeline_input = ml_client.data.get(name=data_asset_name, version="1")

    print("\nSubmitting HPO pipeline...")
    pipeline_job = pipeline(pipeline_job_input_data=pipeline_input)
    returned_job: Job = ml_client.jobs.create_or_update(pipeline_job)
    print(f"✓ Sweep job submitted. Run name: {returned_job.name}")
    print(f"  View in Azure ML Studio: {returned_job.studio_url}")


if __name__ == "__main__":
    main()
