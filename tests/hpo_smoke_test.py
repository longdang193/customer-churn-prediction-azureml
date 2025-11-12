#!/usr/bin/env python3
"""Minimal HyperDrive sweep smoke test to validate HPO workflow."""

import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Job
from azure.identity import DefaultAzureCredential

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from run_hpo import (
    build_parameter_space,
    create_hpo_pipeline,
    load_hpo_config,
)
from run_pipeline import load_azure_config, load_pipeline_components


def skip_if_no_azure_config():
    """Skip test if Azure ML configuration is not available."""
    try:
        load_azure_config()
    except (ValueError, FileNotFoundError):
        pytest.skip("Azure ML configuration not available (missing .env or credentials)")


@pytest.mark.skipif(
    not os.getenv("AZURE_SUBSCRIPTION_ID"),
    reason="Azure ML credentials not configured",
)
def test_hpo_sweep_submission():
    """Submit a minimal HyperDrive sweep (2 trials) and validate it starts successfully."""
    skip_if_no_azure_config()

    # Load Azure config and connect
    config = load_azure_config()
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace_name"],
    )

    # Load components
    components = load_pipeline_components(Path("aml/components"))
    hpo_cfg = load_hpo_config()

    # Override config for minimal smoke test: only 2 trials, 1 concurrent
    hpo_cfg["budget"]["max_trials"] = 2
    hpo_cfg["budget"]["max_concurrent"] = 1

    # Create pipeline
    pipeline = create_hpo_pipeline(components, hpo_cfg)

    # Get data asset (or skip if not available)
    data_asset_name = os.getenv("AZURE_RAW_DATA_ASSET", "bank-churn-raw")
    try:
        pipeline_input = ml_client.data.get(name=data_asset_name, version="1")
    except Exception as e:
        pytest.skip(f"Data asset '{data_asset_name}' not available: {e}")

    # Submit the sweep
    print("\n--- Submitting minimal HPO sweep (2 trials) ---")
    pipeline_job = pipeline(pipeline_job_input_data=pipeline_input)
    returned_job: Job = ml_client.jobs.create_or_update(pipeline_job)

    # Validate submission
    assert returned_job is not None, "Job submission failed"
    assert returned_job.name is not None, "Job name is missing"
    print(f"✓ Sweep job submitted: {returned_job.name}")
    print(f"  Status: {returned_job.status}")
    print(f"  Studio URL: {returned_job.studio_url}")

    # Basic validation: job should be in a valid state
    assert returned_job.status in [
        "NotStarted",
        "Queued",
        "Starting",
        "Preparing",
        "Provisioning",
        "Running",
    ], f"Unexpected job status: {returned_job.status}"

    # Note: We don't wait for completion in the smoke test to keep it fast
    # Full validation would require:
    # 1. Waiting for job completion (ml_client.jobs.stream(returned_job.name))
    # 2. Checking returned_job.status == "Completed"
    # 3. Validating best_trial exists: sweep_job.best_trial
    # 4. Checking MLflow metrics were logged

    print("\n✓ HPO smoke test passed: sweep job submitted successfully")
    print(f"  Note: Check Azure ML Studio to monitor completion: {returned_job.studio_url}")


@pytest.mark.skipif(
    not os.getenv("AZURE_SUBSCRIPTION_ID"),
    reason="Azure ML credentials not configured",
)
def test_hpo_parameter_space_building():
    """Validate that parameter space can be built from config."""
    hpo_cfg = load_hpo_config()
    search_space = hpo_cfg.get("search_space", {})

    # Build parameter space
    param_space = build_parameter_space(search_space)

    # Validate it's not empty
    assert len(param_space) > 0, "Parameter space is empty - check configs/train.yaml::hpo.search_space"

    # Validate it contains expected keys for RF
    rf_space = search_space.get("rf", {})
    if rf_space:
        expected_keys = ["rf_n_estimators", "rf_max_depth", "rf_min_samples_split", "rf_min_samples_leaf"]
        found_keys = [k for k in expected_keys if k in param_space]
        assert len(found_keys) > 0, f"No RF hyperparameters found in parameter space. Found: {list(param_space.keys())}"

    print(f"✓ Parameter space built successfully: {list(param_space.keys())}")


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v", "-s"])

