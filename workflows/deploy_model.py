"""
This script handles the deployment of a credit default prediction model to a Databricks serving endpoint.
Key functionality:
- Loads project configuration from YAML
- Retrieves the model version from previous task values
- Updates the serving endpoint configuration with:
  - Model registry reference
  - Scale to zero capability
  - Workload sizing
  - Specific model version
The endpoint is configured for feature-engineered model serving with automatic scaling.
"""

import argparse
import sys

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput
from loguru import logger

from credit_default.utils import load_config, setup_logging

# Set up logging
setup_logging(log_file="")

try:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path",
        action="store",
        default=None,
        type=str,
        required=True,
    )

    args = parser.parse_args()
    root_path = args.root_path
    logger.info("Parsed arguments successfully.")

    # Load configuration
    logger.info("Loading configuration...")
    config_path = f"{root_path}/project_config.yml"
    config = load_config(config_path)
    logger.info("Configuration loaded successfully.")

    # Retrieve model version from the previous task
    model_version = dbutils.jobs.taskValues.get(taskKey="evaluate_model", key="model_version")  # noqa: F821

    if not model_version:
        raise ValueError("Model version could not be retrieved.")
    logger.info(f"Retrieved model version: {model_version}")

    # Initialize Databricks workspace client
    workspace = WorkspaceClient()
    logger.info("Databricks workspace client initialized.")

    # Extract catalog and schema names
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    logger.debug(f"Catalog: {catalog_name}, Schema: {schema_name}")

    # Update serving endpoint configuration
    endpoint_name = "credit-default-model-serving-feature"
    logger.info(f"Updating serving endpoint: {endpoint_name}")
    workspace.serving_endpoints.update_config_and_wait(
        name=endpoint_name,
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.credit_model_feature",
                scale_to_zero_enabled=True,
                workload_size="Medium",
                entity_version=model_version,
            )
        ],
    )
    logger.success(f"Serving endpoint {endpoint_name} updated successfully with model version {model_version}.")

except Exception as e:
    logger.error(f"An error occurred: {e}")
    sys.exit(1)  # Exit with a failure code
