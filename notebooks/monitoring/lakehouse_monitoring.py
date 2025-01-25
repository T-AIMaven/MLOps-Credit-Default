# Databricks notebook source
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from pyspark.sql import SparkSession

from credit_default.utils import load_config

spark = SparkSession.builder.getOrCreate()

# Load configuration
config = load_config("../../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

workspace = WorkspaceClient()

monitoring_table = f"{catalog_name}.{schema_name}.model_monitoring"

workspace.quality_monitors.create(
    table_name=monitoring_table,
    assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{monitoring_table}",
    output_schema_name=f"{catalog_name}.{schema_name}",
    inference_log=MonitorInferenceLog(
        problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
        prediction_col="prediction",
        timestamp_col="timestamp",
        granularities=["30 minutes"],
        model_id_col="model_name",
        label_col="default",
    ),
)

spark.sql(f"ALTER TABLE {monitoring_table} " "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

# COMMAND ----------

## How to delete a monitor
# workspace.quality_monitors.delete(
#     table_name=f"{catalog_name}.{schema_name}.model_monitoring"
# )
