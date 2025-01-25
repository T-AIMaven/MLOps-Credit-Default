"""
This script evaluates and compares a new credit default prediction model against the currently deployed model.
Key functionality:
- Loads test data and performs feature engineering
- Generates predictions using both new and existing models
- Calculates and compares performance metric (AUC)
- Registers the new model if it performs better
- Sets task values for downstream pipeline steps

The evaluation process:
1. Loads models from the serving endpoint
2. Prepares test data with feature engineering
3. Generates predictions from both models
4. Calculates AUC metric
5. Makes registration decision based on AUC comparison
6. Updates pipeline task values with results
"""

import argparse

import mlflow
from databricks import feature_engineering
from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# from databricks.connect import DatabricksSession
from pyspark.sql import SparkSession

from credit_default.utils import load_config, setup_logging

# Set up logging
setup_logging(log_file="")

try:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", action="store", default=None, type=str, required=True)
    parser.add_argument("--git_sha", action="store", default=None, type=str, required=True)
    parser.add_argument("--job_run_id", action="store", default=None, type=str, required=True)
    parser.add_argument("--new_model_uri", action="store", default=None, type=str, required=True)

    args = parser.parse_args()
    root_path = args.root_path
    git_sha = args.git_sha
    job_run_id = args.job_run_id
    new_model_uri = args.new_model_uri
    logger.debug(f"Git SHA: {git_sha}")
    logger.debug(f"Job Run ID: {job_run_id}")
    logger.debug(f"New Model Uri: {new_model_uri}")
    logger.info("Parsed arguments successfully.")

    # Load configuration
    logger.info("Loading configuration...")
    config_path = f"{root_path}/project_config.yml"
    config = load_config(config_path)
    logger.info("Configuration loaded successfully.")

    # Initialize Databricks workspace client
    workspace = WorkspaceClient()
    logger.info("Databricks workspace client initialized.")

    # Initialize Spark session
    spark = SparkSession.builder.getOrCreate()
    # spark = DatabricksSession.builder.getOrCreate()
    fe = feature_engineering.FeatureEngineeringClient()
    logger.info("Spark session initialized.")

    # Extract configuration details
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    target = config.target[0].new_name
    columns = config.features.clean
    logger.debug(f"Catalog: {catalog_name}, Schema: {schema_name}")

    # Define the serving endpoint
    logger.info("Fetching model serving endpoint...")
    serving_endpoint_name = "credit-default-model-serving-feature"
    serving_endpoint = workspace.serving_endpoints.get(serving_endpoint_name)
    model_name = serving_endpoint.config.served_models[0].model_name
    model_version = serving_endpoint.config.served_models[0].model_version
    previous_model_uri = f"models:/{model_name}/{model_version}"

    # Load test set and create additional features in Spark DataFrame
    logger.info("Loading test dataset...")
    test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

    X_test_spark = test_set.select(columns)
    y_test_spark = test_set.select("Id", target)

    # Generate predictions from both models
    logger.info("Generating predictions for both models...")
    predictions_previous = fe.score_batch(model_uri=previous_model_uri, df=X_test_spark)
    predictions_new = fe.score_batch(model_uri=new_model_uri, df=X_test_spark)

    predictions_new = predictions_new.withColumnRenamed("prediction", "prediction_new")
    predictions_old = predictions_previous.withColumnRenamed("prediction", "prediction_old")
    test_set = test_set.select("Id", target)

    # Join the DataFrames on the 'id' column
    df = test_set.join(predictions_new, on="Id").join(predictions_old, on="Id")
    logger.info("DataFrames joined successfully.")

    # Validate DataFrame structure
    required_columns = ["prediction_new", "prediction_old", "Default"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in DataFrame.")

    # Calculate AUC for the new model
    evaluator_new = BinaryClassificationEvaluator(
        rawPredictionCol="prediction_new", labelCol="Default", metricName="areaUnderROC"
    )
    auc_new = evaluator_new.evaluate(df)

    # Calculate AUC for the old model
    evaluator_old = BinaryClassificationEvaluator(
        rawPredictionCol="prediction_old", labelCol="Default", metricName="areaUnderROC"
    )
    auc_old = evaluator_old.evaluate(df)

    logger.info(f"AUC for new model: {auc_new}")
    logger.info(f"AUC for old model: {auc_old}")

    # Model comparison and registration
    if auc_old < auc_new:
        logger.info("New model is better based on AUC. Registering new model...")
        model_version = mlflow.register_model(
            model_uri=new_model_uri,
            name=f"{catalog_name}.{schema_name}.credit_model_feature",
            tags={"git_sha": git_sha, "job_run_id": job_run_id},
        )
        logger.info(f"New model registered with version: {model_version.version}")
        dbutils.jobs.taskValues.set(key="model_version", value=model_version.version)  # noqa: F821
        dbutils.jobs.taskValues.set(key="model_update", value=1)  # noqa: F821
    else:
        logger.info("Old model is better based on AUC.")
        dbutils.jobs.taskValues.set(key="model_update", value=0)  # noqa: F821

except Exception as e:
    logger.error(f"An error occurred: {e}")
    dbutils.jobs.taskValues.set(key="model_update", value=0)  # noqa: F821
    raise
