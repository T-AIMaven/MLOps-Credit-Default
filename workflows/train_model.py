"""
This script trains a LightGBM model for credit default prediction with feature engineering.
Key functionality:
- Loads training and test data from Databricks tables
- Performs feature engineering using Databricks Feature Store
- Creates a pipeline with preprocessing and LightGBM classifier
- Tracks the experiment using MLflow
- Logs model metrics, parameters and artifacts
- Handles feature lookups
- Outputs model URI for downstream tasks

"""

import argparse
import sys

import mlflow
from databricks import feature_engineering

# from databricks.connect import DatabricksSession
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score  # classification_report, confusion_matrix,
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from credit_default.utils import load_config, setup_logging

# Set up logging
setup_logging(log_file="")

try:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", action="store", default=None, type=str, required=True)
    parser.add_argument("--git_sha", action="store", default=None, type=str, required=True)
    parser.add_argument("--job_run_id", action="store", default=None, type=str, required=True)
    args = parser.parse_args()

    root_path = args.root_path
    git_sha = args.git_sha
    job_run_id = args.job_run_id
    logger.debug(f"Git SHA: {git_sha}")
    logger.debug(f"Job Run ID: {job_run_id}")
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
    parameters = config.parameters
    features_robust = config.features.robust
    columns = config.features.clean
    columns_wo_id = columns.copy()
    columns_wo_id.remove("Id")

    # Convert train and test sets to Pandas DataFrames
    train_pdf = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
    test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

    # Now use create_training_set to create balanced training set
    # Drop the original features that will be looked up from the feature store
    # Define the list of columns you want to drop, including "Update_timestamp_utc"

    feature_table_name = f"{catalog_name}.{schema_name}.features_balanced"
    columns_to_drop = columns_wo_id + ["Update_timestamp_utc"]
    train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop(*columns_to_drop)
    logger.info(f"Train set columns for feature table: {train_set.columns}")

    # Create training set from feature table
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    training_set = fe.create_training_set(
        df=train_set,
        label=target,
        feature_lookups=[
            FeatureLookup(
                table_name=feature_table_name,
                feature_names=columns_wo_id,
                lookup_key="Id",
            )
        ],
        exclude_columns=["Update_timestamp_utc"],
    )
    training_df = training_set.load_df().toPandas()
    logger.info("Training set created and loaded.")

    # Prepare train and test datasets
    X_train = training_df[columns_wo_id]
    y_train = training_df[target]
    X_test = test_set[columns_wo_id]
    y_test = test_set[target]

    # Define preprocessing and pipeline
    preprocessor = ColumnTransformer(
        transformers=[("robust_scaler", RobustScaler(), features_robust)],
        remainder="passthrough",
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**parameters))])

    # MLflow setup
    mlflow.set_experiment(experiment_name="/Shared/credit-feature")
    logger.info("MLflow setup completed.")

    # Train model and log in MLflow
    with mlflow.start_run(tags={"branch": "bundles", "git_sha": git_sha, "job_run_id": job_run_id}) as run:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        auc_test = roc_auc_score(y_test, y_pred)
        logger.info(f"Test AUC: {auc_test}")

        # Log model details
        mlflow.log_param("model_type", "LightGBM with preprocessing")
        mlflow.log_params(parameters)
        mlflow.log_metric("AUC", auc_test)
        input_example = X_train.iloc[:5]
        signature = infer_signature(model_input=input_example, model_output=pipeline.predict(input_example))  # y_pred

        # Log model with feature engineering
        # We will register in next step, if model is better than the previous one (evaluate_model.py)
        fe.log_model(
            model=pipeline,
            flavor=mlflow.sklearn,
            artifact_path="lightgbm-pipeline-model-fe",
            training_set=training_set,
            signature=signature,
            input_example=input_example,
        )
        model_uri = f"runs:/{run.info.run_id}/lightgbm-pipeline-model-fe"
        dbutils.jobs.taskValues.set(key="new_model_uri", value=model_uri)  # noqa: F821
        logger.info(f"Model registered: {model_uri}")

except Exception as e:
    logger.error(f"An error occurred: {e}")
    sys.exit(1)  # Exit with failure code
