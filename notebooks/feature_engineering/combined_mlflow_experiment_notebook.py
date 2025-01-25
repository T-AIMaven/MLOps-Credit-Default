# Databricks notebook source
import json

import mlflow
import pandas as pd

# from pyspark.sql import SparkSession
from databricks.connect import DatabricksSession
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from credit_default.utils import load_config

load_dotenv()


class CreditDefaultModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            predictions = self.model.predict(model_input)
            return {"Prediction": predictions}
        else:
            raise ValueError("Input must be a pandas DataFrame.")


def setup_environment():
    # spark = SparkSession.builder.getOrCreate()
    spark = DatabricksSession.builder.getOrCreate()

    config_path = "../../project_config.yml"

    # Set up MLflow
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    return spark, config_path


def load_data(spark, catalog_name, schema_name):
    train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
    test_set_spark = spark.table(f"{catalog_name}.{schema_name}.test_set")

    train_set = train_set_spark.toPandas()
    test_set = test_set_spark.toPandas()

    X_train = train_set.drop(columns=["Default", "Update_timestamp_utc"])
    y_train = train_set["Default"]
    X_test = test_set.drop(columns=["Default"])
    y_test = test_set["Default"]

    return X_train, y_train, X_test, y_test, train_set_spark


def create_pipeline(config, parameters):
    features_robust = config.features.robust

    preprocessor = ColumnTransformer(
        transformers=[("robust_scaler", RobustScaler(), features_robust)], remainder="passthrough"
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**parameters))])


def main():
    # Setup
    spark, CONFIG_DATABRICKS = setup_environment()
    config = load_config(CONFIG_DATABRICKS)
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    parameters = config.parameters
    client = MlflowClient()

    # Load data
    X_train, y_train, X_test, y_test, train_set_spark = load_data(spark, catalog_name, schema_name)

    # Training and logging
    mlflow.set_experiment(experiment_name="/Shared/credit_default")

    with mlflow.start_run(tags={"branch": "serving"}) as run:
        run_id = run.info.run_id

        # Train the model
        pipeline = create_pipeline(config, parameters)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Evaluate
        auc_test = roc_auc_score(y_test, y_pred)
        print("Test AUC:", auc_test)

        # Create wrapped model
        wrapped_model = CreditDefaultModelWrapper(pipeline)
        example_input = X_test.iloc[0:1]
        example_prediction = wrapped_model.predict(None, example_input)
        print("Example Prediction:", example_prediction)

        # Log everything
        mlflow.log_param("model_type", "LightGBM with preprocessing")
        mlflow.log_params(parameters)
        mlflow.log_metric("AUC", auc_test)

        # Log dataset
        dataset = mlflow.data.from_spark(
            train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0"
        )
        mlflow.log_input(dataset, context="training")

        # Log both models
        signature = infer_signature(model_input=X_train, model_output=example_prediction)

        # Log wrapped model
        mlflow.pyfunc.log_model(
            python_model=wrapped_model,
            artifact_path="pyfunc_credit_default_model",
            code_paths=["wheel/credit_default_databricks-0.0.9-py3-none-any.whl"],
            signature=signature,
        )

        # Register models
        model_name = f"{catalog_name}.{schema_name}.credit_default_model_pyfunc"
        model_version = mlflow.register_model(
            model_uri=f"runs:/{run_id}/pyfunc_credit_default_model", name=model_name, tags={"branch": "serving"}
        )

        # Set alias
        model_version_alias = "the_best_model"
        client.set_registered_model_alias(model_name, model_version_alias, "4")

        # Save model version info
        with open("model_version.json", "w") as json_file:
            json.dump(model_version.__dict__, json_file, indent=4)


if __name__ == "__main__":
    main()

# COMMAND ----------
