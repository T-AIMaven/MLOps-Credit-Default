# Databricks notebook source
import json

import mlflow
from databricks.connect import DatabricksSession
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature

# from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score  # classification_report, confusion_matrix,
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from credit_default.utils import load_config

# spark = SparkSession.builder.getOrCreate()
spark = DatabricksSession.builder.getOrCreate()

# Load environment variables
load_dotenv()


# COMMAND ----------

# tracking and registry URIs
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# Load configuration from YAML file
config = load_config("../../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name
parameters = config.parameters


# COMMAND ----------

# Load training and testing sets from Databricks tables
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

X_train = train_set.drop(columns=["Default", "Id", "Update_timestamp_utc"])
y_train = train_set["Default"]

X_test = test_set.drop(columns=["Default", "Id", "Update_timestamp_utc"])
y_test = test_set["Default"]

# COMMAND ----------

# Show train features
X_train.head()

# COMMAND ----------

features_robust = config.features.robust
print(features_robust)

# COMMAND ----------


preprocessor = ColumnTransformer(
    transformers=[("robust_scaler", RobustScaler(), features_robust)],
    remainder="passthrough",
)

# Create the pipeline with preprocessing and the LightGBM classifier
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**parameters))])

# COMMAND ----------

# Set up the experiment
mlflow.set_experiment(experiment_name="/Shared/credit_default")

# Start an MLflow run to track the training process
with mlflow.start_run(tags={"branch": "serving"}) as run:
    run_id = run.info.run_id

    # Train the model
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model performance
    auc_test = roc_auc_score(y_test, y_pred)

    print("Test AUC:", auc_test)

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("AUC", auc_test)

    # Log the input dataset
    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")

    # Log the model
    signature = infer_signature(model_input=X_train, model_output=y_pred)
    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="lightgbm-pipeline-model", signature=signature)

# Register the model in MLflow
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model",
    name=f"{catalog_name}.{schema_name}.credit_default_model_basic",
    tags={"branch": "serving"},
)

# Optionally, save the model version information
with open("model_version.json", "w") as json_file:
    json.dump(model_version.__dict__, json_file, indent=4)
# COMMAND ----------
