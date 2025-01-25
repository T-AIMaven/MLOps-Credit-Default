# Databricks notebook source
import hashlib
import time

import mlflow
import pandas as pd
import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from lightgbm import LGBMClassifier
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score  # classification_report, confusion_matrix,
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from credit_default.utils import load_config

# COMMAND ----------

# Set up MLflow for tracking and model registry
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

config = load_config("../../project_config.yml")
print(config)

# COMMAND ----------

# Initialize the MLflow client for model management
client = MlflowClient()

# COMMAND ----------

# Extract key configuration details
catalog_name = config.catalog_name
schema_name = config.schema_name
parameters = config.parameters
features_robust = config.features.robust

ab_test_params = config.ab_test


# COMMAND ----------

# Set up specific parameters for model A and model B as part of the A/B test

parameters_a = {
    "learning_rate": ab_test_params["learning_rate_a"],
    "force_col_wise": ab_test_params["force_col_wise"],
}

print(parameters_a)

# COMMAND ----------

# Set up specific parameters for model A and model B as part of the A/B test

parameters_b = {
    "learning_rate": ab_test_params["learning_rate_b"],
    "force_col_wise": ab_test_params["force_col_wise"],
}

print(parameters_b)

# COMMAND ----------

## Load and Prepare Training and Testing Datasets

spark = SparkSession.builder.getOrCreate()

# columns = ['Limit_bal', 'Sex', 'Education', 'Marriage', 'Age', 'Pay_0',
#        'Pay_2', 'Pay_3', 'Pay_4', 'Pay_5', 'Pay_6', 'Bill_amt1', 'Bill_amt2',
#        'Bill_amt3', 'Bill_amt4', 'Bill_amt5', 'Bill_amt6', 'Pay_amt1',
#        'Pay_amt2', 'Pay_amt3', 'Pay_amt4', 'Pay_amt5', 'Pay_amt6']

# Load the training and testing sets from Databricks tables
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = train_set_spark.toPandas()

test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# Define features and target variables
X_train = train_set.drop(columns=["Default", "Id", "Update_timestamp_utc"])
y_train = train_set["Default"]

X_test = test_set.drop(columns=["Default", "Id", "Update_timestamp_utc"])
y_test = test_set["Default"]

# COMMAND ----------

# Define a preprocessor
preprocessor = ColumnTransformer(
    transformers=[("robust_scaler", RobustScaler(), features_robust)],
    remainder="passthrough",
)


# COMMAND ----------

# Create the pipeline with preprocessing and the LightGBM Classifier A
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**parameters_a))])

# COMMAND ----------

# Set the MLflow experiment to track this A/B testing project
mlflow.set_experiment(experiment_name="/Shared/credit_default-ab")
model_name = f"{catalog_name}.{schema_name}.credit_default_model_ab"

# COMMAND ----------

# Start MLflow run to track training of Model A
with mlflow.start_run(tags={"model_class": "A", "branch": "serving"}) as run:
    run_id = run.info.run_id

    # Train the model
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model performance
    auc_test = roc_auc_score(y_test, y_pred)

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters_a)
    mlflow.log_metric("AUC", auc_test)

    # Log the input dataset
    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")

    mlflow.log_input(dataset, context="training")

    # Log the model
    signature = infer_signature(model_input=X_train, model_output=y_pred)
    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="lightgbm-pipeline-model", signature=signature)


## To avoid the integer warning, transform int32 to float64
## X_train.astype({col: 'float64' for col in X_train.select_dtypes(include='int32').columns})


# COMMAND ----------

# Regsiter Model A
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model", name=model_name, tags={"model_class": "A", "branch": "serving"}
)

# COMMAND ----------

print(model_version.version)

# COMMAND ----------

## Assign Alias to registered Model A

# Assign alias for easy reference in future A/B tests
model_version_alias = "model_A"

client.set_registered_model_alias(name=model_name, alias=model_version_alias, version=f"{model_version.version}")

model_uri = f"models:/{model_name}@{model_version_alias}"

model_A = mlflow.sklearn.load_model(model_uri)

# model_A = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

# Create the pipeline with preprocessing and the LightGBM Classifier B
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**parameters_b))])

# COMMAND ----------

# Start MLflow run to track training of Model B
with mlflow.start_run(tags={"model_class": "B", "branch": "serving"}) as run:
    run_id = run.info.run_id

    # Train the model
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model performance
    auc_test = roc_auc_score(y_test, y_pred)

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters_b)
    mlflow.log_metric("AUC", auc_test)

    # Log the input dataset
    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")

    mlflow.log_input(dataset, context="training")

    # Log the model
    signature = infer_signature(model_input=X_train, model_output=y_pred)
    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="lightgbm-pipeline-model", signature=signature)

# COMMAND ----------

# Regsiter Model B
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model", name=model_name, tags={"model_class": "B", "branch": "serving"}
)

# COMMAND ----------

## Assign Alias to registered Model B

# Assign alias for easy reference in future A/B tests
model_version_alias = "model_B"

client.set_registered_model_alias(name=model_name, alias=model_version_alias, version=f"{model_version.version}")

model_uri = f"models:/{model_name}@{model_version_alias}"

model_B = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

## Wrapper that takes both models and send predictions to one
## or the other depending on the row number (hashed)


class CreditDefaultModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, models):
        self.models = models
        self.model_a = models[0]
        self.model_b = models[1]

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            credit_id = str(model_input["Id"].values[0])  # Id number
            hashed_id = hashlib.md5(credit_id.encode(encoding="UTF-8")).hexdigest()

            # convert a hexadecimal (base-16) string into an integer
            if int(hashed_id, 16) % 2:
                predictions = self.model_a.predict(model_input.drop(["Id"], axis=1))
                return {"Prediction": predictions[0], "model": "Model A"}

            else:
                predictions = self.model_b.predict(model_input.drop(["Id"], axis=1))
                return {"Prediction": predictions[0], "model": "Model B"}

        else:
            raise ValueError("Input must be a pandas DataFrame.")


# COMMAND ----------

# Add columns
columns = config.features.clean

X_train = train_set[columns]
X_test = test_set[columns]

# COMMAND ----------

# Run prediction on model A
models = [model_A, model_B]
wrapped_model = CreditDefaultModelWrapper(models)

example_input = X_test.iloc[0:1]  # Select row hashed for mdoel A

example_prediction = wrapped_model.predict(context=None, model_input=example_input)

print("Example Prediction:", example_prediction)

# COMMAND ----------

# Run prediction on model B
models = [model_A, model_B]
wrapped_model = CreditDefaultModelWrapper(models)

example_input = X_test.iloc[112:113]  # Select row hashed for mdoel B

example_prediction = wrapped_model.predict(context=None, model_input=example_input)

print("Example Prediction:", example_prediction)

# COMMAND ----------

# Now we register our wrapped model

mlflow.set_experiment(experiment_name="/Shared/credit_default-ab-testing")
model_name = f"{catalog_name}.{schema_name}.credit_default_model_pyfunc_ab_test"

with mlflow.start_run() as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={"Prediction": 0, "model": "Model B"})

    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")

    mlflow.log_input(dataset, context="training")

    mlflow.pyfunc.log_model(
        python_model=wrapped_model, artifact_path="pyfunc-credit_default-model-ab", signature=signature
    )
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/pyfunc-credit_default-model-ab", name=model_name, tags={"branch": "serving"}
)

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version.version}")

# COMMAND ----------

print(model)

# COMMAND ----------

# Run prediction
predictions_a = model.predict(X_test.iloc[0:1])
predictions_b = model.predict(X_test.iloc[112:113])

print(predictions_a)
print(predictions_b)

# COMMAND ----------

## Create serving endpoint

workspace = WorkspaceClient()

workspace.serving_endpoints.create(
    name="credit_default-model-serving-ab-test",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.credit_default_model_pyfunc_ab_test",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=model_version.version,
            )
        ]
    ),
)


# COMMAND ----------

# token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()  # noqa: F821

token = dbutils.secrets.get(scope="secret-scope", key="databricks-token")  # noqa: F821

host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()

sampled_records = train_set[columns].sample(n=1000, replace=True).to_dict(orient="records")

dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

# Make predictions

start_time = time.time()

model_serving_endpoint = f"https://{host}/serving-endpoints/credit_default-model-serving-ab-test/invocations"

response = requests.post(
    f"{model_serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records[175]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")
