# Databricks notebook source
import json

import mlflow
import pandas as pd

# from pyspark.sql import SparkSession
from databricks.connect import DatabricksSession
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.models import infer_signature

from credit_default.utils import load_config

# spark = SparkSession.builder.getOrCreate()
spark = DatabricksSession.builder.getOrCreate()

# Load environment variables
load_dotenv()

# COMMAND ----------

# Tracking and registry URIs
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# Load configuration from YAML file
config = load_config("../../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name
parameters = config.parameters


# COMMAND ----------

# Check last run
run_id = mlflow.search_runs(
    experiment_names=["/Shared/credit_default"],
    filter_string="tags.branch='serving'",
).run_id[0]

print(run_id)

# COMMAND ----------

# Load Model
model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")


# COMMAND ----------


# Model Wrapper
class CreditDefaultModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            predictions = self.model.predict(model_input)
            predictions = {"Prediction": predictions[0]}
            return predictions
        else:
            raise ValueError("Input must be a pandas DataFrame.")


# COMMAND ----------

# Load training and testing sets from Databricks tables
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set_spark = spark.table(f"{catalog_name}.{schema_name}.test_set")

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

wrapped_model = CreditDefaultModelWrapper(model)  # we pass the loaded model to the wrapper
example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
example_prediction = wrapped_model.predict(context=None, model_input=example_input)
print("Example Prediction:", example_prediction)


# COMMAND ----------

# Set up the experiment
mlflow.set_experiment(experiment_name="/Shared/credit_default_pyfunc")


# Start an MLflow run to track the training process
with mlflow.start_run(tags={"branch": "serving"}) as run:
    run_id = run.info.run_id

    signature = infer_signature(model_input=X_train, model_output={"Prediction": example_prediction})

    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")

    mlflow.log_input(dataset, context="training")

    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc_credit_default_model",
        code_paths=["wheel/credit_default_databricks-0.0.9-py3-none-any.whl"],
        signature=signature,
    )

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc_credit_default_model")
loaded_model.unwrap_python_model()

# COMMAND ----------

model_name = f"{catalog_name}.{schema_name}.credit_default_model_pyfunc"

model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/pyfunc_credit_default_model", name=model_name, tags={"branch": "serving"}
)

# COMMAND ----------

with open("model_version.json", "w") as json_file:
    json.dump(model_version.__dict__, json_file, indent=4)

# COMMAND ----------

model_version_alias = "the_best_model"
client.set_registered_model_alias(model_name, model_version_alias, "2")

model_uri = f"models:/{model_name}@{model_version_alias}"
model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

client.get_model_version_by_alias(model_name, model_version_alias)

# COMMAND ----------
