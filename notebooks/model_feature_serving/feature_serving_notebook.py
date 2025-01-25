# Databricks notebook source
# MAGIC %pip install --upgrade databricks-sdk

# COMMAND ----------

# %dbutils.library.restartPython()

# COMMAND ----------

"""
Create feature table in unity catalog, it will be a delta table
Create online table which uses the feature delta table created in the previous step
Create a feature spec. When you create a feature spec,
you specify the source Delta table.
This allows the feature spec to be used in both offline and online scenarios.
For online lookups, the serving endpoint automatically uses the online table to perform low-latency feature lookups.
The source Delta table and the online table must use the same primary key.

"""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
import pandas as pd
import requests
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable,
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from pyspark.sql import SparkSession

from credit_default.utils import load_config

# COMMAND ----------

config = load_config("../../project_config.yml")

print(config)


# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Initialize Databricks clients
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------

# Set the MLflow registry URI
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

catalog_name = config.catalog_name
schema_name = config.schema_name

# Define table names
"""
FeatureTable (Offline): a table stored as a Delta Table in Unity Catalog
that contains ML model features with primary key constraint

Online table is a Databricks-managed read-only low-latency store for real-time serving
"""

feature_table_name = f"{catalog_name}.{schema_name}.credit_default_preds"
online_table_name = f"{catalog_name}.{schema_name}.credit_default_preds_online"

# COMMAND ----------

# Load training and test sets from Catalog
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# COMMAND ----------

df = pd.concat([train_set, test_set])


# COMMAND ----------

# Load the MLflow model for predictions
pipeline = mlflow.sklearn.load_model(f"models:/{catalog_name}.{schema_name}.credit_default_model_basic/1")

# COMMAND ----------

# Define
columns = config.features.clean

# COMMAND ----------

preds_df = df[columns]

# COMMAND ----------

preds_df.loc[:, "Predicted_Default"] = pipeline.predict(df[columns])

# COMMAND ----------

preds_df = spark.createDataFrame(preds_df)

# COMMAND ----------

print(preds_df)

# COMMAND ----------

# 1. Create the feature table in Databricks

fe.create_table(
    name=feature_table_name, primary_keys=["Id"], df=preds_df, description="Credit Default predictions feature table"
)

# COMMAND ----------

# Enable Change Data Feed (tracking of incremental changes (like inserts, updates, and deletes))

spark.sql(f"""
    ALTER TABLE {feature_table_name}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------

# 2. Create the online table using feature table

## OnlineTableSpecTriggeredSchedulingPolicy: Specifies that updates to the online table will be triggered by events (like new data arrival or changes in the offline table)

## source_table_full_name: Source table in the offline feature store

# perform_full_copy=False: Indicates that only incremental changes will be replicated to the online table

spec = OnlineTableSpec(
    primary_key_columns=["Id"],
    source_table_full_name=feature_table_name,
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=False,
)

# COMMAND ----------

# print(workspace.online_tables.get(name=online_table_name))
print(spec)

# COMMAND ----------

# Create the online table in Databricks
on_table = OnlineTable(name=online_table_name, spec=spec)
print(on_table.as_dict())

# COMMAND ----------

# workspace.online_tables.delete(online_table_name)

# COMMAND ----------

# Create the online table in Databricks
on_table = OnlineTable(name=online_table_name, spec=spec)

# ignore "already exists" error
try:
    # Convert OnlineTable to dictionary before passing to create
    online_table_dict = on_table.as_dict()
    online_table_pipeline = workspace.online_tables.create(table=online_table_dict)

except Exception as e:
    if "already exists" in str(e):
        pass
    else:
        raise e

print(workspace.online_tables.get(online_table_name))


# COMMAND ----------

# Pipeline_id
print(workspace.online_tables.get(online_table_name).spec.pipeline_id)

# COMMAND ----------

print(columns)

# COMMAND ----------

# Define features to look up from the feature table

columns_wo_id = columns.copy()
columns_wo_id.remove("Id")

features = [
    FeatureLookup(table_name=feature_table_name, lookup_key="Id", feature_names=columns_wo_id + ["Predicted_Default"])
]

# COMMAND ----------

# # Create the feature spec for serving (this can be found in the catalog under functions)
feature_spec_name = f"{catalog_name}.{schema_name}.return_predictions"

## Uncomment only the first time it is created
# fe.create_feature_spec(name=feature_spec_name,
#                        features=features,
#                        exclude_columns=None)

# COMMAND ----------

# Create a serving endpoint for the credit default predictions
# It might take some time to create 5-10 min

workspace.serving_endpoints.create(
    name="credit-default-feature-serving",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=feature_spec_name,  # feature spec name defined in the previous step
                scale_to_zero_enabled=True,
                workload_size="Small",  # Define the workload size (Small, Medium, Large)
            )
        ]
    ),
)

# COMMAND ----------

# Call The Endpoint

# token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()  # noqa: F821

token = dbutils.secrets.get(scope="secret-scope", key="databricks-token")  # noqa: F821

host = spark.conf.get("spark.databricks.workspaceUrl")


# COMMAND ----------

print(host)

# COMMAND ----------

# Run a prediction for an specific Id

start_time = time.time()

serving_endpoint = f"https://{host}/serving-endpoints/credit-default-feature-serving/invocations"

response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": [{"Id": "182"}]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

# Another way to call the endpoint

response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_split": {"columns": ["Id"], "data": [["182"]]}},
)

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

## Load Test for more than 1 request

# Initialize variables
serving_endpoint = f"https://{host}/serving-endpoints/credit-default-feature-serving/invocations"

id_list = preds_df.select("Id").rdd.flatMap(lambda x: x).collect()

headers = {"Authorization": f"Bearer {token}"}
num_requests = 10


# Function to make a request and record latency
def send_request():
    random_id = random.choice(id_list)
    start_time = time.time()
    response = requests.post(
        serving_endpoint,
        headers=headers,
        json={"dataframe_records": [{"Id": random_id}]},
    )
    end_time = time.time()
    latency = end_time - start_time  # Calculate latency for this request
    return response.status_code, latency


# COMMAND ----------

# Measure total execution time
total_start_time = time.time()
latencies = []

# Send requests concurrently
with ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(send_request) for _ in range(num_requests)]

    for future in as_completed(futures):
        status_code, latency = future.result()
        latencies.append(latency)

total_end_time = time.time()
total_execution_time = total_end_time - total_start_time

# Calculate the average latency
average_latency = sum(latencies) / len(latencies)

print("\nTotal execution time:", total_execution_time, "seconds")
print("Average latency per request:", average_latency, "seconds")
