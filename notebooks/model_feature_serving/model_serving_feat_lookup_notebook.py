# Databricks notebook source
# MAGIC %pip install --upgrade databricks-sdk

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# !pip list

# COMMAND ----------

import time

import requests
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

spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()

# COMMAND ----------

config = load_config("../../project_config.yml")

print(config)

# COMMAND ----------

catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# Create online table using the balanced features table as source

spec = OnlineTableSpec(
    primary_key_columns=["Id"],
    source_table_full_name=f"{catalog_name}.{schema_name}.features_balanced",
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=False,
)

# COMMAND ----------

# Create the online table in Databricks
online_table_name = f"{catalog_name}.{schema_name}.features_balanced_online"

on_table = OnlineTable(name=online_table_name, spec=spec)
print(on_table.as_dict())

# COMMAND ----------

# online_table_name = f"{catalog_name}.{schema_name}.features_balanced_online"
# workspace.online_tables.delete(online_table_name)

# COMMAND ----------

# Create the online table in Databricks
on_table = OnlineTable(name=online_table_name, spec=spec)

# ignore "already exists" error
try:
    # Convert OnlineTable to dictionary before passing to create
    online_table_dict = on_table
    online_table_pipeline = workspace.online_tables.create(table=online_table_dict)

except Exception as e:
    if "already exists" in str(e):
        pass
    else:
        raise e

print(workspace.online_tables.get(online_table_name))

# COMMAND ----------

# Pipeline_id to be added into the project_config.yml
print(workspace.online_tables.get(online_table_name).spec.pipeline_id)

# COMMAND ----------

## Can build endpoint with "pyarrow>=14.0.0, <15", for model creation
## Used wheel version 0.0.9

## Otherwise there is an error
# 22 99.07 The conflict is caused by:
# 22 99.07     The user requested pyarrow==15.0.2
# 22 99.07     mlflow 2.17.2 depends on pyarrow<18 and >=4.0.0
# 22 99.07     databricks-feature-lookup 1.2.0 depends on pyarrow==14.*

workspace.serving_endpoints.create(
    name="credit-default-model-serving-feature",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.credit_model_feature",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=1,
            )
        ]
    ),
)

# COMMAND ----------

## Call the endpoint

# token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()  # noqa: F821

token = dbutils.secrets.get(scope="secret-scope", key="databricks-token")  # noqa: F821

host = spark.conf.get("spark.databricks.workspaceUrl")


# COMMAND ----------

# Removed columns used for training set creation (see below code) as this will be taken from feature lookup

required_columns = ["Id"]


# COMMAND ----------

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()

sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

print(train_set.dtypes)


# COMMAND ----------

dataframe_records[0]


# COMMAND ----------

start_time = time.time()

model_serving_endpoint = f"https://{host}/serving-endpoints/credit-default-model-serving-feature/invocations"

response = requests.post(
    f"{model_serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records[0]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

credit_features = spark.table(f"{catalog_name}.{schema_name}.features_balanced").toPandas()

print(credit_features.dtypes)
