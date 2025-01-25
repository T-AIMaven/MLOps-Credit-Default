# Databricks notebook source
import datetime
import itertools
import time

import requests
from pyspark.sql import SparkSession

from credit_default.utils import load_config

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# Load configuration
config = load_config("../../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name
columns = config.features.clean

# COMMAND ----------

# Load train/test set and convert to Pandas
inference_data_normal = spark.table(f"{catalog_name}.{schema_name}.inference_set_normal").toPandas()

inference_data_skewed = spark.table(f"{catalog_name}.{schema_name}.inference_set_skewed").toPandas()

test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# COMMAND ----------

# token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

token = dbutils.secrets.get(scope="secret-scope", key="databricks-token")  # noqa: F821

host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# Sample records from inference datasets
sampled_normal_records = inference_data_normal[columns].to_dict(orient="records")

sampled_skewed_records = inference_data_skewed[columns].to_dict(orient="records")

test_set_records = test_set[columns].to_dict(orient="records")

# COMMAND ----------

# Send request to the endpoint


def send_request_https(dataframe_record):
    model_serving_endpoint = f"https://{host}/serving-endpoints/credit-default-model-serving-feature/invocations"
    response = requests.post(
        model_serving_endpoint,
        headers={"Authorization": f"Bearer {token}"},
        json={"dataframe_records": [dataframe_record]},
    )
    return response


# COMMAND ----------

# Loop over test records and send requests for 10 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=10)

for index, record in enumerate(itertools.cycle(test_set_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for test data, index {index}")

    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")

    time.sleep(0.2)

# COMMAND ----------

# Loop over normal records and send requests for 10 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=10)

for index, record in enumerate(itertools.cycle(sampled_normal_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for normal data, index {index}")

    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")

    time.sleep(0.2)

# COMMAND ----------

# Loop over skewed records and send requests for 15 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=15)

for index, record in enumerate(itertools.cycle(sampled_skewed_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for skewed data, index {index}")

    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")

    time.sleep(0.2)
