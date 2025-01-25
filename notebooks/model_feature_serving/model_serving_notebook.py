# Databricks notebook source
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

from credit_default.utils import load_config

# COMMAND ----------

workspace = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

config = load_config("../../project_config.yml")
parameters = config.parameters
print(config)

# COMMAND ----------

catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

print(catalog_name), print(schema_name)

# COMMAND ----------

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()

# COMMAND ----------

# # Serving the model (creates endpoint)

# workspace.serving_endpoints.create(
#     name="credit-default-model-serving",
#     config=EndpointCoreConfigInput(
#         served_entities=[
#             ServedEntityInput(
#                 entity_name=f"{catalog_name}.{schema_name}.credit_default_model_basic",
#                 scale_to_zero_enabled=True,
#                 workload_size="Small",
#                 entity_version=1,
#             )
#         ],
#     # Optional if only 1 entity is served (this allow to split)
#     traffic_config=TrafficConfig(
#         routes=[
#             Route(served_model_name="credit_default_model_basic-1",
#                   traffic_percentage=100)
#         ]
#         ),
#     ),
# )

# COMMAND ----------

# # Serving the model (creates endpoint with split)


# workspace.serving_endpoints.create(
#     name="credit-default-model-serving",
#     config=EndpointCoreConfigInput(
#         served_entities=[
#             ServedEntityInput(
#                 entity_name=f"{catalog_name}.{schema_name}.credit_default_model_basic",
#                 scale_to_zero_enabled=True,
#                 workload_size="Small",
#                 entity_version=1,
#             ),
#             ServedEntityInput(
#                 entity_name=f"{catalog_name}.{schema_name}.credit_default_model_advanced",
#                 scale_to_zero_enabled=True,
#                 workload_size="Medium",
#                 entity_version=1,
#             ),
#         ],
#         traffic_config=TrafficConfig(
#             routes=[
#                 Route(served_model_name="credit_default_model_basic-1",
#                       traffic_percentage=50),
#                 Route(served_model_name="credit_default_model_advanced-1",
#                       traffic_percentage=50),
#             ]
#         ),
#     ),
# )

# COMMAND ----------

## Call the endpoint
# Call The Endpoint

# token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()  # noqa: F821

token = dbutils.secrets.get(scope="secret-scope", key="databricks-token")  # noqa: F821

host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

print(host)

# COMMAND ----------

# Get required columns

columns = config.features.clean

required_columns = columns.copy()
required_columns.remove("Id")

# COMMAND ----------

# Create records for prediction

sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")

dataframe_records = [[record] for record in sampled_records]

len(dataframe_records)

# COMMAND ----------

# Run 1 prediction

start_time = time.time()

model_serving_endpoint = f"https://{host}/serving-endpoints/credit-default-model-serving/invocations"
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

## Load Test

# Initialize variables
model_serving_endpoint = f"https://{host}/serving-endpoints/credit-default-model-serving/invocations"

headers = {"Authorization": f"Bearer {token}"}
num_requests = 1000


# Function to make a request and record latency
def send_request():
    random_record = random.choice(dataframe_records)
    start_time = time.time()
    response = requests.post(
        model_serving_endpoint,
        headers=headers,
        json={"dataframe_records": random_record},
    )
    end_time = time.time()
    latency = end_time - start_time
    return response.status_code, latency


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
