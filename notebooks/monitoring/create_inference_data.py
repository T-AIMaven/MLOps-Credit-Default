# Databricks notebook source
# MAGIC %md
# MAGIC # Generate synthetic datasets for inference

# COMMAND ----------

import time

import numpy as np
import pandas as pd
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.ensemble import RandomForestClassifier

from credit_default.utils import load_config

spark = SparkSession.builder.getOrCreate()

# Load configuration
config = load_config("../../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name
parameters = config.parameters
target = config.target[0].new_name
pipeline_id = config.pipeline_id

# Load train/test set and convert to Pandas
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()

test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# COMMAND ----------

# Define features and target (adjust columns accordingly)
X = train_set.drop(columns=["Id", target, "Update_timestamp_utc"])
y = train_set[target]

# Train a Random Forest model
model = RandomForestClassifier(random_state=parameters["random_state"])
model.fit(X, y)

# Identify the most important features
feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values(
    by="Importance", ascending=False
)

print("Top 5 important features:")
print(feature_importances.head(4))

# Feature importance. Will add drift on this features
# Output: Pay_0        0.097528
#         Age          0.067261
#         Bill_amt1    0.061669
#         Limit_bal    0.061058

# COMMAND ----------

# Get Existing IDs
features_balanced = spark.table(f"{catalog_name}.{schema_name}.features_balanced").toPandas()
existing_ids = set(int(id) for id in features_balanced["Id"])


# COMMAND ----------

# existing_ids = list(int(id) for id in features_balanced["Id"])


# COMMAND ----------

len(list(existing_ids))

# COMMAND ----------

# Define function to create synthetic data without random state
# This will add some data drift in the above columns (if drift=True)


def create_synthetic_data(df, drift=False, num_rows=100):
    synthetic_data = pd.DataFrame()

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]) and column != "Id":
            # Check if the column has a small set of discrete values
            unique_values = df[column].unique()
            if len(unique_values) <= 10:  # Assume discrete values if there are 10 or fewer unique values
                # This includes all above columns except "Age"
                synthetic_data[column] = np.random.choice(unique_values, num_rows)
            elif column.startswith("Pay_amt"):  # Ensure positive values for "Pay_amt" columns
                mean, std = df[column].mean(), df[column].std()
                synthetic_data[column] = np.abs(np.random.normal(mean, std, num_rows)).astype(int).astype(float)
            else:
                # This will add some data drift in the Bill_amt columns
                mean, std = df[column].mean(), df[column].std()
                synthetic_data[column] = np.round(np.random.normal(mean, std, num_rows)).astype(int).astype(float)

        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            if min_date < max_date:
                # Ensure the timestamp is between max_date and current time
                current_time = pd.to_datetime("now")
                if max_date < current_time:
                    timestamp_range_start = max_date.value
                    timestamp_range_end = current_time.value
                    synthetic_data[column] = pd.to_datetime(
                        np.random.randint(timestamp_range_start, timestamp_range_end, num_rows)
                    )
                else:
                    synthetic_data[column] = [max_date] * num_rows
            else:
                synthetic_data[column] = [min_date] * num_rows

    new_ids = []
    # The first synthetic Id must be one greater than the maximum existing Id of the whole dataframe (train + test). If no existing_ids, then starts from 1.
    i = max(existing_ids) + 1 if existing_ids else 1

    while len(new_ids) < num_rows:
        if i not in existing_ids:
            new_ids.append(str(i))  # Convert numeric ID to string
        i += 1

    synthetic_data["Id"] = new_ids

    # Move "Id" to the first position
    columns = ["Id"] + [col for col in synthetic_data.columns if col != "Id"]
    synthetic_data = synthetic_data[columns]

    if drift:
        # Skew the top features to introduce drift
        top_features = ["Limit_bal", "Age", "Pay_0", "Bill_amt1"]  # Select top 4 features
        for feature in top_features:
            if feature in synthetic_data.columns:
                synthetic_data[feature] = synthetic_data[feature] * 1.5

    return synthetic_data


# COMMAND ----------

# Create synthetic data normal
combined_set = pd.concat([train_set, test_set], ignore_index=True)

synthetic_data_normal = create_synthetic_data(combined_set, drift=False, num_rows=200)
print(synthetic_data_normal.dtypes)
print(synthetic_data_normal)

# COMMAND ----------

print(f"Before: {len(existing_ids)}")

# COMMAND ----------

# Update existing_ids with the IDs from synthetic_data_normal
existing_ids.update(int(id) for id in synthetic_data_normal["Id"])

# COMMAND ----------

print(f"After: {len(existing_ids)}")

# COMMAND ----------

# Create synthetic data skewed
synthetic_data_skewed = create_synthetic_data(combined_set, drift=True, num_rows=200)
print(synthetic_data_normal.dtypes)
print(synthetic_data_skewed)

# COMMAND ----------

# Cast columns to match the schema of the Delta table
columns_to_cast = ["Sex", "Education", "Marriage", "Age", "Pay_0", "Pay_2", "Pay_3", "Pay_4", "Pay_5", "Pay_6"]

##  Write normal data to Delta Lake
synthetic_normal_df = spark.createDataFrame(synthetic_data_normal)
for column in columns_to_cast:
    synthetic_normal_df = synthetic_normal_df.withColumn(column, F.col(column).cast("double"))

synthetic_normal_df.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.inference_set_normal")

##  Write synthetic data to Delta Lake
synthetic_skewed_df = spark.createDataFrame(synthetic_data_skewed)
for column in columns_to_cast:
    synthetic_skewed_df = synthetic_skewed_df.withColumn(column, F.col(column).cast("double"))

synthetic_skewed_df.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.inference_set_skewed")

# COMMAND ----------

# Update offline table
workspace = WorkspaceClient()

columns = config.features.clean
columns_str = ", ".join(columns)

# Write normal into feature table; update online table
spark.sql(f"""
    INSERT INTO {catalog_name}.{schema_name}.features_balanced
    SELECT {columns_str}
    FROM {catalog_name}.{schema_name}.inference_set_normal
""")

# Write skewed into feature table; update online table
spark.sql(f"""
    INSERT INTO {catalog_name}.{schema_name}.features_balanced
    SELECT {columns_str}
    FROM {catalog_name}.{schema_name}.inference_set_skewed
""")


# COMMAND ----------

# Update online table
update_response = workspace.pipelines.start_update(pipeline_id=pipeline_id, full_refresh=False)

while True:
    update_info = workspace.pipelines.get_update(pipeline_id=pipeline_id, update_id=update_response.update_id)
    state = update_info.update.state.value
    if state == "COMPLETED":
        break
    elif state in ["FAILED", "CANCELED"]:
        raise SystemError("Online table failed to update.")
    elif state == "WAITING_FOR_RESOURCES":
        print("Pipeline is waiting for resources.")
    else:
        print(f"Pipeline is in {state} state.")
    time.sleep(30)
