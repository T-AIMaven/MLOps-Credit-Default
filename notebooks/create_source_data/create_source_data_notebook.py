# Databricks notebook source
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

# from databricks.connect import DatabricksSession
from credit_default.utils import load_config

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
# spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------

# Load configuration
config = load_config("../../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# 37354 is the original number of rows in the features_balanced after first SMOTE
# 100 is the number of synthetic rows to generate each time running this notebook
# Load train and test sets
features_balanced = spark.table(f"{catalog_name}.{schema_name}.features_balanced").toPandas()
existing_ids = set(int(id) for id in features_balanced["Id"])

# COMMAND ----------

len(existing_ids)

# COMMAND ----------

min(existing_ids), max(existing_ids)

# COMMAND ----------


# Generate a dataframe with unique values for each column with few unique values
# to identify the discrete values
def generate_unique_values_dataframe(df, columns):
    unique_values = {col: df[col].dropna().unique().tolist() for col in columns}
    return pd.DataFrame([unique_values])


# Load train and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
combined_set = pd.concat([train_set, test_set], ignore_index=True)

# Columns with few unique values (Age is the largest with 56 unique values)
columns = ["Sex", "Education", "Marriage", "Age", "Pay_0", "Pay_2", "Pay_3", "Pay_4", "Pay_5", "Pay_6", "Default"]

result = generate_unique_values_dataframe(combined_set, columns)
print(result)

# COMMAND ----------


# Define function to create synthetic data without random state
# This will add some data drift in the Bill_amt columns
def create_synthetic_data(df, num_rows=100):
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

    return synthetic_data


# Create synthetic data
synthetic_df = create_synthetic_data(combined_set)

# Move "Id" to the first position
columns = ["Id"] + [col for col in synthetic_df.columns if col != "Id"]
synthetic_df = synthetic_df[columns]

# COMMAND ----------

synthetic_df.tail()

# COMMAND ----------

list(synthetic_df.Id)

# COMMAND ----------

combined_set.Bill_amt2.min(), combined_set.Bill_amt2.max()

# COMMAND ----------

# Some values are outside the original column names (data drift)
synthetic_df.Bill_amt2.min(), synthetic_df.Bill_amt2.max()

# COMMAND ----------

synthetic_df.info()

# COMMAND ----------

# Create source_data table with the same schema as train_set
train_set_schema = spark.table(f"{catalog_name}.{schema_name}.train_set").schema

# Create an empty DataFrame with the same schema
empty_source_data_df = spark.createDataFrame(data=[], schema=train_set_schema)

# Create an empty source_data table
empty_source_data_df.write.mode("overwrite").saveAsTable(f"{catalog_name}.{schema_name}.source_data")

print(f"Empty table '{catalog_name}.{schema_name}.source_data' created successfully.")

# COMMAND ----------

# Create synthetic data
existing_schema = spark.table(f"{catalog_name}.{schema_name}.source_data").schema

synthetic_spark_df = spark.createDataFrame(synthetic_df, schema=existing_schema)

# Append synthetic data as new data to source_data table
synthetic_spark_df.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.source_data")
