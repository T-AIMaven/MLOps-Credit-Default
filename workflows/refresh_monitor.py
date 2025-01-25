"""
This script processes and monitors predictions for a credit default model deployed on Databricks.
Key functionality:
- Initializes a Spark session and Databricks workspace client.
- Loads project configuration from a YAML file to retrieve catalog, schema, and target column details.
- Extracts and parses request and response data from a model inference payload table, including:
  - Request metadata such as input features.
  - Response metadata such as predictions and traceability details.
- Joins parsed inference data with train, test, and inference datasets to create a comprehensive monitoring dataset.
- Joins the dataset with balanced feature data.
- Appends the processed monitoring data to a dedicated model monitoring table for ongoing analysis.
- Runs Databricks quality monitors to refresh insights and validate data quality for the monitoring table.
The script ensures that model inference, monitoring, and data quality processes are streamlined.
"""

import sys

from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, StringType, StructField, StructType

# from databricks.connect import DatabricksSession
from credit_default.utils import load_config, setup_logging

# Set up logging
setup_logging(log_file="")


try:
    # Initialize Spark session
    spark = SparkSession.builder.getOrCreate()
    logger.info("Spark session initialized.")

    # Initialize Databricks workspace client
    workspace = WorkspaceClient()
    logger.info("Databricks workspace client initialized.")

    # Extract configuration details
    config = load_config("../project_config.yml")
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    target = config.target[0].new_name
    logger.debug(f"Catalog: {catalog_name}, Schema: {schema_name}")

    # Load inference table
    logger.info("Loading inference table...")
    inf_table = spark.sql(f"SELECT * FROM {catalog_name}.{schema_name}.`model-serving-feature_payload`")
    logger.success("Inference table loaded successfully.")

    ## Dataframe records on payload table under response column
    # {"dataframe_records": [{"Id": "43565", "Limit_bal": 198341.0, "Sex": 2.0,
    # "Education": 2.0, "Marriage": 2.0, "Age": 26.0, "Pay_0": 2.0, "Pay_2": 1.0,
    # "Pay_3": 6.0, "Pay_4": 4.0, "Pay_5": 8.0, "Pay_6": 6.0, "Bill_amt1": -44077.0,
    # "Bill_amt2": 15797.0, "Bill_amt3": 66567.0, "Bill_amt4": 54582.0, "Bill_amt5": 79211.0,
    # "Bill_amt6": 129060.0, "Pay_amt1": 13545.0, "Pay_amt2": 20476.0, "Pay_amt3": 8616.0,
    # "Pay_amt4": 3590.0, "Pay_amt5": 22999.0, "Pay_amt6": 3605.0}]}
    logger.info("Defining schemas...")
    request_schema = StructType(
        [
            StructField(
                "dataframe_records",
                ArrayType(
                    StructType(
                        [
                            StructField("Id", StringType(), True),
                            StructField("Limit_bal", DoubleType(), True),
                            StructField("Sex", DoubleType(), True),
                            StructField("Education", DoubleType(), True),
                            StructField("Marriage", DoubleType(), True),
                            StructField("Age", DoubleType(), True),
                            StructField("Pay_0", DoubleType(), True),
                            StructField("Pay_2", DoubleType(), True),
                            StructField("Pay_3", DoubleType(), True),
                            StructField("Pay_4", DoubleType(), True),
                            StructField("Pay_5", DoubleType(), True),
                            StructField("Pay_6", DoubleType(), True),
                            StructField("Bill_amt1", DoubleType(), True),
                            StructField("Bill_amt2", DoubleType(), True),
                            StructField("Bill_amt3", DoubleType(), True),
                            StructField("Bill_amt4", DoubleType(), True),
                            StructField("Bill_amt5", DoubleType(), True),
                            StructField("Bill_amt6", DoubleType(), True),
                            StructField("Pay_amt1", DoubleType(), True),
                            StructField("Pay_amt2", DoubleType(), True),
                            StructField("Pay_amt3", DoubleType(), True),
                            StructField("Pay_amt4", DoubleType(), True),
                            StructField("Pay_amt5", DoubleType(), True),
                            StructField("Pay_amt6", DoubleType(), True),
                        ]
                    )
                ),
                True,
            )
        ]
    )

    # Standard Databricks schema for the response
    response_schema = StructType(
        [
            StructField("predictions", ArrayType(DoubleType()), True),
            StructField(
                "databricks_output",
                StructType(
                    [StructField("trace", StringType(), True), StructField("databricks_request_id", StringType(), True)]
                ),
                True,
            ),
        ]
    )
    logger.success("Schemas defined successfully.")

    # Parse request and response
    logger.info("Parsing request and response columns...")
    inf_table_parsed = inf_table.withColumn("parsed_request", F.from_json(F.col("request"), request_schema))

    inf_table_parsed = inf_table_parsed.withColumn("parsed_response", F.from_json(F.col("response"), response_schema))

    df_exploded = inf_table_parsed.withColumn("record", F.explode(F.col("parsed_request.dataframe_records")))

    df_final = df_exploded.select(
        F.from_unixtime(F.col("timestamp_ms") / 1000).cast("timestamp").alias("timestamp"),
        "timestamp_ms",
        "databricks_request_id",
        "execution_time_ms",
        F.col("record.Id").alias("Id"),
        F.col("parsed_response.predictions")[0].alias("prediction"),
        F.lit("credit_model_feature").alias("model_name"),
    )
    logger.success("Request and response parsed successfully.")

    # Join data with train/test/inference sets
    logger.info("Joining data with train/test/inference sets...")
    test_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
    inference_set_normal = spark.table(f"{catalog_name}.{schema_name}.inference_set_normal")
    inference_set_skewed = spark.table(f"{catalog_name}.{schema_name}.inference_set_skewed")

    inference_set = inference_set_normal.union(inference_set_skewed)

    df_final_with_status = (
        df_final.join(test_set.select("Id", target), on="Id", how="left")
        .withColumnRenamed(target, "default_test")
        .join(inference_set.select("Id", target), on="Id", how="left")
        .withColumnRenamed(target, "default_inference")
        .select("*", F.coalesce(F.col("default_test"), F.col("default_inference")).alias("default"))
        .drop("default_test", "default_inference")
        .withColumn("default", F.col("default").cast("double"))
        .withColumn("prediction", F.col("prediction").cast("double"))
        .dropna(subset=["default", "prediction"])
    )
    logger.success("Data joined successfully.")

    # Join with features and write to model monitoring table
    logger.info("Joining with features and writing to model monitoring table...")
    features_balanced = spark.table(f"{catalog_name}.{schema_name}.features_balanced")
    df_final_with_features = df_final_with_status.join(features_balanced, on="Id", how="left")
    df_final_with_features.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.model_monitoring")
    logger.success("Data written to model monitoring table successfully.")

    # Run quality monitors
    logger.info("Running quality monitors...")
    workspace.quality_monitors.run_refresh(table_name=f"{catalog_name}.{schema_name}.model_monitoring")
    logger.success("Quality monitors refreshed successfully.")

except Exception as e:
    logger.error(f"An error occurred: {e}")
    sys.exit(1)  # Exit with a failure code
