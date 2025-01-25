"""
This script handles data ingestion and feature table updates for a credit default prediction system.

Key functionality:
- Loads the source dataset and identifies new records for processing
- Splits new records into train and test sets based on timestamp
- Updates existing train and test tables with new data
- Inserts the latest feature values into the feature table for serving
- Triggers and monitors pipeline updates for online feature refresh
- Sets task values to coordinate pipeline orchestration

Workflow:
1. Load source dataset and retrieve recent records with updated timestamps.
2. Split new records into train and test sets (80-20 split).
3. Append new train and test records to existing train and test tables.
4. Insert the latest feature data into the feature table for online serving.
5. Trigger a pipeline update and monitor its status until completion.
6. Set a task value indicating whether new data was processed.
"""

import argparse
import sys
import time

# from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import max as spark_max

from credit_default.utils import load_config, setup_logging

# Set up logging
setup_logging(log_file="")

try:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", action="store", default=None, type=str, required=True)

    args = parser.parse_args()
    root_path = args.root_path
    logger.info("Parsed arguments successfully.")

    # Load configuration
    logger.info("Loading configuration...")
    config_path = f"{root_path}/project_config.yml"
    config = load_config(config_path)
    logger.info("Configuration loaded successfully.")

    # Initialize Databricks workspace client
    workspace = WorkspaceClient()
    logger.info("Databricks workspace client initialized.")

    # Initialize Spark session
    spark = SparkSession.builder.getOrCreate()
    logger.info("Spark session initialized.")

    # Extract configuration details
    pipeline_id = config.pipeline_id
    catalog_name = config.catalog_name
    schema_name = config.schema_name
    logger.debug(f"Catalog: {catalog_name}, Schema: {schema_name}")
    logger.debug(f"Pipeline ID: {pipeline_id}")

    # Load source data table
    source_data_table_name = f"{catalog_name}.{schema_name}.source_data"
    source_data = spark.table(source_data_table_name)
    logger.info(f"Loaded source data from {source_data_table_name}.")

    # Get max update timestamps
    max_train_timestamp = (
        spark.table(f"{catalog_name}.{schema_name}.train_set")
        .select(spark_max("Update_timestamp_utc").alias("max_update_timestamp"))
        .collect()[0]["max_update_timestamp"]
    )
    logger.info(f"Latest timestamp across train sets: {max_train_timestamp}")

    max_test_timestamp = (
        spark.table(f"{catalog_name}.{schema_name}.test_set")
        .select(spark_max("Update_timestamp_utc").alias("max_update_timestamp"))
        .collect()[0]["max_update_timestamp"]
    )
    logger.info(f"Latest timestamp across test sets: {max_test_timestamp}")

    latest_timestamp = max(max_train_timestamp, max_test_timestamp)
    logger.info(f"Latest timestamp across train and test sets: {latest_timestamp}")

    # Filter new data
    new_data = source_data.filter(col("Update_timestamp_utc") > latest_timestamp)
    new_data_count = new_data.count()
    logger.info(f"Found {new_data_count} new rows in source data.")

    # Split new data into train and test sets
    new_data_train, new_data_test = new_data.randomSplit([0.8, 0.2], seed=42)
    affected_rows_train = new_data_train.count()
    affected_rows_test = new_data_test.count()
    logger.info(f"New train data rows: {affected_rows_train}, New test data rows: {affected_rows_test}")

    # Append new data to train and test sets
    new_data_train.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.train_set")
    new_data_test.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.test_set")
    logger.info("Train and test sets updated successfully.")

    # Handle feature table update
    if affected_rows_train > 0 or affected_rows_test > 0:
        columns = config.features.clean
        columns_str = ", ".join(f"s.{col}" for col in columns)
        logger.debug(f"Columns for feature table update: {columns_str}")

        # spark.sql(f"""
        #     INSERT INTO {catalog_name}.{schema_name}.features_balanced
        #     SELECT DISTINCT {columns_str}
        #     FROM {catalog_name}.{schema_name}.source_data s
        #     JOIN (
        #         SELECT Id, Update_timestamp_utc
        #         FROM {catalog_name}.{schema_name}.train_set
        #         WHERE Update_timestamp_utc > '{latest_timestamp}'

        #         UNION ALL

        #         SELECT Id, Update_timestamp_utc
        #         FROM {catalog_name}.{schema_name}.test_set
        #         WHERE Update_timestamp_utc > '{latest_timestamp}'
        #     ) new_records
        #     ON s.Id = new_records.Id
        #     WHERE s.Update_timestamp_utc > '{latest_timestamp}'
        # """)
        # logger.info("Feature table updated successfully.")

        # Verify the number of current_rows
        current_rows = spark.sql(f"""
                SELECT COUNT(*) as count
                FROM {catalog_name}.{schema_name}.features_balanced
            """).collect()[0]["count"]

        spark.sql(f"""
            INSERT INTO {catalog_name}.{schema_name}.features_balanced
            SELECT DISTINCT {columns_str}
            FROM {catalog_name}.{schema_name}.source_data s
            WHERE EXISTS (
                SELECT 1
                FROM (
                    SELECT Id FROM {catalog_name}.{schema_name}.train_set
                    WHERE Update_timestamp_utc > '{latest_timestamp}'
                    UNION ALL
                    SELECT Id FROM {catalog_name}.{schema_name}.test_set
                    WHERE Update_timestamp_utc > '{latest_timestamp}'
                ) new_records
                WHERE s.Id = new_records.Id
            )
            AND s.Update_timestamp_utc > '{latest_timestamp}'
        """)

        # Verify the number of current_rows updated
        new_rows = spark.sql(f"""
                SELECT COUNT(*) as count
                FROM {catalog_name}.{schema_name}.features_balanced
            """).collect()[0]["count"]

        logger.info(f"Feature table updated with {new_rows - current_rows} new rows")
        logger.info("Feature table updated successfully.")

        # Update the online feature table via pipeline
        logger.info(f"Starting incremental pipeline update for pipeline ID: {pipeline_id}")
        update_response = workspace.pipelines.start_update(pipeline_id=pipeline_id, full_refresh=False)

        # Monitor pipeline update
        while True:
            update_info = workspace.pipelines.get_update(pipeline_id=pipeline_id, update_id=update_response.update_id)
            state = update_info.update.state.value
            if state == "COMPLETED":
                logger.success("Pipeline update completed successfully.")
                break
            elif state in ["FAILED", "CANCELED"]:
                logger.error(f"Pipeline update failed with state: {state}")
                raise SystemError("Online table failed to update.")
            else:
                logger.debug(f"Pipeline update state: {state}")
            time.sleep(30)

        refreshed = 1
    else:
        logger.info("No new rows to update in train, test, or feature tables.")
        refreshed = 0

    # Set task value
    dbutils.jobs.taskValues.set(key="refreshed", value=refreshed)  # noqa: F821
    logger.info(f"Task value 'refreshed' set to: {refreshed}")

except Exception as e:
    logger.error(f"An error occurred: {e}")
    sys.exit(1)  # Exit with failure code
