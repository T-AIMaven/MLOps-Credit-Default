# Databricks notebook source
import os

from databricks.connect import DatabricksSession
from dotenv import load_dotenv
from loguru import logger

# from pyspark.sql import SparkSession
from credit_default.data_preprocessing_spark import DataPreprocessor
from credit_default.utils import load_config, setup_logging

# spark = SparkSession.builder.getOrCreate()
spark = DatabricksSession.builder.getOrCreate()

# Load environment variables
load_dotenv()


# COMMAND ----------
# Load configuration from YAML file
config = load_config("../../project_config.yml")

print(config.catalog_name)
print(config.schema_name)

# COMMAND ----------

# FILEPATH_DATABRICKS = os.environ["FILEPATH_DATABRICKS"]
FILEPATH_DATABRICKS = f"dbfs:/Volumes/{config.catalog_name}/{config.schema_name}/data/data.csv"
PREPROCESSING_LOGS = os.environ["PREPROCESSING_LOGS"]
print(FILEPATH_DATABRICKS)

# COMMAND ----------
# from IPython.display import display

# display(spark.read.csv(FILEPATH_DATABRICKS, header=True, inferSchema=True))

# COMMAND ----------
# Test the DataPreprocessor class
try:
    # Configure logger using setup_logging
    setup_logging(PREPROCESSING_LOGS)  # Set up logging with the log file path

    logger.info(f"Initializing DataPreprocessor with config: {config}")
    preprocessor = DataPreprocessor(FILEPATH_DATABRICKS, config, spark)
    X, y, preprocessor_model = preprocessor.get_processed_data()

    # Split data into training and test sets
    train_set, test_set = preprocessor.split_data()
    logger.info(f"Data split completed. Train shape: {train_set.shape}, Test shape: {test_set.shape}")

    # Save train and test sets to the Databricks catalog
    preprocessor.save_to_catalog(train_set, test_set, spark)
    logger.info("Train and test sets saved to catalog successfully.")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")

    logger.info("DataPreprocessor script completed")

# COMMAND ----------
