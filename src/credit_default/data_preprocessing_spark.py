import os
from typing import Tuple

import pandas as pd
from databricks.connect import DatabricksSession
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from credit_default.data_cleaning_spark import DataCleaning
from credit_default.utils import Config

# Load environment variables
load_dotenv()

# spark = SparkSession.builder.getOrCreate()
spark = DatabricksSession.builder.getOrCreate()

FILEPATH_DATABRICKS = os.environ["FILEPATH_DATABRICKS"]
PREPROCESSING_LOGS = os.environ["PREPROCESSING_LOGS"]


class DataPreprocessor:
    """
    A class for preprocessing credit default data, including scaling features.

    Attributes:
        data_cleaning (DataCleaning): An instance of the DataCleaning class used for data preprocessing.
        cleaned_data (pd.DataFrame): The cleaned DataFrame after preprocessing.
        features_robust (list): List of feature names for robust scaling.
        X (pd.DataFrame): Features DataFrame after cleaning.
        y (pd.Series): Target Series after cleaning.
        preprocessor (ColumnTransformer): ColumnTransformer for scaling the features.
    """

    def __init__(self, filepath: str, config: Config, spark: SparkSession):
        """
        Initializes the DataPreprocessor class.

        Args:
            filepath (str): The path to the CSV file containing the data.
            config (Config): The configuration model containing preprocessing settings.
        """
        self.catalog_name = config.catalog_name
        self.schema_name = config.schema_name
        self.spark = spark

        try:
            # Initialize DataCleaning to preprocess data
            logger.info("Initializing data cleaning process")
            self.data_cleaning = DataCleaning(filepath, config, spark)
            self.cleaned_data = self.data_cleaning.preprocess_data()
            logger.info("Data cleaning process completed")

            # Define robust features for scaling from config
            self.features_robust = config.features.robust

            # Define features and target
            self.X = self.cleaned_data.drop(columns=[target.new_name for target in config.target])
            self.y = self.cleaned_data[config.target[0].new_name]

            # Set up the ColumnTransformer for scaling
            logger.info("Setting up ColumnTransformer for scaling")
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("robust_scaler", RobustScaler(), self.features_robust)  # Apply RobustScaler to selected features
                ],
                remainder="passthrough",  # Keep other columns unchanged
            )
        except KeyError as e:
            logger.error(f"KeyError encountered during initialization: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"An error occurred during initialization: {str(e)}")
            raise

    def get_processed_data(self) -> Tuple:
        """
        Retrieves the processed features, target, and preprocessor.

        Returns:
            Tuple: A tuple containing:
                - pd.DataFrame: The features DataFrame.
                - pd.Series: The target Series.
                - ColumnTransformer: The preprocessor for scaling.
        """
        try:
            logger.info("Retrieving processed data and preprocessor")
            logger.info(f"Feature columns in X: {self.X.columns.tolist()}")

            # Log shapes of processed data
            logger.info(f"Data preprocessing completed. Shape of X: {self.X.shape}, Shape of y: {self.y.shape}")

            return self.X, self.y, self.preprocessor

        except Exception as e:
            logger.error(f"An error occurred during data preprocessing: {str(e)}")

    def split_data(self, test_size=0.2, random_state=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the cleaned DataFrame into training and test sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        train_set = pd.concat([X_train, y_train], axis=1)
        test_set = pd.concat([X_test, y_test], axis=1)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save the train and test sets into Databricks tables."""
        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "Update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "Update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.catalog_name}.{self.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("overwrite").saveAsTable(f"{self.catalog_name}.{self.schema_name}.test_set")

        spark.sql(
            f"ALTER TABLE {self.catalog_name}.{self.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {self.catalog_name}.{self.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )


# if __name__ == "__main__":
#     # Configure logger using setup_logging
#     setup_logging(PREPROCESSING_LOGS)  # Set up logging with the log file path

#     # Load configuration from YAML file
#     config = load_config("../../project_config.yml")  # Returns Config instance

#     # Test the DataPreprocessor class
#     try:
#         logger.info(f"Initializing DataPreprocessor with config: {config}")
#         preprocessor = DataPreprocessor(FILEPATH_DATABRICKS, config)
#         X, y, preprocessor_model = preprocessor.get_processed_data()

#         logger.info(f"Feature columns in X: {X.columns.tolist()}")

#         # Log shapes of processed data
#         logger.info(f"Data preprocessing completed. Shape of X: {X.shape}, Shape of y: {y.shape}")
#     except Exception as e:
#         logger.error(f"An error occurred during data preprocessing: {str(e)}")

#     logger.info("DataPreprocessor script completed")
