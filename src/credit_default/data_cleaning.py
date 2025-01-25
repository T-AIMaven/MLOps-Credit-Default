import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pydantic import ValidationError

from credit_default.utils import Config, Target, load_config, setup_logging

# Load environment variables
load_dotenv()

FILEPATH = os.environ["FILEPATH"]
CLEANING_LOGS = os.environ["CLEANING_LOGS"]


class DataCleaning:
    """
    A class for cleaning and preprocessing credit default data.

    Attributes:
        config (Config): Configuration model containing preprocessing settings
        df (pd.DataFrame): DataFrame containing the data to be processed
        target_config (Target): Configuration for target variable
    """

    def __init__(self, filepath: str, config: Config):
        """
        Initializes the DataCleaning class.

        Args:
            filepath (str): Path to the CSV file containing the data
            config (Config): Configuration model containing preprocessing settings

        Raises:
            FileNotFoundError: If data file doesn't exist
            Exception: If data cleaning fails
        """
        self._validate_file_exists(filepath)
        self.config = config
        self.df = self._load_data(filepath)
        self._setup_target_config()

    @staticmethod
    def _validate_file_exists(filepath: str) -> None:
        """
        Validates that the input file exists.

        Args:
            filepath (str): Path to the CSV file containing the data

        Raises:
            FileNotFoundError: If file does not exist
        """
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"The file {filepath} does not exist")

    def _setup_target_config(self) -> None:
        """Sets up target configuration from config."""
        target_info = self.config.target[0]
        self.target_config = Target(name=target_info.name, dtype=target_info.dtype, new_name=target_info.new_name)

    @staticmethod
    def _load_data(filepath: str) -> pd.DataFrame:
        """
        Loads and validates the input data.

        Args:
            filepath (str): Path to the CSV file

        Returns:
            pd.DataFrame: Loaded DataFrame

        Raises:
            Exception: If data loading or validation fails
        """
        try:
            logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath)
            if df.empty:
                raise Exception("Loaded DataFrame is empty")
            return df
        except pd.errors.EmptyDataError as e:
            raise Exception(f"Failed to load data: {str(e)}") from e

    def _validate_columns(self) -> None:
        """
        Validates that required columns exist in the DataFrame.

        Raises:
            Exception: If DataFrame validation fails
        """
        columns_to_check = [feature.name for feature in self.config.num_features] + [self.target_config.name]
        missing_columns = [col for col in columns_to_check if col not in self.df.columns]
        if missing_columns:
            raise Exception(f"Missing required columns: {', '.join(missing_columns)}")

    def _validate_data_types(self) -> None:
        """Validates data types of key columns.

        Raises:
            Exception: If DataFrame validation fails
        """
        target_col = self.target_config.name
        if not np.issubdtype(self.df[target_col].dtype, np.number):
            raise Exception(f"Target column '{target_col}' must be numeric")

    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocesses the data by performing several cleaning steps.

        Returns:
            pd.DataFrame: The cleaned DataFrame after preprocessing.

        Raises:
            Exception: If preprocessing fails
        """
        try:
            logger.info("Starting data preprocessing")
            self._rename_and_capitalize_columns()
            self._apply_value_corrections()
            self._validate_preprocessed_data()

            logger.info("Data cleaning completed successfully")
            logger.info(f"Final data shape: {self.df.shape}")
            logger.info(f"Final columns: {self.df.columns.tolist()}")
            logger.info(f"ID column data type: {self.df['Id'].dtype}")
            logger.info(f"Sample of cleaned data:\n{self.df.head().to_string()}")
            logger.info("Data cleaning script completed successfully")

            return self.df

        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

    def _rename_and_capitalize_columns(self) -> None:
        """Renames and capitalizes key columns."""
        self.df.rename(columns={self.target_config.name: self.target_config.new_name}, inplace=True)
        self.df.columns = [col.capitalize() if col else col for col in self.df.columns]
        self.df["Id"] = self.df["Id"].astype("str")
        logger.info("Renamed and capitalized columns")

    def _apply_value_corrections(self) -> None:
        """Corrects unknown values in specified columns."""
        logger.info("Applying value corrections for Education, Marriage, and Pay columns")
        corrections = getattr(
            self.config,
            "value_corrections",
            {
                "Education": {0: 4, 5: 4, 6: 4},
                "Marriage": {0: 3},
                "Pay": {-1: 0, -2: 0},
            },
        )
        for col_prefix, replacement_dict in corrections.items():
            columns = [col for col in self.df.columns if col.startswith(col_prefix)]
            for col in columns:
                self.df[col] = self.df[col].replace(replacement_dict)

    def _validate_preprocessed_data(self) -> None:
        """Validates the preprocessed data before returning."""
        if self.df.empty:
            raise Exception("Preprocessing resulted in an empty DataFrame")
        target_col = self.target_config.new_name
        if target_col not in self.df.columns:
            raise Exception(f"Target column '{target_col}' missing after preprocessing")
        if self.df.isnull().any().any():
            raise Exception("Unexpected null values found after preprocessing")


if __name__ == "__main__":
    # Set up logging
    setup_logging(CLEANING_LOGS)

    try:
        # Load configuration
        config = load_config("../../project_config.yml")  # Returns Config instance
        logger.info("Loaded configuration from ../../project_config.yml")

        # Create and run data cleaner
        data_cleaner = DataCleaning(FILEPATH, config)
        cleaned_data = data_cleaner.preprocess_data()

    #     # Log results
    #     logger.info("Data cleaning completed successfully")
    #     logger.info(f"Final data shape: {cleaned_data.shape}")
    #     logger.info(f"Final columns: {cleaned_data.columns.tolist()}")
    #     logger.info(f"ID column data type: {cleaned_data['Id'].dtype}")
    #     logger.info(f"Sample of cleaned data:\n{cleaned_data.head().to_string()}")

    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

    # logger.info("Data cleaning script completed successfully")
