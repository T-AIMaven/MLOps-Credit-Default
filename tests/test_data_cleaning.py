import os

import pandas as pd
import pytest
from dotenv import load_dotenv

from src.credit_default.data_cleaning import DataCleaning
from src.credit_default.utils import load_config

# Load environment variables
load_dotenv()
FILEPATH = os.environ["FILEPATH"]


@pytest.fixture
def real_data():
    """Load real data for testing."""
    return pd.read_csv(FILEPATH)


@pytest.fixture
def config():
    """Load configuration from the YAML file for testing."""
    return load_config("project_config.yml")


def test_load_data(real_data, config):
    """Test loading data from the real."""
    data_cleaning = DataCleaning(FILEPATH, config)
    assert data_cleaning.df.equals(real_data)


def test_preprocess_data(config):
    """Test the data preprocessing steps with real data."""
    data_cleaning = DataCleaning(FILEPATH, config)
    cleaned_df = data_cleaning.preprocess_data()

    # Check if the ID column is present and correctly cased
    assert "Id" in cleaned_df.columns

    # Check if the target column is renamed correctly
    assert "Default" in cleaned_df.columns
    assert "default.payment.next.month" not in cleaned_df.columns

    # Check if column names are capitalized
    assert all(col.capitalize() == col for col in cleaned_df.columns)

    # Check if unique values in Education are exactly {1, 2, 3, 4}
    assert set(cleaned_df["Education"].unique()) == {1, 2, 3, 4}

    # Check if unique values in Marriage are exactly {1, 2, 3}
    assert set(cleaned_df["Marriage"].unique()) == {1, 2, 3}

    # Check that PAY_0 values were replaced correctly
    assert set(cleaned_df["Pay_6"].unique()) == {0, 2, 3, 4, 5, 6, 7, 8}


def test_columns_after_preprocessing(real_data, config):
    """Test if columns match the expected columns in the config."""
    data_cleaning = DataCleaning(FILEPATH, config)
    data_cleaning.preprocess_data()

    expected_columns = [
        "Id",
        "Limit_bal",
        "Sex",
        "Education",
        "Marriage",
        "Age",
        "Pay_0",
        "Pay_2",
        "Pay_3",
        "Pay_4",
        "Pay_5",
        "Pay_6",
        "Bill_amt1",
        "Bill_amt2",
        "Bill_amt3",
        "Bill_amt4",
        "Bill_amt5",
        "Bill_amt6",
        "Pay_amt1",
        "Pay_amt2",
        "Pay_amt3",
        "Pay_amt4",
        "Pay_amt5",
        "Pay_amt6",
        "Default",
    ]

    # Check if the columns in the cleaned data match expected columns
    assert all(col in expected_columns for col in data_cleaning.df.columns)
