import os

import pytest
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from src.credit_default.data_preprocessing import DataPreprocessor
from src.credit_default.utils import load_config

# Load environment variables
load_dotenv()
FILEPATH = os.environ["FILEPATH"]


@pytest.fixture
def config():
    """Load configuration from the YAML file for testing."""
    return load_config("project_config.yml")


@pytest.fixture
def data_preprocessor(config):
    """Create an instance of DataPreprocessor for testing."""
    return DataPreprocessor(FILEPATH, config)


def test_data_preprocessor(data_preprocessor, config):
    """Test the DataPreprocessor class for preprocessing and scaling."""

    # Check that the cleaned data is as expected
    assert data_preprocessor.cleaned_data.equals(data_preprocessor.data_cleaning.df)

    # Check that features and target are correctly separated
    assert data_preprocessor.X.shape[0] == data_preprocessor.cleaned_data.shape[0]
    assert data_preprocessor.y.shape[0] == data_preprocessor.cleaned_data.shape[0]

    # Check that the correct features are used for scaling
    expected_features_robust = [feature for feature in config.features.robust]
    assert data_preprocessor.features_robust == expected_features_robust

    # Check that the preprocessor is of type ColumnTransformer
    assert isinstance(data_preprocessor.preprocessor, ColumnTransformer)

    # Check that the robust scaler is applied to the correct features
    transformers = data_preprocessor.preprocessor.transformers
    assert transformers[0][0] == "robust_scaler"
    assert isinstance(transformers[0][1], RobustScaler)
    assert set(transformers[0][2]) == set(data_preprocessor.features_robust)

    # Check if the output shape after scaling is as expected
    X_scaled, y, preprocessor = data_preprocessor.get_processed_data()
    assert X_scaled.shape[0] == data_preprocessor.X.shape[0]
    assert y.shape[0] == data_preprocessor.y.shape[0]
