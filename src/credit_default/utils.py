import sys
from typing import Any, Dict, List, Literal, Optional

import yaml
from loguru import logger
from pydantic import BaseModel, Field, ValidationError


class NumFeature(BaseModel):
    name: str
    dtype: Literal["float64", "int64"]


class Target(BaseModel):
    name: str
    dtype: Literal["float64", "int64"]
    new_name: str


class Features(BaseModel):
    clean: List[str]
    robust: List[str]


class Config(BaseModel):
    catalog_name: str
    schema_name: str
    pipeline_id: str
    parameters: Dict[str, Any] = Field(description="Parameters for model training")
    ab_test: Dict[str, Any] = Field(description="Parameters for A/B testing")
    num_features: List[NumFeature]
    target: List[Target]
    features: Features


def setup_logging(log_file: Optional[str] = "", log_level: str = "DEBUG") -> None:
    """
    Sets up logging configuration with optional file logging.

    Args:
        log_file (str, optional): Path to the log file. Defaults to None.
        log_level (str, optional): Logging level to use. Defaults to "DEBUG".
    """

    # Remove the default logger
    logger.remove()

    # Add file logger with rotation if log_file is provided
    if log_file != "":
        logger.add(log_file, level=log_level, rotation="500 MB")

    # Add stdout logger
    logger.add(
        sys.stdout,
        level=log_level,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )


def load_config(config_path: str) -> Config:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        config = Config(**config_data)  # Pydantic validation
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file: {str(e)}")
        raise
    except ValidationError as e:
        logger.error(f"Validation error in configuration: {e}")
        raise
