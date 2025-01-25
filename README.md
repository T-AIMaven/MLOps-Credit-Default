## MLOps Credit Default ✈️

<p align="center">
<img width="737" alt="cover" src="https://github.com/user-attachments/assets/a1c18fba-9e39-45b5-8fcd-bceb1f5f5af9">
</p>

This is a personal MLOps project based on a [Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset/data) dataset for credit default predictions.

It was developed as part of the this [End-to-end MLOps with Databricks](https://maven.com/marvelousmlops/mlops-with-databricks) course.

Feel free to ⭐ and clone this repo 😉

## Tech Stack

![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=Databricks&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Spark](https://img.shields.io/badge/Apache_Spark-FFFFFF?style=for-the-badge&logo=apachespark&logoColor=#E35A16)
![MLflow](https://img.shields.io/badge/MLflow-0194E2.svg?style=for-the-badge&logo=MLflow&logoColor=white)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

## Project Structure

The project has been structured with the following folders and files:

- `.github/workflows`: CI/CD configuration files
  - `cd.yml`
  - `ci.yml`
- `data`: raw data
  - `data.csv`
- `notebooks`: notebooks for various stages of the project
  - `create_source_data`: notebook for generating synthetic data
    - `create_source_data_notebook.py`
  - `feature_engineering`: feature engineering and MLflow experiments
    - `basic_mlflow_experiment_notebook.py`
    - `combined_mlflow_experiment_notebook.py`
    - `custom_mlflow_experiment_notebook.py`
    - `prepare_data_notebook.py`
  - `model_feature_serving`: notebooks for serving models and features
    - `AB_test_model_serving_notebbok.py`
    - `feature_serving_notebook.py`
    - `model_serving_feat_lookup_notebook.py`
    - `model_serving_notebook.py`
  - `monitoring`: monitoring and alerts setup
    - `create_alert.py`
    - `create_inference_data.py`
    - `lakehouse_monitoring.py`
    - `send_request_to_endpoint.py`
- `src`: source code for the project
  - `credit_default`
    - `data_cleaning.py`
    - `data_cleaning_spark.py`
    - `data_preprocessing.py`
    - `data_preprocessing_spark.py`
    - `utils.py`
- `tests`: unit tests for the project
  - `test_data_cleaning.py`
  - `test_data_preprocessor.py`
- `workflows`: workflows for Databricks asset bundle
  - `deploy_model.py`
  - `evaluate_model.py`
  - `preprocess.py`
  - `refresh_monitor.py`
  - `train_model.py`
- `.pre-commit-config.yaml`: configuration for pre-commit hooks
- `Makefile`: helper commands for installing requirements, formatting, testing, linting, and cleaning
- `project_config.yml`: configuration settings for the project
- `databricks.yml`: Databricks asset bundle configuration
- `bundle_monitoring.yml`: monitoring settings for Databricks asset bundle

## Project Set Up

The Python version used for this project is Python 3.11.

1. Clone the repo:

   ```bash
   git clone https://github.com/satel33/mlops-databricks-credit-default.git
   ```

2. Create the virtual environment using `uv` with Python version 3.11 and install the requirements:

   ```bash
    uv venv -p 3.11.0 .venv
    source .venv/bin/activate
    uv pip install -r pyproject.toml --all-extras
    uv lock
    ```

3. Build the wheel package:

    ```bash
    # Build
    uv build
    ```

4. Install the Databricks extension for VS Code and Databricks CLI:

   ```bash
   curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
   ```

5. Authenticate on Databricks:

   ```bash
   # Authentication
   databricks auth login --configure-cluster --host <workspace-url>

   # Profiles
   databricks auth profiles
   cat ~/.databrickscfg
   ```

After entering your information, the CLI will prompt you to save it under a Databricks configuration profile `~/.databrickscfg`


## Catalog Set Up

Once the project is set up, you need to create the volumes to store the data and the wheel package that will you have to install in the cluster:

- **catalog name**: *credit*
- **schema_name**: *default*
- **volume name**: *data* and *packages*

  ```bash
  # Create volumes
  databricks volumes create credit default data MANAGED
  databricks volumes create credit default packages MANAGED

  # Push volumes
  databricks fs cp data/data.csv dbfs:/Volumes/credit/default/data/data.csv
  databricks fs cp dist/credit_default_databricks-0.0.1-py3-none-any.whl dbfs:/Volumes/credit/default/packages

  # Show volumes
  databricks fs ls dbfs:/Volumes/credit/default/data
  databricks fs ls dbfs:/Volumes/credit/default/packages
  ```

## Token Creation

Some project files require a Databricks authentication token. This token allows secure access to Databricks resources and APIs:

1. Create a token in the Databricks UI:

   - Navigate to `Settings` --> `User` --> `Developer` --> `Access tokens`

   - Generate a new personal access token

2. Create a secret scope for securely storing the token:

    ```bash
    # Create Scope
    databricks secrets create-scope secret-scope

    # Add secret after running command
    databricks secrets put-secret secret-scope databricks-token

    # List secrets
    databricks secrets list-secrets secret-scope
    ```
