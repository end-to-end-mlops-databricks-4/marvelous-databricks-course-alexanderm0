"""Unit tests for DataProcessor."""

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from titanic import PROJECT_DIR
from titanic.config import ProjectConfig
from titanic.data_processor import DataProcessor

# from databricks.connect import SparkSession


MLRUNS_DIR = PROJECT_DIR / "tests" / "mlruns"
CATALOG_DIR = PROJECT_DIR / "tests" / "catalog"
CATALOG_DIR.mkdir(parents=True, exist_ok=True)  # noqa


def test_data_ingestion(sample_data: pd.DataFrame) -> None:
    """Test the data ingestion process by checking the shape of the sample data.

    Asserts that the sample data has at least one row and one column.

    :param sample_data: The sample data to be tested
    """
    assert sample_data.shape[0] > 0
    assert sample_data.shape[1] > 0


def test_invalid_env_raises_value_error() -> None:
    """Tests whether ValueError is thrown in case of invalid environment."""
    with pytest.raises(ValueError, match="Invalid environment: test. Expected 'prd', 'acc', or 'dev'"):
        ProjectConfig.from_yaml(PROJECT_DIR / "project_config.yaml", env="test")


def test_dataprocessor_init(
    sample_data: pd.DataFrame,
    config: ProjectConfig,
    spark_session: SparkSession,
) -> None:
    """Test the initialization of DataProcessor.

    :param sample_data: Sample DataFrame for testing
    :param config: Configuration object for the project
    :param spark: SparkSession object
    """
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    assert isinstance(processor.df, pd.DataFrame)
    assert processor.df.equals(sample_data)

    assert isinstance(processor.config, ProjectConfig)
    # assert isinstance(processor.spark, SparkSession)


def test_preprocess(sample_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test whether preprocessing happens correctly."""
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess()
    df = processor.df
    print(df.dtypes)

    expected_columns = list(config.cat_features + config.num_features + [config.target, config.id_col])
    assert sorted(df.columns.tolist()) == sorted(expected_columns)

    for col in config.num_features:
        assert not df[col].isnull().any()


def test_split_data(sample_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test whether splitting of data happens correctly."""
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess()

    train_df, test_df = processor.split_data(test_size=0.3, random_state=123)

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    total_rows = len(processor.df)
    assert len(train_df) + len(test_df) == total_rows

    expected_test_size = int(total_rows * 0.3)
    assert abs(len(test_df) - expected_test_size) <= 1

    train_ids = set(train_df["PassengerId"])
    test_ids = set(test_df["PassengerId"])
    assert train_ids.isdisjoint(test_ids)


@pytest.mark.skip(reason="depends on delta tables on Databricks")
def test_save_to_catalog_succesfull(
    sample_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """Test the successful saving of data to the catalog.

    This function processes sample data, splits it into train and test sets, and saves them to the catalog.
    It then asserts that the saved tables exist in the catalog.

    :param sample_data: The sample data to be processed and saved
    :param config: Configuration object for the project
    :param spark: SparkSession object for interacting with Spark
    """
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess()
    train_set, test_set = processor.split_data()
    processor.save_to_catalog(train_set, test_set)
    processor.enable_change_data_feed()

    # Assert
    assert spark_session.catalog.tableExists(f"{config.catalog_name}.{config.schema_name}.train_set")
    assert spark_session.catalog.tableExists(f"{config.catalog_name}.{config.schema_name}.test_set")
