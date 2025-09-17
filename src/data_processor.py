"""Data preprocessing module."""

import json

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from src.config import ProjectConfig


class DataProcessor:
    """A class for preprocessing and managing DataFrame operations.

    This class handles data preprocessing, splitting, and saving to Databricks tables.
    """

    def __init__(
        self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession
    ) -> None:
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.spark = spark

    def preprocess(self) -> None:
        """Preprocess the DataFrame stored in self.df.

        This method handles missing values, converts data types, and performs feature engineering.
        """
        processed = []

        for _, row in self.df.iterrows():
            ocr_text = row["OCRed Text"]
            json_data = row["Json Data"]

            # Extract only the "items" key as the label, empty if not present
            label = {"items": json_data.get("items", [])}

            processed.append(
                {
                    "input_text": ocr_text,
                    "target_text": json.dumps(label, ensure_ascii=False),
                }
            )
        self.df = pd.DataFrame(processed)
        self.df["input_text"] = self.df["input_text"].astype(str)
        self.df["target_text"] = self.df["target_text"].astype(str)

    def split_data(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.

        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        :return: A tuple containing the training and test DataFrames.
        """
        train_set, test_set = train_test_split(
            self.df, test_size=test_size, random_state=random_state
        )
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks tables.

        :param train_set: The training DataFrame to be saved.
        :param test_set: The test DataFrame to be saved.
        """
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
