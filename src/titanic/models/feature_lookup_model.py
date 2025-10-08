"""FeatureLookUp model implementation."""

from datetime import datetime

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from titanic.config import ProjectConfig, Tags


class FeatureLookUpModel:
    """A class to manage FeatureLookupModel."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration."""
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters["xgb"]
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = (
            f"{self.catalog_name}.{self.schema_name}.titanic_features"
        )
        self.function_name = (
            f"{self.catalog_name}.{self.schema_name}.calculate_double_fare_over_age"
        )

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.dict()

    def create_feature_table(self) -> None:
        """Create or update the house_features table and populate it.

        This table stores features related to houses.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (PassengerId INT NOT NULL, Age INT, SibSp INT);
        """)
        self.spark.sql(
            f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT titanic_key PRIMARY KEY(PassengerId);"
        )
        self.spark.sql(
            f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT PassengerId, Age, SibSp FROM {self.catalog_name}.{self.schema_name}.train_set"
        )
        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT PassengerId, Age, SibSp FROM {self.catalog_name}.{self.schema_name}.test_set"
        )
        logger.info("✅ Feature table created and populated.")

    def define_feature_function(self) -> None:
        """Define a function to calculate the house's age.

        This function subtracts the year built from the current year.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(Fare DOUBLE, Age INT)
        RETURNS BOOLEAN
        RETURN Fare * 2 > Age
        """)
        logger.info("✅ Feature function defined.")

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.train_set = self.spark.table(
            f"{self.catalog_name}.{self.schema_name}.train_set"
        )
        self.test_set = self.spark.table(
            f"{self.catalog_name}.{self.schema_name}.test_set"
        ).toPandas()
        self.data_version = "0"  # describe history -> retrieve

        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]
        logger.info("✅ Data successfully loaded.")

    def feature_engineering(self) -> None:
        """Perform feature engineering by linking data with feature tables.

        Creates a training set using FeatureLookup and FeatureFunction.
        """
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=["PassengerId", "Age", "SibSp"],
                    lookup_key="PassengerId",
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="double_fare_over_age",
                    input_bindings={"Fare": "Fare", "Age": "Age"},
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()
        self.test_set["double_fare_over_age"] = (
            self.test_set["Fare"] * 2 > self.test_set["Age"]
        )

        self.X_train = self.training_df[
            self.num_features + self.cat_features + ["double_fare_over_age"]
        ]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[
            self.num_features + self.cat_features + ["double_fare_over_age"]
        ]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    def train(self) -> None:
        """Train the model and log results to MLflow.

        Uses a pipeline with preprocessing and LightGBM regressor.
        """
        logger.info("🚀 Starting training...")

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)
            ],
            remainder="passthrough",
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", XGBClassifier(**self.parameters)),
            ]
        )

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            # Training
            logger.info("🚀 Starting training...")
            pipeline.fit(self.X_train, self.y_train)

            # Testing
            y_pred = self.pipeline.predict(self.X_test)
            y_proba = self.pipeline.predict_proba(self.X_test)[:, 1]

            # Evaluate classification metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average="binary")
            recall = recall_score(self.y_test, y_pred, average="binary")
            f1 = f1_score(self.y_test, y_pred, average="binary")
            roc_auc = roc_auc_score(self.y_test, y_proba)

            logger.info(f"📊 Accuracy: {accuracy}")
            logger.info(f"📊 Precision: {precision}")
            logger.info(f"📊 Recall: {recall}")
            logger.info(f"📊 F1 Score: {f1}")
            logger.info(f"📊 ROC AUC: {roc_auc}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "xgbclassifier with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metrics(
                {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "roc_auc": roc_auc,
                }
            )

            # Log the model
            signature = infer_signature(model_input=self.X_train, model_output=y_pred)
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=self.data_version,
            )
            mlflow.log_input(dataset, context="training")
            mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                artifact_path="xgb-classifier-pipeline-model-fe",
                signature=signature,
            )

    def register_model(self) -> str:
        """Register the trained model to MLflow registry.

        Registers the model and sets alias to 'latest-model'.
        """
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/xgb-classifier-pipeline-model-fe",
            name=f"{self.catalog_name}.{self.schema_name}.titanic_model_fe",
            tags=self.tags,
        )

        # Fetch the latest version dynamically
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.titanic_model_fe",
            alias="latest-model",
            version=latest_version,
        )

        return latest_version

    def load_latest_model_and_predict(self, X: DataFrame) -> DataFrame:
        """Load the trained model from MLflow using Feature Engineering Client and make predictions.

        Loads the model with the alias 'latest-model' and scores the batch.
        :param X: DataFrame containing the input features.
        :return: DataFrame containing the predictions.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.titanic_model_fe@latest-model"

        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions

    def update_feature_table(self) -> None:
        """Update the titanic_features table with the latest records from train and test sets.

        Executes SQL queries to insert new records based on timestamp.
        """
        queries = [
            f"""
            WITH max_timestamp AS (
                SELECT MAX(update_timestamp_utc) AS max_update_timestamp
                FROM {self.config.catalog_name}.{self.config.schema_name}.train_set
            )
            INSERT INTO {self.feature_table_name}
            SELECT PassengerId, Age, SibSp
            FROM {self.config.catalog_name}.{self.config.schema_name}.train_set
            WHERE update_timestamp_utc >= (SELECT max_update_timestamp FROM max_timestamp)
            """,
            f"""
            WITH max_timestamp AS (
                SELECT MAX(update_timestamp_utc) AS max_update_timestamp
                FROM {self.config.catalog_name}.{self.config.schema_name}.test_set
            )
            INSERT INTO {self.feature_table_name}
            SELECT PassengerId, Age, SibSp
            FROM {self.config.catalog_name}.{self.config.schema_name}.test_set
            WHERE update_timestamp_utc >= (SELECT max_update_timestamp FROM max_timestamp)
            """,
        ]

        for query in queries:
            logger.info("Executing SQL update query...")
            self.spark.sql(query)
        logger.info("Titanic features table updated successfully.")
