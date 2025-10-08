# Databricks notebook source
import os

import mlflow
from dotenv import load_dotenv
from pyspark.sql import SparkSession

from titanic.config import ProjectConfig, Tags
from titanic.models.feature_lookup_model import FeatureLookUpModel

# Configure tracking uri
load_dotenv()
profile = os.environ.get("PROFILE", "DEFAULT")
mlflow.set_tracking_uri(f"databricks://{profile}")
mlflow.set_registry_uri(f"databricks-uc://{profile}")

spark = SparkSession.builder.getOrCreate()
# spark = DatabricksSession.builder.remote(cluster_id="0911-113128-1v3eeo04").getOrCreate()
tags_dict = {"git_sha": "abcd12345", "branch": "week3"}
tags = Tags(**tags_dict)

config = ProjectConfig.from_yaml(config_path="../project_config.yml")


# COMMAND ----------

# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# COMMAND ----------

# Create feature table
fe_model.create_feature_table()

# COMMAND ----------

# Define excessive costs feature function
fe_model.define_feature_function()

# COMMAND ----------

# Load data
fe_model.load_data()

# COMMAND ----------

# Perform feature engineering
fe_model.feature_engineering()

# COMMAND ----------

# Train the model
fe_model.train()

# COMMAND ----------

# Train the model
fe_model.register_model()
