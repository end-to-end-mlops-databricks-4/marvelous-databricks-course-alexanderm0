# Databricks notebook source
import os

import mlflow
from dotenv import load_dotenv
from pyspark.sql import SparkSession

from titanic.config import ProjectConfig, Tags
from titanic.models.basic_model import BasicModel
from titanic.utils import is_databricks

# COMMAND ----------
if not is_databricks():
    load_dotenv()
    profile = os.environ.get("PROFILE", "DEFAULT")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})
