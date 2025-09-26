# COMMAND ----------
import pandas as pd
import yaml
from loguru import logger
from pyspark.sql import SparkSession

from titanic.config import ProjectConfig
from titanic.data_processor import DataProcessor

spark = SparkSession.builder.getOrCreate()

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))
# COMMAND ----------
# Load Titanic dataset from volume
filepath = "../data/Titanic-Dataset.csv"

df = pd.read_csv(filepath)

# COMMAND ----------
# Preprocess data
data_processor = DataProcessor(df, config, spark)

data_processor.preprocess()

logger.info("Data processing completed")

# COMMAND ----------
X_train, X_test = data_processor.split_data(0.2)
logger.info(f"Training set shape: {X_train.shape}")
logger.info(f"Test set shape: {X_test.shape}")

data_processor.save_to_catalog(X_train, X_test)
logger.info("Data saved to catalog.")

# COMMAND ----------
data_processor.enable_change_data_feed()
logger.info("Successfully enabled change data feed.")
