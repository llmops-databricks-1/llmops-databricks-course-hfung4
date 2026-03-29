import argparse

from loguru import logger
from pyspark.sql import SparkSession

from open_alex_curator.config import load_config
from open_alex_curator.vector_search import VectorSearchManager

spark = SparkSession.builder.getOrCreate()

# Parse --env argument passed by the Databricks job (see resources/process_data.yml)
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="dev", help="Environment: dev, acc, or prd")
args = parser.parse_args()
env = args.env
# Get config
cfg = load_config("../project_config.yml", env)

logger.info("Configuration loaded:")
logger.info(f" Environment: {env}")
logger.info(f" Catalog: {cfg.project.catalog}")
logger.info(f" Schema: {cfg.project.schema}")


# Sync Vector Search Index
vs_manager = VectorSearchManager(config=cfg)
vs_manager.sync_index()

logger.info("✓ Created and sync vector search index successfully!")
