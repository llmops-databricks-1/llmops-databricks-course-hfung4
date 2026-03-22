"""
This script downloads, parses, and chunks papers from OpenAlex
and syncs the vector search index.

Pipeline steps:
1. Download new PDFs from OpenAlex
2. Parse PDFs with AI Parse Documents
3. Extract and clean chunks
4. Sync embedded chunks to vector search index
"""

import argparse

from loguru import logger
from open_alex_curator.config import load_config
from open_alex_curator.data_processor import DataProcessor
from pyspark.sql import SparkSession

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

# Step 1: Process new papers
processor = DataProcessor(spark=spark, config=cfg)
processor.process_and_save()
