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
from pyspark.sql import SparkSession

from open_alex_curator.config import load_config
from open_alex_curator.data_processor import DataProcessor

spark = SparkSession.builder.getOrCreate()

# Parse arguments passed by the Databricks job (see resources/1_data_processing_job.yml)
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="dev", help="Environment: dev, acc, or prd")
parser.add_argument(
    "--query",
    default="large language models machine learning",
    help="Search query for OpenAlex API",
)
parser.add_argument(
    "--max-results",
    type=int,
    default=20,
    help="Maximum number of papers to fetch (max 200)",
)
parser.add_argument(
    "--custom-start-date",
    default=None,
    help="Pipeline start date override (YYYYMMDDHHMM). Defaults to last run timestamp.",
)
parser.add_argument(
    "--custom-end-date",
    default=None,
    help="Pipeline end date override (YYYYMMDDHHMM). Defaults to now.",
)
args = parser.parse_args()

cfg = load_config("../project_config.yml", args.env)

logger.info("Configuration loaded:")
logger.info(f" Environment: {args.env}")
logger.info(f" Catalog: {cfg.project.catalog}")
logger.info(f" Schema: {cfg.project.schema}")
logger.info(f" Query: {args.query}")
logger.info(f" Max results: {args.max_results}")
logger.info(f" Custom start date: {args.custom_start_date}")
logger.info(f" Custom end date: {args.custom_end_date}")

# Download new papers; create metadata table; parse, chunk and process chunked papers
processor = DataProcessor(
    spark=spark,
    config=cfg,
    query=args.query,
    max_results=args.max_results,
    custom_start_date=args.custom_start_date,
    custom_end_date=args.custom_end_date,
)
processor.process_and_save()
logger.info("✓ Downloaded new papers, parsed with AI, and created chunks successfully!")
