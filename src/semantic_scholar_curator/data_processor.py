"""
semantic_scholar API
   ↓ (download_and_store_papers in Databricks Volume)
PDFs in Volume + semantic_scholar_papers table (contains metadata)
   ↓ (parse_pdfs_with_ai)
ai_parsed_docs_table (the parsed document is stored in
a JSON-like string in the "parse_content" column of the table )
   ↓ (process_chunks)
semantic_scholar_chunks_table (clean text for each chunk merged with metadata)
   ↓ (VectorSearchManager - separate class) (2.4 notebook)
Vector Search Index (embeddings)
"""

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from loguru import logger
from pyspark.sql import SparkSession

from semantic_scholar_curator.config import Config


class DataProcessor:
    """
     DataProcessor handles the complete workflow of:
    - Downloading papers from semantic scholar
    - Storing paper metadata
    - Parsing PDFs with ai_parse_document
    - Extracting and cleaning text chunks
    - Saving chunks to Delta tables
    """

    def __init__(self, spark: SparkSession, config: Config):
        """
        Initialize DataProcessor with Spark session and configuration.

        Args:
            spark: SparkSession instance
            config: Config object with table configurations
        """
        self.spark = spark
        self.cfg = config
        self.catalog = config.project.catalog
        self.schema = config.project.schema
        self.volume = config.project.volume

        # Current timestamp used as a run identifier and upper bound for paper search
        # I download papers from the start time (last pipeline run) to current timestamp
        self.end = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d%H%M")
        # Databricks Volume path where PDFs for this run will be stored
        self.pdf_dir = f"/Volumes/{self.catalog}/{self.schema}/{self.volume}/{self.end}"
        # Create the PDF directory if it doesn't already exist
        os.makedirs(self.pdf_dir, exist_ok=True)
        # Delta table name for raw paper metadata
        self.papers_table = f"{self.catalog}.{self.schema}.semantic_scholar_papers"
        # Delta table name for table containing AI-parsed PDF content
        self.parsed_table = f"{self.catalog}.{self.schema}.ai_parsed_docs_table"

    def _get_range_start(self) -> str:
        """
        Get the start time for the Semantic Scholar paper search range.

        Uses max(processed) from the papers table if it exists (i.e. the timestamp of
        the most recent prior run), otherwise defaults to 3 days ago (first run).

        Returns:
            start string in "YYYYMMDDHHMM" format
        """
        # Neat way to check if metadata table exist in Unity Catalog
        # In Databricks, the spark.catalog is backed by the Unity Catalog
        if self.spark.catalog.tableExists(self.papers_table):
            # Fetch all rows from the table to the driver as a Python list of row objects
            # But since this is a single aggregted value, the output should be
            # [Row(max(processed)='202503181045')]
            result = self.spark.sql(
                f"""
                                    SELECT max(processed)
                                    FROM {self.papers_table}
                                    """
            ).collect()

            start = str(result[0][0])
            logger.info(
                f"Found existing papers metadata table. Start searching from: {start}"
            )
        else:
            # First run: metadata table does not exist, so search papers
            # between 3 days ago and current timestamp
            # So start = current timestamp - 3 days
            start = (
                datetime.now(ZoneInfo("America/New_York")) - timedelta(days=3)
            ).strftime("%Y%m%d%H%M")
        return start
