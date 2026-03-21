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
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from loguru import logger
from pyspark.sql import SparkSession
from semanticscholar import SemanticScholar

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

    def download_and_store_papers(
        self,
    ) -> list[dict] | None:
        """
        Download papers from semantic scholoar API and store metadata
        in semantic_scholar_papers table.
        NOTE: this is done already in the semantic_scholar_data_ingestion.py notebook
        but we will integrate the same code in the DataProcessor class.

        Returns:
          List of metadata dicts for papers successfully downloaded in this run,
          or None if no papers were downloaded.
        """
        # Start of the time range for paper downloads
        start = self._get_range_start()

        # Convert YYYYMMDDHHMM timestamps to YYYY-MM-DD for SemanticScholar API
        start_date = datetime.strptime(start, "%Y%m%d%H%M").strftime("%Y-%m-%d")
        end_date = datetime.strptime(self.end, "%Y%m%d%H%M").strftime("%Y-%m-%d")

        # Search for papers with the semantic scholar API
        client = SemanticScholar()
        query = (
            "behavioral science financial decision making "
            "insurance investment advisory consumer purchase behavior"
        )
        papers = client.search_paper(
            query,
            fields=[
                "paperId",
                "title",
                "authors",
                "abstract",
                "openAccessPdf",
                "publicationDate",
            ],
            publication_date_or_year=f"{start_date}:{end_date}",
        )

        # Download papers and collect metadata
        records = []

        for paper in papers:
            paper_id = paper.paperId
            if not paper.openAccessPdf:
                continue
            pdf_url = paper.openAccessPdf["url"]
            try:
                response = requests.get(pdf_url, timeout=30)
                response.raise_for_status()
                with open(f"{self.pdf_dir}/{paper_id}.pdf", "wb") as f:
                    f.write(response.content)
                # Collect metadata
                records.append(
                    {
                        "paper_id": paper_id,
                        "title": paper.title,
                        "authors": [author.name for author in paper.authors],
                        "summary": paper.abstract,
                        "pdf_url": pdf_url,
                        "published": int(paper.publicationDate.strftime("%Y%m%d%H%M")),
                        "processed": int(self.end),
                        "volume_path": f"{self.pdf_dir}/{paper_id}.pdf",
                    }
                )
            except Exception:
                logger.warning(f"Paper {paper_id} was not successfully processed.")
            # Avoid hitting API rate limits
            time.sleep(3)

        # Only process if we have records
        if len(records) == 0:
            logger.info("No new papers found.")
            return None

        logger.info(f"Downloaded {len(records)} papers to {self.pdf_dir}")
