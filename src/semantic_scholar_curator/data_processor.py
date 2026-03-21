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

import json
import os
import re
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import (
    current_timestamp,
)
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
            # But since this is a single aggregated value, the output should be
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
        Download papers from Semantic Scholar API and extract + persist metadata
        in the semantic_scholar_papers table.
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

        # Download papers AND collect metadata
        records = []

        for paper in papers:
            # Skip papers that are not open access (no downloadable PDF)
            if not paper.openAccessPdf:
                continue
            paper_id = paper.paperId
            # Download paper to Volume with Request package
            pdf_url = paper.openAccessPdf["url"]
            try:
                response = requests.get(pdf_url, timeout=30)
                response.raise_for_status()
                with open(f"{self.pdf_dir}/{paper_id}.pdf", "wb") as f:
                    f.write(response.content)
                # Collect metadata
                records.append(
                    {
                        "semantic_scholar_id": paper_id,
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

        # Only upsert records to metadata table if we have new records
        if len(records) == 0:
            logger.info("No new papers found.")
            return None

        logger.info(f"Downloaded {len(records)} papers to {self.pdf_dir}")

        # From the list of dictionaries of metadata (records), create a Spark Dataframe
        # and persist it as a Delta Table

        # Define schema of the metadata table
        schema = T.StructType(
            [
                T.StructField("semantic_scholar_id", T.StringType(), False),
                T.StructField("title", T.StringType(), True),
                T.StructField("authors", T.ArrayType(T.StringType()), True),
                T.StructField("summary", T.StringType(), True),
                T.StructField("pdf_url", T.StringType(), True),
                T.StructField("published", T.LongType(), True),
                T.StructField("processed", T.LongType(), True),
                T.StructField("volume_path", T.StringType(), True),
            ]
        )

        # Create Spark DataFrame from list of dictionaries (records)
        metadata_df = self.spark.createDataFrame(records, schema=schema).withColumn(
            "ingest_ts", current_timestamp()
        )

        # Create Delta table if it doesn't exist, do nothing if table already exists
        metadata_df.write.format("delta").mode("ignore").saveAsTable(self.paper_table)

        # Merge (upsert) to avoid duplicates:
        # Insert to table if the record does not exist
        # in terms of semantic_scholar_id.
        # Skip duplicates already in the table

        # Create a temp view so it can be referenced in the MERGE SQL statement
        metadata_df.createOrReplaceTempView("new_papers")

        self.spark.sql(
            f"""
        MERGE INTO {self.papers_table} target
        USING new_papers source
        ON target.semantic_scholar_id = source.semantic_scholar_id
        WHEN NOT MATCHED THEN INSERT(
            semantic_scholar_id,
            title,
            authors,
            summary,
            pdf_url,
            published,
            processed,
            volume_path)
            VALUES (
            source.semantic_scholar_id,
            source.title,
            source.authors,
            source.summary,
            source.pdf_url,
            source.published,
            source.processed,
            source.volume_path
            )
            """
        )
        logger.info(f"Merged {len(records)} paper records into {self.papers_table}")
        return records

    def parse_pdf_with_ai(self) -> None:
        """Parse PDFs using ai_parse_document and store in ai_parsed_docs table."""

        # Create ai_parsed_docs table if it doesn't exist
        self.spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {self.parsed_table} (
                path STRING,
                parsed_content STRING,
                processed LONG
            )
            """
        )

        # Parse raw PDFs with ai_parse_document() from AgentBricks.
        # Populates parsed_content with a JSON-like string of the parsed document.
        # NOTE: ai_parse_document() requires Databricks workspace context —
        # it cannot be run outside of a Databricks workspace or job.
        self.spark.sql(
            f"""
            INSERT INTO {self.parsed_table}
            SELECT
                path,
                ai_parse_document(content) AS parsed_content,
                {self.end} AS processed
            FROM READ_FILES(
                "{self.pdf_dir}",
                format => 'binaryFile'
            )
            """
        )
        logger.info(f"Parsed PDFs from {self.pdf_dir} and saved to {self.parsed_table}")

        @staticmethod
        def _extract_chunks(parsed_content_json: str) -> list[tuple[str, str]]:
            """Extract text chunks from the parsed_content JSON string.

            Args:
                parsed_content_json: JSON string from ai_parse_document output.

            Returns:
                List of (chunk_id, content) tuples for text-type elements only.

            Example:
                >>> json_str = '{"document": {"elements": [
                ...     {"id": "c1", "type": "text", "content": "Hello"},
                ...     {"id": "c2", "type": "image", "content": "fig.png"}
                ... ]}}'
                >>> _extract_chunks(json_str)
                [("c1", "Hello")]  # image skipped
            """
            # Deserialize the JSON-like string into a Python dict
            parsed_dict = json.loads(parsed_content_json)
            chunks = []

            # Navigate to the "elements" list inside the parsed document structure
            # Only extract elements of type "text" (skip images, tables, etc.)
            for element in parsed_dict.get("document", {}).get("elements", []):
                if element.get("type") == "text":
                    chunk_id = element.get("id", "")
                    content = element.get("content", "")
                    chunks.append((chunk_id, content))
            return chunks

    @staticmethod
    def _extract_paper_id(path: str) -> str:
        """Extract paper ID from file path.

        NOTE: Volume path format: f"{self.pdf_dir}/{paper_id}.pdf"

        Args:
            path: File path ending in "<paper_id>.pdf"

        Returns:
            Paper ID extracted from the filename.

        Example:
            >>> _extract_paper_id(
            ...     "/Volumes/catalog/schema/vol/202603211045/abc123.pdf"
            ... )
            'abc123'
        """
        # strip the .pdf extension, splits stirng on / and get last element
        return path.replace(".pdf", "").split("/")[-1]

    @staticmethod
    def _clean_chunk(text: str) -> str:
        """Clean and normalize chunk text.

        Args:
            text (str): Raw text content

        Returns:
            str: Cleaned text content
        """
        # Fix hyphenation across line breaks:
        # "doc-\nments" -> "documents"
        cleaned_text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

        # Collapse internal newlines into spaces
        cleaned_text = re.sub(r"\s*\n\s*", " ", cleaned_text)

        # Collapse repeated whitespace
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

        return cleaned_text.strip()
