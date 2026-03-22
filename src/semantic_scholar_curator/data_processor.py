"""
OpenAlex API
   ↓ (download_and_store_papers in Databricks Volume)
PDFs in Volume + open_alex_papers table (contains metadata)
   ↓ (parse_pdfs_with_ai)
ai_parsed_docs_table (the parsed document is stored in
a JSON-like string in the "parse_content" column of the table )
   ↓ (process_chunks)
open_alex_chunks_table (clean text for each chunk merged with metadata)
   ↓ (VectorSearchManager - separate class) (2.4 notebook)
Vector Search Index (embeddings)
"""

import json
import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from loguru import logger
from open_alex_curator.config import Config
from pyalex import Works
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import (
    col,
    concat_ws,
    current_timestamp,
    explode,
    udf,
)
from pyspark.sql.types import ArrayType, StringType, StructField, StructType


class DataProcessor:
    """
     DataProcessor handles the complete workflow of:
    - Downloading papers from OpenAlex
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
        self.papers_table = f"{self.catalog}.{self.schema}.open_alex_papers"
        # Delta table name for table containing AI-parsed PDF content
        self.parsed_table = f"{self.catalog}.{self.schema}.ai_parsed_docs_table"

    def _get_range_start(self) -> str:
        """
        Get the start time for the OpenAlex paper search range.

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

    @staticmethod
    def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
        """Reconstruct abstract text from OpenAlex inverted index format.

        OpenAlex stores abstracts as an inverted index: {word: [positions]}.
        This method reverses it back into a readable string.

        Args:
            inverted_index: dict mapping words to their position list, or None.

        Returns:
            Reconstructed abstract string, or None if not available.

        Example:
            >>> _reconstruct_abstract({"Hello": [0], "world": [1]})
            'Hello world'
        """
        if not inverted_index:
            return None
        word_positions = [
            (pos, word) for word, positions in inverted_index.items() for pos in positions
        ]
        word_positions.sort()
        return " ".join(word for _, word in word_positions)

    def download_and_store_papers(
        self,
    ) -> list[dict] | None:
        """
        Download papers from OpenAlex API and extract + persist metadata
        in the open_alex_papers table.

        Returns:
          List of metadata dicts for papers successfully downloaded in this run,
          or None if no papers were downloaded.
        """
        # Start of the time range for paper downloads
        start = self._get_range_start()

        # Convert YYYYMMDDHHMM timestamps to YYYY-MM-DD for OpenAlex API
        start_date = datetime.strptime(start, "%Y%m%d%H%M").strftime("%Y-%m-%d")
        end_date = datetime.strptime(self.end, "%Y%m%d%H%M").strftime("%Y-%m-%d")

        # Fetch open-access papers from OpenAlex in a single API call.
        # is_oa=True filters to papers that have a downloadable PDF.
        # per_page=max_results caps the result count (max 100 per OpenAlex docs).
        works = (
            Works()
            .search(self.cfg.project.query)
            .filter(
                from_publication_date=start_date,
                to_publication_date=end_date,
                is_oa=True,
            )
            # Tells openAlex the max number of papers to return in a API call
            # OpenAlex allows up to 200 papers per call
            .get(per_page=self.cfg.project.max_results)
        )

        logger.info(
            f"OpenAlex returned {len(works)} works for date range {start_date}:{end_date}"
        )

        # Download papers AND collect metadata
        records = []
        skipped_no_pdf = 0

        for work in works:
            # OpenAlex ID is a URL e.g. "https://openalex.org/W123456" — strip prefix
            paper_id = work["id"].replace("https://openalex.org/", "")
            pdf_url = (work.get("open_access") or {}).get("oa_url")

            if not pdf_url:
                skipped_no_pdf += 1
                continue

            try:
                response = requests.get(
                    pdf_url,
                    timeout=30,
                    stream=True,
                    headers={"user-agent": "requests/2.0.0"},
                )
                response.raise_for_status()
                with open(f"{self.pdf_dir}/{paper_id}.pdf", "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Extract author names
                authors = [
                    a["author"]["display_name"]
                    for a in work.get("authorships", [])
                    if (a.get("author") or {}).get("display_name")
                ]

                # publication_date is "YYYY-MM-DD"; convert to YYYYMMDDHHMM int
                pub_date_str = work.get("publication_date") or ""
                published = (
                    int(
                        datetime.strptime(pub_date_str, "%Y-%m-%d").strftime("%Y%m%d%H%M")
                    )
                    if pub_date_str
                    else None
                )

                records.append(
                    {
                        "open_alex_id": paper_id,
                        "title": work.get("title"),
                        "authors": authors,
                        "summary": self._reconstruct_abstract(
                            work.get("abstract_inverted_index")
                        ),
                        "pdf_url": pdf_url,
                        "published": published,
                        "processed": int(self.end),
                        "volume_path": f"{self.pdf_dir}/{paper_id}.pdf",
                    }
                )
            except Exception:
                logger.warning(f"Paper {paper_id} was not successfully processed.")

        logger.info(
            f"Skipped {skipped_no_pdf} works with no direct PDF URL. "
            f"Successfully downloaded {len(records)} papers."
        )

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
                T.StructField("open_alex_id", T.StringType(), False),
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
        metadata_df.write.format("delta").mode("ignore").saveAsTable(self.papers_table)

        # Merge (upsert) to avoid duplicates:
        # Insert to table if the record does not exist
        # in terms of open_alex_id.
        # Skip duplicates already in the table

        # Create a temp view so it can be referenced in the MERGE SQL statement
        metadata_df.createOrReplaceTempView("new_papers")

        self.spark.sql(
            f"""
        MERGE INTO {self.papers_table} target
        USING new_papers source
        ON target.open_alex_id = source.open_alex_id
        WHEN NOT MATCHED THEN INSERT(
            open_alex_id,
            title,
            authors,
            summary,
            pdf_url,
            published,
            processed,
            volume_path)
            VALUES (
            source.open_alex_id,
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
        # Strip the .pdf extension, split string on / and get last element
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

    def process_chunks(self) -> None:
        """Process parsed documents: extract chunks and paper id, and clean chunks
        Reads from ai_parsed_docs table (containing the "raw chunks" json-like strings)
        and writes to open_alex_chunks table
        """
        logger.info(
            f"Processing parsed documents from "
            f"{self.parsed_table} for end date {self.end}"
        )

        # Read the ai_parsed_docs table as a Spark Dataframe
        # Filter the newest records
        df = self.spark.table(self.parsed_table).where(f"processed = {self.end}")

        # Define schema for the extracted chunks
        # When we extract the chunks from the json-like strings,
        # the _extract_chunks() method returns a list of tuples: (chunk_id, content)
        chunk_schema = ArrayType(
            StructType(
                [
                    StructField("chunk_id", StringType(), True),
                    StructField("content", StringType(), True),
                ]
            )
        )

        # Define Spark UDFs — custom Python functions applied across worker nodes.
        # Similar to col() or concat_ws() but user-defined and slower to execute.
        extract_chunks_udf = udf(self._extract_chunks, chunk_schema)
        extract_paper_id_udf = udf(self._extract_paper_id, StringType())
        clean_chunk_udf = udf(self._clean_chunk, StringType())

        # Read metadata table (open_alex_papers)
        metadata_df = self.spark.table(self.papers_table).select(
            col("open_alex_id"),
            col("title"),
            col("summary"),
            # Combine the authors array column into a single string separated by comma
            concat_ws(", ", col("authors")).alias("authors"),
            # published is stored as YYYYMMDDHHMM integer (e.g. 202503181045).
            # Extract date parts using division and modulo arithmetic:
            #   year  = 202503181045 / 100000000 → 2025
            #   month = 202503181045 % 100000000 / 1000000 → 3
            #   day   = 202503181045 % 1000000   / 10000   → 18
            (col("published") / 100000000).cast("int").alias("year"),
            ((col("published") % 100000000) / 1000000).cast("int").alias("month"),
            ((col("published") % 1000000) / 10000).cast("int").alias("day"),
        )

        # Process the chunks data and create the transformed dataframe from parsed_table
        chunks_df = (
            # Extract paper id from the file path
            df.withColumn("open_alex_id", extract_paper_id_udf(col("path")))
            # Extract chunks from json-like strings in parsed_table
            # Result: array of {chunk_id, content} structs — one array per PDF
            .withColumn("chunks", extract_chunks_udf(col("parsed_content")))
            # Explode: turns one row per paper → many rows, one per text chunk
            .withColumn("chunk", explode(col("chunks")))
            .select(
                col("open_alex_id"),
                col("chunk.chunk_id").alias("chunk_id"),
                # Normalize text: fix hyphenation, collapse whitespace
                clean_chunk_udf(col("chunk.content")).alias("text"),
                # Build primary key by combining paper id and chunk id
                concat_ws("_", col("open_alex_id"), col("chunk.chunk_id")).alias("id"),
            )
            # Left join to enrich each chunk row with paper metadata
            # (title, authors, summary, year, month, day)
            .join(metadata_df, "open_alex_id", "left")
        )

        open_alex_processed_chunks_table = (
            f"{self.catalog}.{self.schema}.open_alex_chunks_table"
        )

        # Check before writing — once saveAsTable creates the table,
        # tableExists will return True and CDF would never be enabled
        is_first_run = not self.spark.catalog.tableExists(
            open_alex_processed_chunks_table
        )

        # Write the combined chunks and metadata df to a processed chunks table
        chunks_df.write.mode("append").saveAsTable(open_alex_processed_chunks_table)
        logger.info(f"Processed and saved chunks to {open_alex_processed_chunks_table}")

        # Enable Change Data Feed on first run only — allows the Vector Search index
        # to sync only new/changed chunks rather than reindexing everything
        if is_first_run:
            self.spark.sql(
                f"""
                ALTER TABLE {open_alex_processed_chunks_table}
                SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
                """
            )
            logger.info(
                f"Change Data Feed enabled for {open_alex_processed_chunks_table}"
            )

    def process_and_save(self) -> None:
        """Complete workflow: download papers, parse PDFs, and process chunks"""
        # Step 1: Download papers and create and store metadata
        records = self.download_and_store_papers()

        # Only continue if we have downloaded new papers
        if records is None:
            logger.info("No new papers to process. Exiting.")
            return

        # Step 2: Parse PDFs with ai_parse_document()
        self.parse_pdf_with_ai()
        logger.info("Parsed documents with AI.")

        # Step 3: Process chunks
        self.process_chunks()
        logger.info("Processing chunks completed!")
