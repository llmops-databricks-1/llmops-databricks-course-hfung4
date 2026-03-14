# Databricks notebook source
"""
Notebook 1.3 — Semantic Scholar Data Ingestion

Overview:
    This notebook fetches research paper metadata from the Semantic Scholar API
    and stores it as a Delta table in Unity Catalog for downstream processing.

Steps:
    1. Load environment configuration (catalog, schema, endpoints)
       from project_config.yml.
    2. Fetch paper metadata (title, authors, abstract, publication date, PDF URL, etc.)
       from the Semantic Scholar API using a configurable search query.
    3. Create a Spark DataFrame with a defined schema and write it to a Delta table
       in Unity Catalog ({catalog}.{schema}.semantic_scholar_papers).
    4. Verify the ingested data by printing the schema, record count, and sample rows.
    5. Compute data statistics: paper counts by primary category and most recently
       published papers.

Output:
    Delta table: {catalog}.{schema}.semantic_scholar_papers
"""

import random
from datetime import datetime

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
from pyspark.sql.types import ArrayType, LongType, StringType, StructField, StructType
from semanticscholar import SemanticScholar

from semantic_scholar_curator.config import get_env, load_config

# COMMAND ----------
# Spark session
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
# Get environment and load config
env = get_env(spark)  # from dbutils widget or default to dev if widget not available
env  # noqa: B018

# COMMAND ----------
cfg = load_config("../project_config.yml", env)
cfg.project  # noqa: B018

# COMMAND ----------
# Data management

CATALOG = cfg.project.catalog
SCHEMA = cfg.project.schema
TABLE_NAME = "semantic_scholar_papers"

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
logger.info(f"Schema {CATALOG}.{SCHEMA} is ready.")

# COMMAND ----------


def fetch_semantic_scholar_papers(
    query: str = "artificial intelligence OR machine learning",
    max_results: int = 100,
    batch_size: int = 100,
) -> list[dict]:
    """Fetch Semantic Scholar paper metadata from the Semantic Scholar API.

    Args:
        query (str, optional): search query.
        max_results (int, optional): maximum number of papers to fetch.
        batch_size (int, optional): page size per API request.

    Returns:
        List of paper metadata dictionaries

    Example:
        >>> papers = fetch_semantic_scholar_papers(
        ...     query="large language models",
        ...     max_results=50,
        ...     batch_size=25,
        ... )
        >>> papers[0]["title"]
        'Attention Is All You Need'
        >>> papers[0]["authors"]
        ['Ashish Vaswani', 'Noam Shazeer']
    """
    # Initialise the Semantic Scholar API client
    client = SemanticScholar()

    # Clamp batch_size so we never request more per page than the total we want
    batch_size = min(batch_size, max_results)

    # search_paper returns a lazy paginated iterator; batch_size controls page size
    results = client.search_paper(
        query=query,
        limit=batch_size,
        fields=[
            "paperId",
            "title",
            "authors",
            "abstract",
            "publicationDate",
            "year",
            "fieldsOfStudy",
            "openAccessPdf",
        ],
    )

    papers = []

    for count, result in enumerate(results):
        # Stop once we've collected the requested number of papers
        if count >= max_results:
            break

        publication_date = getattr(result, "publicationDate", None)
        year = getattr(result, "year", None)

        # Normalise date to a compact integer (YYYYMMDDHHmm) for sorting/filtering.
        # publicationDate is a datetime when present; fall back to year-only if not.
        if publication_date:
            try:
                published = int(publication_date.strftime("%Y%m%d%H%M"))
            except Exception:
                published = None
        else:
            # No full date — use Jan 1 00:00 of the publication year
            published = int(f"{year}01010000") if year else None

        # openAccessPdf is a dict with a "url" key, or None if no open-access PDF exists
        open_access_pdf = getattr(result, "openAccessPdf", None)
        pdf_url = open_access_pdf.get("url") if open_access_pdf else None

        # Extract author names, skipping entries where the name is missing
        authors = []
        if getattr(result, "authors", None):
            authors = [
                author.name for author in result.authors if getattr(author, "name", None)
            ]

        # Default to empty list so joins and index lookups below are safe
        fields_of_study = getattr(result, "fieldsOfStudy", None) or []

        paper = {
            "paper_id": getattr(result, "paperId", None),
            "title": getattr(result, "title", None),
            "authors": authors,
            "summary": getattr(result, "abstract", None),
            "published": published,
            "categories": ", ".join(fields_of_study),  # comma-separated string
            "pdf_url": pdf_url,
            "primary_category": fields_of_study[0] if fields_of_study else None,
            "ingestion_timestamp": datetime.now().isoformat(),
            "processed": None,  # populated downstream after processing
            "volume_path": None,  # populated downstream after writing to volume
        }

        papers.append(paper)

    return papers


# COMMAND ----------
# Fetch papers

logger.info("Fetching Semantic Scholar papers...")

query = (
    "behavioral science financial decision making "
    "insurance investment advisory consumer purchase behavior"
)
papers = fetch_semantic_scholar_papers(query=query, max_results=50)

logger.info(f"Fetched {len(papers)} papers")

# COMMAND ----------
sample_number = random.randint(0, len(papers) - 1)

logger.info(f"Sample paper (index {sample_number}):")
logger.info(f"Title: {papers[sample_number]['title']}")
logger.info(f"Authors: {papers[sample_number]['authors']}")
logger.info(f"Paper ID: {papers[sample_number]['paper_id']}")
logger.info(f"PDF URL: {papers[sample_number]['pdf_url']}")

# COMMAND ----------
# Create Delta Table in Unity Catalog that holds paper metadata.
# I will read this table later for further processing.

schema = StructType(
    [
        StructField("paper_id", StringType(), False),
        StructField("title", StringType(), False),
        StructField(
            "authors", ArrayType(StringType()), True
        ),  # Array to match reference code
        StructField("summary", StringType(), True),
        StructField("published", LongType(), True),  # Long to match reference code
        StructField("categories", StringType(), True),
        StructField("pdf_url", StringType(), True),
        StructField("primary_category", StringType(), True),
        StructField("ingestion_timestamp", StringType(), True),
        StructField("processed", LongType(), True),  # Long to match reference code
        StructField("volume_path", StringType(), True),  # Will be set in Lecture 2.2
    ]
)

# Create Spark DataFrame (NOTE: papers is a list of dict)
df = spark.createDataFrame(papers, schema=schema)

# COMMAND ----------
# Write to delta table
output_table_path = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

(
    df.write.format("delta")
    .mode("overwrite")
    .option("mergeSchema", "true")
    .saveAsTable(output_table_path)
)

logger.info(f"Created Delta Table: {output_table_path}")
logger.info(f"Records: {df.count()}")


# COMMAND ----------
# Verify the data

# Read the delta table
papers_df = spark.table(output_table_path)
logger.info(f"Table: {output_table_path}")
logger.info(f"Total papers: {papers_df.count()}")
logger.info("Table Schema:")
papers_df.printSchema()


# Sample records
logger.info("Sample records:")
papers_df.select("paper_id", "title", "primary_category", "published").show(
    5, truncate=50
)


# COMMAND ----------
# Data Statistics

logger.info("Papers by primary category:")
papers_df.groupBy("primary_category").count().orderBy("count", ascending=False).show()

logger.info("Most recent papers:")
papers_df.select("title", "published", "paper_id").orderBy(
    "published", ascending=False
).show(5, truncate=60)

# Papers with open-access PDFs available — useful for a research assistant
# that needs to retrieve and read full text
logger.info("Papers with open-access PDFs:")
papers_df.filter(papers_df.pdf_url.isNotNull()).select(
    "title", "pdf_url", "primary_category"
).show(5, truncate=60)

# Prolific authors — identifies researchers who publish frequently in this area
logger.info("Most prolific authors:")
papers_df.select(explode(col("authors")).alias("author")).groupBy(
    "author"
).count().orderBy("count", ascending=False).show(10, truncate=40)

# Papers missing abstracts — important for data quality since the abstract
# is used as the source text for embedding and retrieval
logger.info("Papers missing abstracts:")
papers_df.filter(col("summary").isNull()).select("paper_id", "title", "published").show(
    10, truncate=60
)

# COMMAND ----------
