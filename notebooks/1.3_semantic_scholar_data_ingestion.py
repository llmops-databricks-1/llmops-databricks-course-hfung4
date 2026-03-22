# Databricks notebook source
"""
Notebook 1.3 — OpenAlex Data Ingestion

Overview:
    This notebook fetches research paper metadata from the OpenAlex API
    and stores it as a Delta table in Unity Catalog for downstream processing.

Steps:
    1. Load environment configuration (catalog, schema, endpoints)
       from project_config.yml.
    2. Fetch paper metadata (title, authors, abstract, publication date, PDF URL, etc.)
       from the OpenAlex API using a configurable search query.
    3. Create a Spark DataFrame with a defined schema and write it to a Delta table
       in Unity Catalog ({catalog}.{schema}.open_alex_papers).
    4. Verify the ingested data by printing the schema, record count, and sample rows.
    5. Compute data statistics: paper counts by primary category and most recently
       published papers.

Output:
    Delta table: {catalog}.{schema}.open_alex_papers
"""

import random
from datetime import datetime

from loguru import logger
from pyalex import Works
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
from pyspark.sql.types import ArrayType, LongType, StringType, StructField, StructType

from open_alex_curator.config import get_env, load_config

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
TABLE_NAME = "open_alex_papers"

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
logger.info(f"Schema {CATALOG}.{SCHEMA} is ready.")

# COMMAND ----------


def fetch_openalex_papers(
    query: str = "artificial intelligence OR machine learning",
    max_results: int = 100,
) -> list[dict]:
    """Fetch paper metadata from the OpenAlex API.

    Args:
        query (str, optional): search query.
        max_results (int, optional): maximum number of papers to fetch (max 100).

    Returns:
        List of paper metadata dictionaries

    Example:
        >>> papers = fetch_openalex_papers(
        ...     query="large language models",
        ...     max_results=50,
        ... )
        >>> papers[0]["title"]
        'Attention Is All You Need'
        >>> papers[0]["authors"]
        ['Ashish Vaswani', 'Noam Shazeer']
    """
    # .get() returns a plain Python list
    works = Works().search(query).get(per_page=max_results)

    papers = []

    for work in works:
        # OpenAlex ID is a URL e.g. "https://openalex.org/W123456" — strip prefix
        paper_id = work["id"].replace("https://openalex.org/", "")

        # Abstract is stored as an inverted index {word: [positions]} — reconstruct it
        inverted_index = work.get("abstract_inverted_index") or {}
        if inverted_index:
            word_positions = [
                (pos, word)
                for word, positions in inverted_index.items()
                for pos in positions
            ]
            word_positions.sort()
            abstract = " ".join(word for _, word in word_positions)
        else:
            abstract = None

        # Extract author display names
        authors = [
            a["author"]["display_name"]
            for a in work.get("authorships", [])
            if (a.get("author") or {}).get("display_name")
        ]

        # publication_date is "YYYY-MM-DD"; convert to YYYYMMDDHHMM int
        pub_date_str = work.get("publication_date") or ""
        if pub_date_str:
            try:
                published = int(
                    datetime.strptime(pub_date_str, "%Y-%m-%d").strftime("%Y%m%d%H%M")
                )
            except Exception:
                published = None
        else:
            published = None

        # PDF URL from open_access metadata
        pdf_url = (work.get("open_access") or {}).get("oa_url")

        # Topics/field of study
        topics = [t["display_name"] for t in work.get("topics", [])]
        primary_topic = work.get("primary_topic")
        primary_category = (
            (primary_topic.get("field") or {}).get("display_name")
            if primary_topic
            else None
        )

        paper = {
            "paper_id": paper_id,
            "title": work.get("title"),
            "authors": authors,
            "summary": abstract,
            "published": published,
            "categories": ", ".join(topics),
            "pdf_url": pdf_url,
            "primary_category": primary_category,
            "ingestion_timestamp": datetime.now().isoformat(),
            "processed": None,  # populated downstream after processing
            "volume_path": None,  # populated downstream after writing to volume
        }

        papers.append(paper)

    return papers


# COMMAND ----------
# Fetch papers

logger.info("Fetching OpenAlex papers...")

query = (
    "behavioral science financial decision making "
    "insurance investment advisory consumer purchase behavior"
)
papers = fetch_openalex_papers(query=query, max_results=50)

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
