# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.5: Simple RAG with Vector Search
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What is RAG (Retrieval-Augmented Generation)?
# MAGIC - Using Vector Search for document retrieval
# MAGIC - Enriching prompts with retrieved context
# MAGIC - Building a simple Q&A system
# MAGIC
# MAGIC **RAG Flow:**
# MAGIC ```
# MAGIC User Question
# MAGIC     ↓
# MAGIC Vector Search (retrieve relevant documents)
# MAGIC     ↓
# MAGIC Build Prompt (question + context)
# MAGIC     ↓
# MAGIC LLM (generate answer)
# MAGIC     ↓
# MAGIC Response
# MAGIC ```
# COMMAND ----------
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from openai import OpenAI
from loguru import logger

from open_alex_curator.config import load_config, get_env

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup
# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

cfg.project.llm_endpoint
# COMMAND ----------
# Create clients: Databricks worksapce client, OpenAI client,
# and Vector Search client

# Databricks Workspace Client
w = WorkspaceClient()

# OpenAI client for Databricks
client = OpenAI(
    api_key=w.config.authenticate()["Authorization"].removeprefix("Bearer "),
    base_url=f"{w.config.host}/serving-endpoints",
)

# Vector Search client
vsc = VectorSearchClient(
    workspace_url=w.config.host,
    personal_access_token=w.config.authenticate()["Authorization"].removeprefix(
        "Bearer "
    ),
)

logger.info(f"✓ Connected to workspace: {w.config.host}")
logger.info(f"✓ Using LLM endpoint: {cfg.project.llm_endpoint}")
# COMMAND ----------


# MAGIC %md
# MAGIC ## 2. Vector Search Retrieval
# MAGIC
# MAGIC First, let's create a function to retrieve relevant documents from our vector search index.
# MAGIC Note that this retrevial step is done OUTSIDE of the LLM (not decided by LLM),
# MAGIC hence, this is an augmented LLM workflow and NOT an agentic workflow
# COMMAND ----------
def retrieve_documents(query: str, num_results: int = 5) -> list[dict]:
    """Retrieve relevant documents from vector search.

    Args:
        query: The search query
        num_results: Number of documents to retrieve

    Returns:
        List of document dictionaries with title, text, and metadata
    """
    # Build the fully-qualified index name: <catalog>.<schema>.<index> — matches how it was created
    index_name = f"{cfg.project.catalog}.{cfg.project.schema}.open_alex_index"
    # Get a handle to the vector search index (does not fetch data yet)
    index = vsc.get_index(index_name=index_name)

    # Run a hybrid search (combines dense vector similarity + keyword BM25) against the index
    results = index.similarity_search(
        query_text=query,  # the user's natural-language query; the SDK embeds this automatically
        # specify which columns to return in results — order here determines row[i] positions below
        columns=["text", "title", "open_alex_id", "authors", "year"],
        num_results=num_results,  # how many top-k results to return
        query_type="hybrid",  # "hybrid" = dense + sparse; use "ann" for pure vector search
    )

    # Accumulate parsed documents into a list to return
    documents = []
    # Guard: results can be None or missing "result" key if the index is empty or the call failed
    if results and "result" in results:
        # "data_array" is a list of rows; each row is a list of values in the same order as `columns`
        data_array = results["result"].get("data_array", [])
        # Convert each raw row (list) into a named dict for easier downstream access
        for row in data_array:
            documents.append(
                {
                    "text": row[
                        0
                    ],  # full text of the paper chunk (used as RAG context)
                    "title": row[1],  # paper title
                    "open_alex_id": row[2],  # unique OpenAlex identifier for the paper
                    "authors": row[3],  # author list
                    "year": row[4],  # publication year
                }
            )
    # Return the list of document dicts (empty list if no results found)
    return documents
