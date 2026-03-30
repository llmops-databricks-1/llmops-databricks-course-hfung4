# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.1: Custom Functions & Tools for Agents
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What are agent tools?
# MAGIC - Creating custom functions
# MAGIC - Tool specifications (OpenAI format)
# MAGIC - Integrating tools with agents
# MAGIC - Vector search as a tool

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understanding Agent Tools
# MAGIC
# MAGIC **Tools** are functions that agents can call to perform specific tasks.
# MAGIC
# MAGIC ### Why Tools?
# MAGIC
# MAGIC LLMs alone cannot:
# MAGIC - Access external data (databases, APIs)
# MAGIC - Perform calculations
# MAGIC - Execute code
# MAGIC - Search documents
# MAGIC
# MAGIC - Tools bridge this gap by giving LLMs the ability to take actions.
# MAGIC
# MAGIC ### Tool Calling Flow:
# MAGIC
# MAGIC ```
# MAGIC User: "What papers discuss transformers?"
# MAGIC   ↓
# MAGIC Agent: Decides to use vector_search tool
# MAGIC   ↓
# MAGIC Tool: vector_search(query="transformers")
# MAGIC   ↓
# MAGIC Tool Result: [paper1, paper2, paper3]
# MAGIC   ↓
# MAGIC Agent: Synthesizes answer from results
# MAGIC   ↓
# MAGIC Response: "Here are papers about transformers..."
# MAGIC ```

# COMMAND ----------
import json
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from loguru import logger
from pyspark.sql import SparkSession

from open_alex_curator.config import load_config, get_env

from open_alex_curator.mcp import ToolInfo

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

print(env)
print(cfg.project.vector_search_endpoint)

# COMMAND ----------
# Create Workspace Client for authentication and to get various workspace attributes
w = WorkspaceClient()

# COMMAND ----------
# Create VectorSearchClient using the token from the existing WorkspaceClient
# auth context (metadata-service / SPN).  Calling w.tokens.create() requires
# the 'tokens' user entitlement and fails for service principals — so we
# retrieve the bearer token that the SDK has already negotiated instead.
vsc = VectorSearchClient(
    workspace_url=w.config.host,
    personal_access_token=w.config.authenticate()["Authorization"].removeprefix("Bearer "),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Tool Specification Format
# MAGIC
# MAGIC Tools are defined using the **OpenAI function calling format**:
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "type": "function",
# MAGIC   "function": {
# MAGIC     "name": "tool_name",
# MAGIC     "description": "What the tool does",
# MAGIC     "parameters": {
# MAGIC       "type": "object",
# MAGIC       "properties": {
# MAGIC         "param1": {
# MAGIC           "type": "string",
# MAGIC           "description": "Description of param1"
# MAGIC         }
# MAGIC       },
# MAGIC       "required": ["param1"]
# MAGIC     }
# MAGIC   }
# MAGIC }
# MAGIC ```

# COMMAND ----------
