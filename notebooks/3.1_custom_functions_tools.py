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
# The VS Code Databricks extension injects metadata-service env vars and proxies
# your personal OAuth token through a local service.  w.tokens.create() fails
# because it calls a separate REST API requiring the PAT-creation entitlement.
# Instead, extract the OAuth bearer token the SDK has already negotiated.
vsc = VectorSearchClient(
    workspace_url=w.config.host,
    personal_access_token=w.config.authenticate()["Authorization"].removeprefix(
        "Bearer "
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Tool Specification Format
# MAGIC (value for the spec key in ToolInfo object) </br>
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

# MAGIC %md
# MAGIC ## 3. Creating a Simple Calculator Tool

# COMMAND ----------


def calculator(operation: str, a: float, b: float) -> float:
    """Perform basic arithmetic operations.
     Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First number
        b: Second number

    Returns:
        Result of the operation
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float("inf"),
    }

    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")

    return operations[operation](a, b)


# Test the function
result = calculator("multiply", 5, 3)
logger.info(f"5 * 3 = {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tool Specification for Calculator

# COMMAND ----------

calculator_tool_spec = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform basic arithmetic operations (add, subtract, multiply, divide)",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform",
                },
                "a": {"type": "number", "description": "The first number"},
                "b": {"type": "number", "description": "The second number"},
            },
            "required": ["operation", "a", "b"],
        },
    },
}

logger.info("Calculator Tool Specification:")
logger.info(json.dumps(calculator_tool_spec, indent=2))
# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Creating a Vector Search Tool (not from MCP, but code my own)


# COMMAND ----------
# Helper to parse vector search results
def parse_vector_search_results(results):
    """Parse vector search results from array format to dict format.

    Args:
        results: Raw results from similarity_search()

    Returns:
        List of dictionaries with column names as keys

    Example:
        >>> results = {
        ...     "manifest": {"columns": [{"name": "title"}, {"name": "abstract"}]},
        ...     "result": {"data_array": [["Attention is All You Need", "We propose..."],
        ...                               ["BERT", "We introduce BERT..."]]},
        ... }
        >>> parse_vector_search_results(results)
        [
            {"title": "Attention is All You Need", "abstract": "We propose..."},
            {"title": "BERT", "abstract": "We introduce BERT..."},
        ]
    """
    columns = [col["name"] for col in results.get("manifest", {}).get("columns", [])]
    data_array = results.get("result", {}).get("data_array", [])

    return [dict(zip(columns, row_data)) for row_data in data_array]


# COMMAND ----------
# Custom search_paper function (tool to be called by agent)
def search_papers(query: str, num_results: int = 5, year_filter: str = None) -> str:
    """Search for relevant papers using vector search.

    Args:
        query: Search query
        num_results: Number of results to return
        year_filter: Optional year filter (e.g., "2024")

    Returns:
        JSON string with search results
    """
    index_name = f"{cfg.project.catalog}.{cfg.project.schema}.open_alex_index"
    index = vsc.get_index(index_name=index_name)

    # Build search parameters
    search_params = {
        "query_text": query,
        # columns to return
        "columns": ["text", "title", "open_alex_id", "authors", "year"],
        "num_results": num_results,
        "query_type": "hybrid",
    }

    # Add year filter if provided
    if year_filter:
        search_params["filters"] = {"year": year_filter}

    # Perform search
    results = index.similarity_search(**search_params)

    # Format results using the helper function
    papers = []
    for row in parse_vector_search_results(results):
        papers.append(
            {
                "title": row.get("title", "N/A"),
                "open_alex_id": row.get("open_alex_id", "N/A"),
                "authors": str(row.get("authors", "N/A")),
                "year": row.get("year", "N/A"),
                "excerpt": row.get("text", "")[:200] + "...",
            }
        )
    return json.dumps(papers, indent=2)


# Test the function
results = search_papers("machine learning", num_results=2)
logger.info("Search Results:")
logger.info(results)


# COMMAND ----------
# MAGIC %md
# MAGIC ### Tool Specification for Vector Search
# COMMAND ----------

search_papers_tool_spec = {
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": "Search for academic papers using semantic search. Returns relevant papers with titles, authors, and excerpts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing what papers to find",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5,
                },
                "year_filter": {
                    "type": "string",
                    "description": "Optional year filter to limit results (e.g., '2024')",
                },
            },
            "required": ["query"],
        },
    },
}

logger.info("Search Papers Tool Specification:")
logger.info(json.dumps(search_papers_tool_spec, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Tool Information Class

# COMMAND ----------

# Using ToolInfo from open_alex_curator.mcp package
# This class represents a tool with name, spec, and execution function

# Create ToolInfo objects for the two tools I am creating: 1) calculator_tool and 2) search_papers_tool

# Calculator tool
calculator_tool = ToolInfo(
    name="calculator", spec=calculator_tool_spec, exec_fn=calculator
)

search_papers_tool = ToolInfo(
    name="search_papers", spec=search_papers_tool_spec, exec_fn=search_papers
)

logger.info("Available Tools:")
logger.info(f"1. {calculator_tool.name}")
logger.info(f"2. {search_papers_tool.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Tool Registry Pattern

# COMMAND ----------

# Create a Tool Registry Class that manage agent tools
# For example, register a tool to a dictionary of {tool name, ToolInfo object}
# get tool by name, get spec from all tools, execute a specific tool, list all tools available

# With MCP, the LLM discovers and calls tools through the protocol — your agent code
# doesn't need to manage a registry, pass specs manually, or dispatch calls. The registry pattern is essentially
# what MCP formalizes as a protocol.

# TL;DR: ToolRegistry is a lightweight DIY version of what MCP does.


from typing import Any


class ToolRegistry:
    """
    A central registry that manages all tools available to an LLM agent.

    Tools are stored in a dictionary keyed by tool name, allowing O(1) lookup.
    Each entry maps a tool name (str) to a ToolInfo object, which bundles the
    tool's JSON spec with its Python execution function.

    Typical usage in an agent loop:
      1. Instantiate the registry and register tools via register().
      2. Pass get_all_specs() to the LLM so the
         it knows what tools exist and how to call them.
      3. When the LLM returns a tool_use block, call execute() with the tool
         name and arguments extracted from that block.
    """

    def __init__(self):
        # Internal store: maps tool name -> ToolInfo (spec + execution function).
        # Private by convention (_tools) — callers should use the public methods.
        self._tools: dict[str, ToolInfo] = {}

    def register(self, tool: ToolInfo) -> None:
        """
        Add a tool to the registry.

        If a tool with the same name is already registered, it will be
        silently overwritten. This allows re-registration when a tool's
        spec or implementation is updated during development.

        Args:
            tool: A ToolInfo object containing the tool name, JSON spec,
                  and the Python callable that implements the tool.
        """
        self._tools[tool.name] = tool
        logger.info(f"✓ Registered tool: {tool.name}")

    def get_tool(self, name: str) -> ToolInfo:
        """
        Retrieve a single tool by its name.

        Args:
            name: The exact tool name used when the tool was registered
                  (must match the name in the tool's JSON spec).

        Returns:
            The ToolInfo object for the requested tool.

        Raises:
            ValueError: If no tool with the given name has been registered.
        """
        if name not in self._tools:
            raise ValueError(f"Tool not found: {name}")
        return self._tools[name]

    def get_all_specs(self) -> list[dict]:
        """
        Return the JSON specs for every registered tool.

        This list is passed directly to the LLM so
        it can understand the available tools and generate correctly
        structured tool_use blocks.

        Returns:
            A list of tool specification dicts, one per registered tool,
            in the format expected by the LLM.
        """
        return [tool.spec for tool in self._tools.values()]

    def execute(self, name: str, args: dict) -> Any:
        """
        Execute a registered tool with the provided arguments.

        Looks up the tool by name, then calls its exec_fn with the supplied
        keyword arguments. This is the method to call when the model returns
        a tool_use content block — pass the tool_use name and input dict here.

        Args:
            name: The name of the tool to execute (from the tool_use block).
            args: A dict of keyword arguments to pass to the tool function
                  (from the tool_use input field in the model's response).

        Returns:
            Whatever the tool's underlying Python function returns. The caller
            is responsible for serializing this into a tool_result message.

        Raises:
            ValueError: If the tool name is not found in the registry.
        """
        tool = self.get_tool(name)
        return tool.exec_fn(**args)

    def list_tools(self) -> list[str]:
        """
        Return the names of all registered tools.

        Useful for logging, debugging, or displaying available tools to a user
        without exposing full spec details.

        Returns:
            A list of tool name strings in registration order.
        """
        return list(self._tools.keys())

    def get_all_tools(self) -> list[ToolInfo]:
        """
        Return all registered ToolInfo objects as a list.

        Provides direct access to the full ToolInfo objects (name, spec, and
        exec_fn) when more than just the spec or name is needed, e.g. for
        inspection or testing.

        Returns:
            A list of ToolInfo objects in registration order.
        """
        return list(self._tools.values())


# Create registry and register tools
registry = ToolRegistry()
registry.register(calculator_tool)  # register the ToolInfo object
registry.register(search_papers_tool)

logger.info(f"Total tools registered: {len(registry.list_tools())}")
logger.info(f"Tools: {registry.list_tools()}")

# COMMAND ----------
