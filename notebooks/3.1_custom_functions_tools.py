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
from openai import OpenAI
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

# MAGIC %md
# MAGIC ## 7. Executing Tools

# COMMAND ----------

# Execute calculator tool
# In real life, the tool name and parameters will be decided by the agent
# Here, I am setting it manually to see if the tool (function) will be called using registry.execute

calc_result = registry.execute(
    name="calculator", args={"operation": "add", "a": 10, "b": 5}
)
logger.info(f"Calculator result: {calc_result}")

# Execute search tool
search_result = registry.execute(
    name="search_papers", args={"query": "neural networks", "num_results": 3}
)
logger.info(f"Calculator result: {search_result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Best Practices for Tool Design
# MAGIC
# MAGIC ### Do:
# MAGIC 1. **Clear descriptions**: Help the LLM understand when to use the tool
# MAGIC 2. **Type hints**: Use proper Python type hints
# MAGIC 3. **Error handling**: Handle errors gracefully
# MAGIC 4. **Return structured data**: JSON or clear text format
# MAGIC 6. **Validate inputs**: Check parameters before execution
# MAGIC 7. **Document parameters**: Clear parameter descriptions
# MAGIC
# MAGIC ### Don't:
# MAGIC 1. Create tools that are too complex
# MAGIC 2. Return unstructured or ambiguous data
# MAGIC 3. Forget error handling
# MAGIC 4. Make tools that take too long to execute
# MAGIC 5. Overlap tool functionality
# MAGIC 6. Use unclear tool names


# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Tool Design Patterns
# COMMAND ----------

# MAGIC %md
# MAGIC ### Pattern 1: Data Retrieval Tool
# MAGIC ```python
# MAGIC def get_data(query: str) -> str:
# MAGIC     # Fetch data from database/API
# MAGIC     # Format and return
# MAGIC     pass
# MAGIC ```
# MAGIC
# MAGIC ### Pattern 2: Computation Tool
# MAGIC ```python
# MAGIC def calculate(formula: str, values: dict) -> float:
# MAGIC     # Perform calculation
# MAGIC     # Return result
# MAGIC     pass
# MAGIC ```
# MAGIC
# MAGIC ### Pattern 3: Action Tool
# MAGIC ```python
# MAGIC def send_notification(message: str, recipient: str) -> str:
# MAGIC     # Perform action
# MAGIC     # Return confirmation
# MAGIC     pass
# MAGIC ```
# MAGIC
# MAGIC ### Pattern 4: Analysis Tool
# MAGIC ```python
# MAGIC def analyze_data(data: list, metric: str) -> dict:
# MAGIC     # Analyze data
# MAGIC     # Return insights
# MAGIC     pass
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Testing Tools

# COMMAND ----------


def test_tool(tool_name: str, test_cases: list[dict]):
    """
    Run a series of test cases against a registered tool and log the results.

    Iterates over each test case dict, executes the named tool via the global
    registry, and logs whether it succeeded or raised an exception. Results are
    truncated to 100 characters to keep the log output readable.

    Args:
        tool_name:  The name of the tool to test (must be registered in registry).
        test_cases: A list of argument dicts, each passed as **kwargs to the tool.
                    Example: [{"a": 1, "b": 2}, {"a": 10, "b": -3}]
    """
    logger.info(f"Testing tool: {tool_name}")
    logger.info("=" * 80)

    # enumerate starting at 1 so test case numbers are human-readable (1, 2, 3…)
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"Test Case {i}:")
        logger.info(f"  Input: {test_case}")  # fixed: was ". Input" (misaligned prefix)

        try:
            result = registry.execute(tool_name, test_case)
            logger.info(f"  ✓ Success")
            # Truncate long results so they don't flood the log
            logger.info(f"  Result: {str(result)[:100]}...")
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")


# COMMAND ----------

# Test the calculator tool
test_tool(
    "calculator",
    [
        {"operation": "add", "a": 5, "b": 3},
        {"operation": "multiply", "a": 4, "b": 7},
        {"operation": "divide", "a": 10, "b": 2},
    ],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Using Tools with an Agent
# MAGIC
# MAGIC Now let's create a simple agent that can call our tools.

# COMMAND ----------


class SimpleAgent:
    """A simple agent that can call tools in a loop."""

    def __init__(self, llm_endpoint: str, system_prompt: str, tools: list[ToolInfo]):
        # Store the Databricks serving endpoint name (used as the model ID in OpenAI-compatible calls)
        self.llm_endpoint = llm_endpoint
        # Store the system prompt that sets the agent's persona/behaviour for every conversation
        self.system_prompt = system_prompt
        # Build a name → ToolInfo lookup dict so tools can be dispatched by name in O(1)
        self._tools_dict = {tool.name: tool for tool in tools}
        # Use the OAuth bearer token the SDK has already negotiated, same
        # pattern as VectorSearchClient above — w.tokens.create() fails in
        # VS Code because it requires the PAT-creation entitlement.
        self._client = OpenAI(
            # Extract the raw token from the "Bearer <token>" Authorization header
            api_key=w.config.authenticate()["Authorization"].removeprefix("Bearer "),
            # Point the OpenAI client at Databricks serving endpoints instead of api.openai.com
            base_url=f"{w.config.host}/serving-endpoints",
        )

    def get_tool_specs(self) -> list[dict]:
        """Return the JSON schema specs for all registered tools, passed to the LLM so it knows what tools are available."""
        # Each ToolInfo.spec is the OpenAI-format function schema dict (name, description, parameters)
        return [tool.spec for tool in self._tools_dict.values()]

    def execute_tool(self, tool_name: str, args: dict) -> str:
        """Look up a tool by name and call it with the provided arguments."""
        # Guard against the LLM hallucinating a tool name that doesn't exist
        if tool_name not in self._tools_dict:
            raise ValueError(f"Unknown tool: {tool_name}")
        # Unpack args dict as keyword arguments so the call matches the tool function's signature
        return self._tools_dict[tool_name].exec_fn(**args)

    def chat(self, user_message: str, max_iterations: int = 10) -> str:
        """Run the agentic loop: send a message, handle tool calls, repeat until the LLM gives a final text answer."""
        # Initialise the conversation with the system prompt and the user's first message
        messages = [
            {"role": "system", "content": self.system_prompt},  # sets agent behaviour
            {"role": "user", "content": user_message},  # the user's question/request
        ]

        # Loop up to max_iterations times to prevent infinite tool-call chains
        for _ in range(max_iterations):
            # Send the full conversation history to the LLM and get its next response
            response = self._client.chat.completions.create(
                model=self.llm_endpoint,  # Databricks serving endpoint acting as model ID
                messages=messages,  # full conversation history so far
                # Pass tool schemas only when tools are registered; None disables tool use entirely
                tools=self.get_tool_specs() if self._tools_dict else None,
            )

            # Extract the assistant's message object (contains content and optional tool_calls)
            assistant_message = response.choices[0].message

            # Branch: did the LLM decide to call one or more tools?
            if assistant_message.tool_calls:
                # Append the assistant's turn to history — includes tool_calls so the LLM can
                # correlate tool results back to the requests it made
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content,  # may be None when only calling tools
                        # Serialize each tool call into the OpenAI wire format (id, type, function)
                        "tool_calls": [
                            {
                                "id": tc.id,  # unique ID to match result to request
                                "type": "function",  # always "function" for function-calling tools
                                "function": {
                                    "name": tc.function.name,  # which tool to call
                                    "arguments": tc.function.arguments,  # JSON string of arguments
                                },
                            }
                            for tc in assistant_message.tool_calls  # there may be multiple parallel tool calls
                        ],
                    }
                )

                # Execute every tool call the LLM requested in this turn
                for tool_call in assistant_message.tool_calls:
                    # Name of the tool the LLM wants to invoke
                    tool_name = tool_call.function.name
                    # Arguments arrive as a JSON string from the LLM; parse into a Python dict
                    tool_args = json.loads(tool_call.function.arguments)

                    logger.info(f"Calling tool: {tool_name} with args {tool_args}")

                    try:
                        # Dispatch to the registered tool function and capture its return value
                        result = self.execute_tool(tool_name, tool_args)
                    except Exception as e:
                        # Surface errors back to the LLM as a tool result so it can recover gracefully
                        result = f"Error: {str(e)}"

                    # Append the tool result to history; role="tool" + matching tool_call_id is required
                    # by the OpenAI API so the LLM can pair each result with the request that caused it
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,  # must match the id from the tool_calls list above
                            "content": str(result),  # tool output as a string
                        }
                    )
            else:
                # No tool calls — the LLM produced a final text answer, so exit the loop and return it
                return assistant_message.content

        # Safety net: if the loop exhausts all iterations without a final answer, return a fallback
        return "Max iterations reached."


# COMMAND ----------

# Let's instantiate the agent and test it out
agent = SimpleAgent(
    llm_endpoint=cfg.project.llm_endpoint,
    system_prompt="You are a helpful assistant. Use the available tools to answer questions.",
    tools=[calculator_tool, search_papers_tool],
)

logger.info("✓ Agent created with tools:")
for tool_name in agent._tools_dict.keys():
    logger.info(f"  - {tool_name}")

# COMMAND ----------
# Testing the agent: it should use calculator
logger.info("Testing agent with calculator:")
logger.info("=" * 80)

response = agent.chat("What is 42 multipleid by 17?")
logger.info(f"Agent response: {response}")

# COMMAND ----------
# Testing the agent: it should use the search tool
logger.info("Testing agent with search tool:")
logger.info("=" * 80)

response = agent.chat("Find papers about attention mechanisms.")
logger.info(f"Agent response: {response}")

# COMMAND ----------
