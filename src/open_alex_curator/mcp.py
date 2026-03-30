"""MCP (Model Context Protocol) integration utilities.

This module provides helpers for connecting to Databricks-hosted MCP servers,
discovering the tools they expose, and wrapping those tools in a uniform
ToolInfo structure that agent code can consume directly.
"""

from collections.abc import Callable

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from pydantic import BaseModel


class ToolInfo(BaseModel):
    """Tool information for agent integration.

    Bundles everything an agent needs to advertise and invoke a single tool:
    the tool's name, its JSON schema (used to describe it to an LLM in the
    OpenAI Responses API format), and the callable that actually runs it.

    Attributes:
        name: Tool name — must match the name registered on the MCP server so
              the agent can route LLM tool-call requests to the right exec_fn.
        spec: JSON description of the tool in OpenAI Responses format, e.g.::

                  {
                      "type": "function",
                      "function": {
                          "name": "search_papers",
                          "description": "...",
                          "parameters": { ... }   # JSON Schema object
                      }
                  }

        exec_fn: Callable that accepts the tool's keyword arguments and returns
                 the tool's text output.  Created by create_managed_exec_fn so
                 it closes over the server URL and workspace credentials.
    """

    name: str
    spec: dict
    exec_fn: Callable

    class Config:
        # Callable is not a Pydantic-native type; allow it without validation.
        arbitrary_types_allowed = True


def create_managed_exec_fn(
    server_url: str, tool_name: str, w: WorkspaceClient
) -> Callable:
    """Create an execution function for a single MCP tool.

    Returns a closure — an inner function (exec_fn) that captures and remembers
    server_url, tool_name, and w from this outer scope, even after
    create_managed_exec_fn has returned.  Each tool gets its own exec_fn
    pre-loaded with the right connection details, so callers just invoke
    exec_fn(**kwargs) without knowing anything about the underlying MCP server.

    A new DatabricksMCPClient is created on every exec_fn call so the function
    is stateless and safe to reuse across concurrent agent turns.

    Args:
        server_url: Full URL of the MCP server (e.g. an Apps or Model Serving
                    endpoint on Databricks).
        tool_name:  Name of the specific tool to call on that server.
        w:          Authenticated Databricks WorkspaceClient used by
                    DatabricksMCPClient for credential propagation.

    Returns:
        A zero-dependency callable with signature ``(**kwargs) -> str`` that
        executes the tool and returns its concatenated text output.
    """

    def exec_fn(**kwargs: str) -> str:
        # Instantiate the MCP client inside the closure so each invocation
        # gets a fresh connection — avoids stale-session issues across turns.
        client = DatabricksMCPClient(server_url=server_url, workspace_client=w)

        # Call the tool; kwargs are forwarded as-is from the LLM's tool call.
        response = client.call_tool(tool_name, kwargs)

        # MCP responses carry a list of content blocks; join all text blocks
        # into a single string for easy consumption by the agent.
        return "".join([c.text for c in response.content])

    return exec_fn


async def create_mcp_tools(w: WorkspaceClient, url_list: list[str]) -> list[ToolInfo]:
    """Discover and wrap all tools exposed by a list of MCP servers.

    Iterates over each server URL, fetches the tool manifest via list_tools(),
    converts each tool's input schema into an OpenAI Responses-compatible spec,
    and pairs it with an execution function created by create_managed_exec_fn.

    Args:
        w:        Authenticated Databricks WorkspaceClient passed through to
                  both DatabricksMCPClient (for auth) and each exec_fn closure.
        url_list: Ordered list of MCP server URLs to query.  All tools from all
                  servers are collected into a single flat list; if the same
                  tool name appears on multiple servers the last one wins.

    Returns:
        Flat list of ToolInfo objects — one per tool across all servers —
        ready to be registered with an agent's tool dispatcher.
    """
    tools = []
    for server_url in url_list:
        # Connect to this server to retrieve its tool manifest.
        mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=w)

        # list_tools() is async; await to get the full list before iterating.
        mcp_tools = await mcp_client.list_tools()

        for mcp_tool in mcp_tools:
            # Defensive copy so we don't mutate the schema object returned by
            # the MCP client; fall back to an empty dict if no schema provided.
            input_schema = mcp_tool.inputSchema.copy() if mcp_tool.inputSchema else {}

            # Build the OpenAI Responses-format tool specification that the LLM
            # will use to decide when and how to call this tool.
            tool_spec = {
                "type": "function",
                "function": {
                    "name": mcp_tool.name,
                    "parameters": input_schema,
                    # Use the server-provided description when available; fall
                    # back to a minimal placeholder so the spec is always valid.
                    "description": mcp_tool.description or f"Tool: {mcp_tool.name}",
                },
            }

            # Bind server_url and tool_name into a standalone callable so the
            # agent doesn't need to manage connection details per-tool.
            exec_fn = create_managed_exec_fn(server_url, mcp_tool.name, w)

            tools.append(ToolInfo(name=mcp_tool.name, spec=tool_spec, exec_fn=exec_fn))

    return tools
