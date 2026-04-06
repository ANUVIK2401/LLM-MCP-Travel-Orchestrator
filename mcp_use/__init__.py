"""
mcp_use - An MCP library for LLMs.

This library provides a unified interface for connecting different LLMs
to MCP tools through existing LangChain adapters.
"""

from importlib.metadata import PackageNotFoundError, version

from .agents.mcpagent import MCPAgent
from .client import MCPClient
from .config import load_config_file
from .connectors import BaseConnector, HttpConnector, StdioConnector, WebSocketConnector
from .logging import logger
from .session import MCPSession

try:
    __version__ = version("mcp-use")
except PackageNotFoundError:
    __version__ = "1.1.5"

__all__ = [
    "MCPAgent",
    "MCPClient",
    "MCPSession",
    "BaseConnector",
    "StdioConnector",
    "WebSocketConnector",
    "HttpConnector",
    "create_session_from_config",
    "load_config_file",
    "logger",
]
