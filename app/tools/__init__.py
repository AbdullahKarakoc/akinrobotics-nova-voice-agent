from __future__ import annotations

from app.config import Settings
from app.tools import database, mcp_tools, web_apis
from app.tools.registry import Tool, ToolRegistry, ToolResult


def create_tool_registry(settings: Settings) -> ToolRegistry:
    """Wire all tools. A new data source is one more `register(...)` call here,
    or one more entry in MCP_SERVERS when it is served over MCP."""
    registry = ToolRegistry()
    database.ensure_database(settings.db_path, settings.data_dir / "seed.sql")
    database.register(registry, settings.db_path)
    web_apis.register(registry, timeout_s=settings.http_tool_timeout_s, timezone=settings.timezone)
    mcp_tools.register(registry, settings.mcp_servers, timeout_s=settings.mcp_connect_timeout_s)
    return registry


__all__ = ["Tool", "ToolRegistry", "ToolResult", "create_tool_registry"]
