
from mcp.server.fastmcp import FastMCP
from langchain_community.utilities import SerpAPIWrapper
import os

mcp = FastMCP("Simple MCP Example")

# Register this function as a callable MCP tool.
@mcp.tool()
def multiply(a: float, b: float) -> int:
    """Multiply two float numbers"""
    return a * b

@mcp.tool()
def add(a: float, b: float) -> int:
    """Add two float numbers"""
    return a + b


@mcp.tool()
def google_search(query: str) -> str:
    """Apply Online Realtime Web Search"""
    serpapi = SerpAPIWrapper(serpapi_api_key="4f25e816def8a9de25e6b81f7a21ca8fd959a1cfeb19c12bf629f86caedc98e6")
    return serpapi.run(query)

if __name__ == "__main__":
    mcp.run(transport="stdio")
