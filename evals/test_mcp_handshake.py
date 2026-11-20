import asyncio
import os
import sys
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters


def extract_tool_output(call_result):
    """Safely extract output from MCP v2 CallToolResult."""
    if getattr(call_result, "structured_content", None) is not None:
        return call_result.structured_content
    if call_result.content:
        return [getattr(c, "text", str(c)) for c in call_result.content]
    return []


async def main() -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server.research_tools"],
        env=os.environ.copy(),
    )

    async with Client(server_params) as client:
        # 1. Discover all tools exposed by your MCP v2 server
        tools_response = await client.list_tools()
        tool_names = [tool.name for tool in tools_response.tools]
        print("MCP Tools Discovered:", tool_names)

        # 2. Test Qdrant Long-Term Memory Recall over MCP
        mem_res = await client.call_tool(
            "recall_long_term_memory",
            {"topic": "How do agents connect to tools?"},
        )
        print("MCP Memory Recall Result:", extract_tool_output(mem_res))

        # 3. Test Live Web Search over MCP
        search_res = await client.call_tool(
            "search_web",
            {"query": "Python FastAPI framework", "max_results": 2},
        )
        print("MCP Web Search Result:", extract_tool_output(search_res))


if __name__ == "__main__":
    asyncio.run(main())
