"""python -m mcp_servers.nova_service            # stdio (how the agent starts it)
python -m mcp_servers.nova_service --http     # Streamable HTTP on http://127.0.0.1:8765/mcp
"""

import argparse

from mcp_servers.nova_service.server import mcp

parser = argparse.ArgumentParser(description="Nova Robotik after-sales MCP server")
parser.add_argument("--http", action="store_true", help="serve Streamable HTTP instead of stdio")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8765)
args = parser.parse_args()

if args.http:
    mcp.run("streamable-http", host=args.host, port=args.port)
else:
    mcp.run()
