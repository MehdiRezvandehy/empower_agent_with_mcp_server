

from mcp.server.fastmcp import FastMCP
import os

mcp = FastMCP("local-filesystem")

BASE_DIR = os.path.abspath("./mcp_workspace")

os.makedirs(BASE_DIR, exist_ok=True)


@mcp.tool()
# async means this function can run cooperatively with other tasks without blocking the program.
async def create_file(path: str, content: str = ""):
    full_path = os.path.join(BASE_DIR, path)

    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)

    return {"status": "created", "path": full_path}


@mcp.tool()
async def write_file(path: str, content: str):
    full_path = os.path.join(BASE_DIR, path)

    with open(full_path, "a", encoding="utf-8") as f:
        f.write(content)

    return {"status": "written", "path": full_path}

@mcp.tool()
async def create_xy_file(n: int = 100):
    lines = ["x y"]
    for i in range(1, n+1):
        lines.append(f"{i} {n-i+1}")
    return "\n".join(lines)
    
if __name__ == "__main__":
    mcp.run(transport="stdio")
