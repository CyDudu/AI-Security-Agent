"""Tool implementations for the native MCP server.

Each module here defines one tool (or a small related set) and exposes a
``register(mcp)`` function that attaches it to the server. ``server.py`` imports
every module listed in :data:`ALL` and calls its ``register`` — so adding a new
tool is: drop a module in here, give it a ``register(mcp)``, and add its name to
``ALL``. Nothing else changes.
"""
from __future__ import annotations

from mcp_native.tools import (
    arjun_tool,
    attack_tool,
    crawl_tool,
    crawl_v2_tool,
    ffuf_tool,
    hacktricks_tool,
    http_tool,
    katana_tool,
    playwright_live,
    recon_tool,
    sandbox_tool,
    sqlmap_tool,
    xss_prove,
)

# The modules server.py will register, in order. Add new tool modules here.
# The new-recon pipeline (RECON-MIGRATION.md) adds katana/ffuf/arjun/crawl_v2 and
# the playwright_live peek+primitives.
ALL = (http_tool, xss_prove, recon_tool, sqlmap_tool, hacktricks_tool,
       crawl_tool, attack_tool, sandbox_tool,
       katana_tool, ffuf_tool, arjun_tool, crawl_v2_tool, playwright_live)


def register_all(mcp) -> list[str]:
    """Register every tool module on ``mcp``. Returns the tool names registered."""
    names: list[str] = []
    for module in ALL:
        names.extend(module.register(mcp))
    return names
