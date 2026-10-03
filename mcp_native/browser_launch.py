"""Shared Chromium launch configuration for every Playwright launch in mcp_native.

Playwright's own bundled Chromium (chromium-1148 for playwright 1.49) does not run
under Docker Desktop's Linux VM on Apple-Silicon (arm64): the full binary crashes
on launch with SIGSEGV, and the newer "headless shell" build exits immediately —
so crawl_v2 / peek / the live pw_* session all come up empty and recon records no
requests. Debian's own ``chromium`` package is a *different* build that launches
cleanly on arm64 (and amd64), so the image installs it and we point Playwright at
it with ``executable_path``.

``APTS_CHROMIUM_PATH`` selects the binary. Set to ``/usr/bin/chromium`` in the
image, every launch uses the Debian build. Unset (any other host), the key is
omitted and Playwright falls back to its own bundled browser — so this is a no-op
wherever the bundled Chromium already works, and the behaviour stays generic.
"""
from __future__ import annotations

import os

from typing import Any

# --no-sandbox: Chromium cannot use its setuid sandbox as a non-root container
# user. --disable-dev-shm-usage: the container's tiny default /dev/shm otherwise
# crashes the renderer. Both are standard for headless Chromium in a container.
LAUNCH_ARGS = ["--no-sandbox", "--disable-dev-shm-usage"]


def launch_kwargs(**overrides: Any) -> dict[str, Any]:
    """kwargs for ``await pw.chromium.launch(**launch_kwargs())`` — headless, the
    standard container args, and the pinned system Chromium path when one is set.
    Extra keyword args override the defaults (e.g. a longer ``timeout``)."""
    kw: dict[str, Any] = {"headless": True, "args": list(LAUNCH_ARGS)}
    exe = os.environ.get("APTS_CHROMIUM_PATH")
    if exe:
        kw["executable_path"] = exe
    kw.update(overrides)
    return kw
