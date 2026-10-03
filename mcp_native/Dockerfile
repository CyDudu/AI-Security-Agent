# Isolated image for the native MCP server (mcp_native/).
#
# Separate from the shared apts-skeleton image because the MCP SDK pulls a
# heavier, newer dependency set (pydantic>=2.12, starlette, uvicorn, httpx2)
# than the six agents pin. Keeping it here means the agents' image is untouched.
FROM python:3.11-slim-bookworm

RUN groupadd --system --gid 1000 apts \
 && useradd  --system --uid 1000 --gid apts --shell /usr/sbin/nologin apts

WORKDIR /app

COPY mcp_native/requirements.txt /app/mcp_native/requirements.txt
RUN pip install --no-cache-dir -r /app/mcp_native/requirements.txt

# The crawl tools drive headless Chromium via Playwright. Playwright's OWN bundled
# Chromium build (chromium-1148 for pw 1.49) does not run under Docker Desktop's
# Linux VM on Apple-Silicon (arm64): the full binary crashes on launch with SIGSEGV
# and the newer headless-shell exits immediately, so every crawl comes up empty.
# Debian's `chromium` package is a different build that launches cleanly on BOTH
# arm64 and amd64 — install it and point Playwright at it via APTS_CHROMIUM_PATH
# (read by mcp_native.browser_launch). `playwright install-deps` adds only the
# shared OS libraries Chromium needs, not the broken bundled binary — so there is
# no /ms-playwright browser directory any more.
RUN apt-get update \
 && apt-get install -y --no-install-recommends chromium \
 && playwright install-deps chromium \
 && rm -rf /var/lib/apt/lists/* \
 && chromium --version && echo "chromium OK"
ENV APTS_CHROMIUM_PATH=/usr/bin/chromium

# Recon tools, added AFTER the heavy Playwright layer so rebuilds don't re-pull
# Chromium: katana (crawl/endpoint enumeration) + ffuf (content/param fuzzing),
# both Go release binaries, and arjun (hidden-parameter discovery) via pip.
# Arch-aware: `dpkg --print-architecture` yields amd64|arm64, which matches the
# katana/ffuf release asset naming — so this builds on both x86_64 CI/prod and an
# Apple-Silicon (arm64) dev machine (amd64 Go binaries can't exec in an arm64 image).
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl unzip \
 && ARCH="$(dpkg --print-architecture)" \
 && curl -sSL "https://github.com/projectdiscovery/katana/releases/download/v1.1.0/katana_1.1.0_linux_${ARCH}.zip" -o /tmp/katana.zip \
 && unzip -o /tmp/katana.zip katana -d /usr/local/bin \
 && curl -sSL "https://github.com/ffuf/ffuf/releases/download/v2.1.0/ffuf_2.1.0_linux_${ARCH}.tar.gz" -o /tmp/ffuf.tgz \
 && tar -xzf /tmp/ffuf.tgz -C /usr/local/bin ffuf \
 && chmod +x /usr/local/bin/katana /usr/local/bin/ffuf \
 && pip install --no-cache-dir arjun \
 && rm -f /tmp/katana.zip /tmp/ffuf.tgz \
 && rm -rf /var/lib/apt/lists/* \
 && katana -version && ffuf -V && arjun --help >/dev/null 2>&1 && echo "recon tools OK"

# SecLists wordlists (curated subset) for ffuf. A sparse checkout keeps only the
# web-content + LFI/param lists we need, not the full ~1GB repo. The ffuf tool
# exposes these by name so the recon agent picks the right list per task.
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/danielmiessler/SecLists.git /opt/seclists \
 && git -C /opt/seclists sparse-checkout set \
      Discovery/Web-Content Fuzzing/LFI \
 && apt-get purge -y --auto-remove git && rm -rf /var/lib/apt/lists/* \
 && ls /opt/seclists/Discovery/Web-Content/common.txt && echo "seclists OK"

# shared.db (recon lookup) and the mcp_native package (which now carries its own
# browser crawler under mcp_native/crawler/ — no longer the retired mcp_tools).
COPY shared/     /app/shared/
COPY mcp_native/ /app/mcp_native/

RUN mkdir -p /etc/apts && chown -R apts:apts /app /etc/apts
USER apts

# No Chrome warm needed: both Playwright (mcp_native.browser_launch) and headless
# katana (katana_tool `-system-chrome-path`) use the system Chromium installed above
# (APTS_CHROMIUM_PATH=/usr/bin/chromium) — so there is no go-rod runtime download to
# pre-pay (and go-rod cannot download a Chrome for arm64 anyway).
CMD ["python", "-m", "mcp_native.server", "--http"]
