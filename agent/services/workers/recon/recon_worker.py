"""Reconnaissance — an agentic, phased, LangGraph-driven crawl (RECON-MIGRATION.md).

Ported from the langgraph-branch PoC onto our ChatOpenAI + Langfuse stack. Phases:

1. **Phase 0 — deep katana sweep (code).** Headless katana (JS crawl + jsluice + xhr +
   known-files) builds the breadth URI/API map. Fail-loud: errors or 0 endpoints raise
   ``WorkFailed`` (no blind run).
2. **Phase 0.5 — peek (code).** Render each route, record per-page input TYPES + on-load
   XHRs — the behaviour map.
3. **Phase 1 — agent loop (StateGraph).** The model drives crawl_v2 / ffuf / arjun to
   CAPTURE real requests and discover unlinked/hidden surface (``recon_graph``), one
   decision at a time with a de-dup guard.
4. **Phase 2 — assemble + grade (code owns the raw surface).** Merge every crawl into the
   AssetMap byte-intact, then ONE model call writes a summary + grades findings (additive).

Modes (``broad``/``focused``) are a one-line preamble on every prompt (loop + grade).
"""
from __future__ import annotations

import json
import logging
import os
import re
from urllib.parse import parse_qsl, urlparse

from mcp import Client

from shared import prompts
from shared.llm import build_chat_model, flush, run_config, set_run_output, trace_run
from shared.models import (
    AssetMap, Endpoint, Form, FormField, ObservedRequest, Technology,
)

from agent.services.workers.common import common_agentic as agentic
from agent.services.workers.base import WorkContext, WorkFailed
from .recon_graph import build_recon_graph, normalize_url

log = logging.getLogger("agent.workers.recon")

MAX_STEPS = int(os.environ.get("APTS_RECON_MAX_STEPS", "20"))
MAX_TOKENS = int(os.environ.get("APTS_RECON_MAX_TOKENS", "1200"))
# Phase-0.4 SPA seed: one CODE-driven crawl_v2 of the base that ALWAYS runs before
# the agent loop, so the SPA is rendered, its hash routes discovered and its forms
# submitted every run — independent of whether the Phase-1 model calls crawl_v2.
# (Ported from origin/main, whose recon is reliable precisely because of this seed.)
SPA_SEED_MAX_PAGES = int(os.environ.get("APTS_RECON_SPA_SEED_PAGES", "25"))
SPA_SEED_MAX_DEPTH = int(os.environ.get("APTS_RECON_SPA_SEED_DEPTH", "3"))
DEFAULT_MODE = (os.environ.get("APTS_RECON_MODE", "broad") or "broad").lower()


def _mode_preamble(mode: str) -> str:
    """One-line mode steer, prepended to EVERY recon prompt (loop + grading)."""
    if mode == "focused":
        return ("MODE: FOCUSED — precision over coverage. Go only for fruitful "
                "surface: authentication, user input, APIs, admin/config, anything "
                "taking a parameter or a body. Skip purely static/marketing pages. "
                "Keep and grade only genuinely test-worthy things.\n\n")
    return ("MODE: BROAD — coverage over precision. Do not miss surface: when "
            "unsure whether a route is interesting, crawl it, and keep the maybes. "
            "Grade anything that might be worth testing, even if marginal.\n\n")


RECON_LOOP_SYSTEM = """\
You are a reconnaissance agent for an authorised penetration test. Your job is to
CAPTURE the security-relevant attack surface — the real HTTP requests and hidden
surface the vuln/exploit stages will test. You do NOT exploit; you discover and map.

A breadth sweep (katana) and a cheap behaviour peek have already run — you are GIVEN
the routes, the concrete API/file endpoints, and, per interactive page, the input
TYPES it exposes and the APIs it calls on load. Act on that map with three tools:

- crawl_v2(url): browser that fills a page's inputs by TYPE and submits (Enter/click),
  recording the REAL request the form fires. Use it on INTERACTIVE pages (the peek
  shows which have inputs) so the submit (e.g. a login/search/complaint POST) is
  captured. This is the point of recon — one route per call; the guard refuses repeats.
- ffuf(url, wordlist): fuzz for UNLINKED paths and enumerable IDs a crawler can NEVER
  see. Put FUZZ in the url. wordlist by purpose: 'common'/'big'/'raft-dirs'/'raft-files'
  = dirs/files, 'api' = API routes, 'ids' = numeric ID enumeration (IDOR:
  url="<base>/<resource>/FUZZ" wordlist="ids"), 'lfi' = path-traversal, 'params'.
- arjun(url): discover HIDDEN parameters an endpoint accepts — concrete injection points.

How to work:
- Start with the interactive pages the peek flagged (forms, user input, auth, upload):
  crawl_v2 each so its real request is captured.
- Then go BEYOND the UI: ffuf for unlinked dirs/IDs (fuzz an id on any /resource/<id>
  endpoint), ffuf under any /api or /rest prefix, arjun on real endpoints for hidden params.
- Read each OBSERVATION and act on it. Never repeat an identical call. Stay on host.
- TRANSPORT NOISE IS NOT SURFACE: WebSocket/Socket.IO handshakes and long-polling
  (e.g. /socket.io/?EIO=…&transport=polling), heartbeats/keepalives, and analytics
  beacons are plumbing, not attack surface — recognise them early and dismiss them.
  Never spend a tool call on them and never treat their parameters as interesting,
  even though they carry params. Focus your calls on real app/API input instead.
- STOP when the worthwhile surface is captured: reply WITHOUT a tool call, "DONE — <why>".
"""

RECON_GRADE_SYSTEM = """\
You are a reconnaissance analyst. You are given the mapped attack surface (endpoints,
forms, parameterised requests, API paths, interactive pages with their input types,
technologies). Summarise it and produce GROUPED, per-finding gradings for the
vuln/exploit stages.

Reply with ONLY JSON, no prose:
{"summary":"<2-4 sentences on the security-relevant surface>",
 "findings":[{"finding":"<what it is>","where":"<path or parameter>",
              "category":"<authentication|injection|access-control|file-upload|user-input|info-disclosure|api|other>",
              "grade":"high|medium|low","reason":"<one short line>"}]}
One object PER finding (do not merge several into one). Group by assigning the right
`category`. Grade by test-worthiness: high = authentication / user-input / injection /
access-control / upload surface; medium = app or API surface with less obvious risk;
low = minor. Only REAL findings grounded ONLY in the surface given; invent nothing.
IGNORE TRANSPORT NOISE: WebSocket/Socket.IO handshakes and long-polling
(/socket.io/?EIO=…&transport=polling), heartbeats/keepalives and analytics beacons are
plumbing, not attack surface — do NOT emit findings for them even though they carry
parameters. They are never test-worthy.
"""

RECON_LOOP_PROMPT_ID = "recon.loop"
RECON_GRADE_PROMPT_ID = "recon.grade"
prompts.register(RECON_LOOP_PROMPT_ID, RECON_LOOP_SYSTEM)
prompts.register(RECON_GRADE_PROMPT_ID, RECON_GRADE_SYSTEM)


class ReconWorker:
    """Agentic recon: katana sweep, peek, agent-driven crawl loop, then grade."""

    async def __call__(self, ctx: WorkContext) -> dict:
        mode = (getattr(ctx, "recon_mode", "") or DEFAULT_MODE).lower()
        if mode not in ("broad", "focused"):
            mode = "broad"
        pre = agentic.directive_block(ctx.directive)
        modepre = _mode_preamble(mode)
        creds = ctx.credentials or {}
        domains = list(ctx.scope.domains)
        config = run_config(engagement_id=str(ctx.engagement_id), agent="recon")
        log.info("recon starting in %s mode against %s", mode, ctx.target_url)

        with trace_run(name="recon", run_input=f"recon {mode} {ctx.target_url}",
                       engagement_id=str(ctx.engagement_id)) as _stage:
            try:
                async with Client(agentic.MCP_URL) as client:
                    # ---- Phase 0: deep katana sweep (code) → breadth URI map ----
                    try:
                        kat = (await client.call_tool("katana", {
                            "url": ctx.target_url, "depth": 2, "js_crawl": True,
                            "jsluice": True, "xhr": True, "headless": True,
                            "known_files": True, "timeout_sec": 300})).structured_content or {}
                    except Exception as exc:  # noqa: BLE001
                        raise WorkFailed(
                            f"recon phase0: katana (headless deep sweep) failed to run "
                            f"({type(exc).__name__}: {exc}). This is the breadth map the "
                            f"stage depends on — aborting recon.") from exc
                    kat_endpoints = [] if kat.get("error") else (kat.get("endpoints") or [])
                    if not kat_endpoints:
                        # Headless katana can't start Chrome in some environments — on
                        # Apple-Silicon (arm64) go-rod's Chromium segfaults / exec-format-
                        # errors. Fall back to a STATIC sweep: jsluice mines the SPA's
                        # routes straight from the JS bundle with no browser, and known-
                        # files adds /ftp etc. Phase 1's crawl_v2 (Playwright) renders the
                        # SPA for depth, so coverage holds without headless katana.
                        log.warning("recon phase0: headless katana yielded nothing "
                                    "(%s) — retrying STATIC (no headless)",
                                    kat.get("error") or "0 endpoints")
                        try:
                            kat = (await client.call_tool("katana", {
                                "url": ctx.target_url, "depth": 2, "js_crawl": True,
                                "jsluice": True, "xhr": False, "headless": False,
                                "known_files": True, "timeout_sec": 200})).structured_content or {}
                        except Exception as exc:  # noqa: BLE001
                            raise WorkFailed(
                                f"recon phase0: static katana fallback failed to run "
                                f"({type(exc).__name__}: {exc}).") from exc
                        kat_endpoints = [] if kat.get("error") else (kat.get("endpoints") or [])
                    if not kat_endpoints:
                        raise WorkFailed(
                            "recon phase0: katana returned 0 endpoints (headless AND "
                            "static) — target unreachable or not crawlable; aborting "
                            "rather than running blind.")
                    routes, api_file_eps = self._split_endpoints(kat_endpoints, ctx.target_url)
                    log.info("recon phase0: katana %d endpoint(s) -> %d route(s), %d api/file",
                             len(kat_endpoints), len(routes), len(api_file_eps))
                    self._dump("phase0_katana", {"endpoints": kat_endpoints,
                               "count": len(kat_endpoints), "routes": routes,
                               "api_file_endpoints": api_file_eps})

                    # ---- Phase 0.4: SPA seed (code) — one real-browser crawl of the
                    # base that ALWAYS runs. katana does NOT mine client-side SPA routes
                    # (hash routes like #/login live only in the JS router), so without
                    # this the peek and the agent never see them and menu-gated forms
                    # (login/register) are never submitted. crawl_v2 drives the live UI,
                    # discovers those routes and fills+submits the forms it reaches —
                    # so recon renders the app and records its traffic EVERY run, not
                    # only when the Phase-1 model happens to call crawl_v2. Generic: no
                    # route names; grading judges what is kept. (Ported from main.)
                    seed_crawls: list[dict] = []
                    try:
                        seed = (await client.call_tool("crawl_v2", {
                            "url": ctx.target_url,
                            "max_pages": SPA_SEED_MAX_PAGES,
                            "max_depth": SPA_SEED_MAX_DEPTH,
                            "allow_domains": domains,
                            **self._cred_args(creds)})).structured_content or {}
                        seed_crawls = [seed]
                        spa_routes = [p["url"] for p in (seed.get("pages") or [])
                                      if "#/" in (p.get("url") or "")]
                        # SPA routes FIRST so peek (routes[:60]) always covers them.
                        seen_r: set = set(spa_routes)
                        routes = spa_routes + [r for r in routes
                                               if not (r in seen_r or seen_r.add(r))]
                        log.info("recon phase0.4: SPA seed %d page(s), +%d hash route(s), "
                                 "%d request(s) captured",
                                 seed.get("pages_crawled", 0), len(spa_routes),
                                 len(seed.get("observed_requests", []) or []))
                    except Exception as exc:  # noqa: BLE001 — seed is best-effort
                        log.warning("recon phase0.4 SPA seed failed: %s", exc)
                    self._dump("phase04_spa_seed", {"routes": routes,
                               "seed_requests": (len(seed_crawls[0].get("observed_requests", []) or [])
                                                 if seed_crawls else 0)})

                    # ---- Phase 0.5: peek (code) → behaviour map of the routes ----
                    peek_pages = []
                    try:
                        pk = (await client.call_tool(
                            "peek", {"urls": routes[:60]})).structured_content or {}
                        peek_pages = pk.get("pages", []) or []
                    except Exception as exc:  # noqa: BLE001
                        log.warning("recon phase0.5 peek failed: %s", exc)
                    log.info("recon phase0.5: peeked %d route(s)", len(peek_pages))
                    self._dump("phase05_peek", {"pages": peek_pages})

                    # ---- Phase 1: agent loop — crawl_v2, ffuf, arjun ----
                    descriptors = [t for t in (await client.list_tools()).tools
                                   if t.name in ("crawl_v2", "ffuf", "arjun")]
                    schemas = [{"type": "function", "function": {
                        "name": d.name,
                        "description": (getattr(d, "description", "") or "")[:1024],
                        "parameters": agentic._schema_of(d)}} for d in descriptors]
                    model = build_chat_model("recon", max_tokens=MAX_TOKENS).bind_tools(schemas)

                    async def call_tool(name, args):
                        out = await client.call_tool(name, args)
                        return out.structured_content or {}

                    loop_prompt = prompts.get(RECON_LOOP_PROMPT_ID)
                    graph = build_recon_graph(model, call_tool, max_steps=MAX_STEPS,
                                              budget_chars=agentic.CONTEXT_BUDGET_CHARS)
                    task = self._phase1_task(ctx.target_url, routes, api_file_eps,
                                             peek_pages, mode, pre)
                    out = await graph.ainvoke({
                        "mode": mode, "target": ctx.target_url, "domains": domains,
                        "creds": creds,
                        "messages": [agentic.SystemMessage(content=modepre + loop_prompt.text),
                                     agentic.HumanMessage(content=task)],
                        "tried": {}, "crawled": [], "crawls": [], "enum": [], "step": 0,
                    }, config={**config, "recursion_limit": 2 * MAX_STEPS + 5})

                    # Prepend the Phase-0.4 seed so its captured requests (e.g. the
                    # login POST and search XHR it fired) ALWAYS merge into the surface
                    # alongside the agent's crawls — recon records traffic every run,
                    # not only when the Phase-1 model chooses to call crawl_v2.
                    crawls = seed_crawls + (out.get("crawls") or [])
                    self._dump("phase1_agent", {
                        "tool_calls": out.get("tried", {}), "enum": out.get("enum", []),
                        "crawled_pages": [p.get("url") for c in crawls
                                          for p in c.get("pages", [])],
                        "captured_requests": [{"method": r.get("method"), "url": r.get("url"),
                                               "post_data": r.get("post_data")}
                                              for c in crawls
                                              for r in c.get("observed_requests", [])],
                        "verdict": out.get("verdict_text", "")})

                    # ---- Phase 2: assemble raw surface (CODE) + summarize/grade ----
                    merged = self._merge(crawls)
                    observed = self._observed_requests(merged)
                    self._assign_interest(observed)
                    if mode == "focused":
                        observed = [r for r in observed if r.interest != "low"]
                    extra_paths = {urlparse(u).path for u in api_file_eps if urlparse(u).path}
                    enum_findings: list[dict] = []
                    for e in (out.get("enum") or []):
                        tool, res, a = e.get("tool"), e.get("result") or {}, e.get("args") or {}
                        if tool in ("katana", "ffuf"):
                            for u in (res.get("endpoints", [])
                                      + [h.get("url", "") for h in res.get("hits", [])]):
                                pth = urlparse(u).path
                                if pth:
                                    extra_paths.add(pth)
                        elif tool == "arjun":
                            for prm in res.get("parameters", []):
                                enum_findings.append({
                                    "finding": f"hidden parameter '{prm}'",
                                    "where": a.get("url", ""), "grade": "high",
                                    "category": "hidden-parameter",
                                    "reason": "endpoint accepts a parameter not exposed in the UI (arjun)"})
                    peek_findings = self._peek_findings(peek_pages)
                    summary, graded = await self._grade(
                        merged, observed, peek_pages, api_file_eps, modepre + pre, config)
                    findings = self._group_findings((graded or []) + enum_findings + peek_findings)
            except WorkFailed:
                raise
            except Exception as exc:  # noqa: BLE001 — retryable stage failure
                raise WorkFailed(f"recon loop failed: {type(exc).__name__}: {exc}") from exc
            finally:
                flush()

            findings = self._attach_requests(findings, observed)
            findings_tree = self._findings_tree(summary, findings)
            self._dump("phase2_findings", {"summary": summary, "findings_tree": findings_tree,
                                           "findings_flat": findings})
            request_tree = self._request_tree(summary, merged, observed)
            asset = AssetMap(
                target_url=ctx.target_url, mode=mode, summary=summary, findings=findings,
                findings_tree=findings_tree, request_tree=request_tree,
                endpoints=self._endpoints(merged, routes), forms=self._forms(merged),
                technologies=[Technology(name=t["name"], source=t.get("source", ""),
                                         version=t.get("version") or None)
                              for t in merged.get("technologies", [])],
                api_paths=sorted(set(merged.get("js_api_paths", [])) | extra_paths),
                observed_requests=observed,
                third_party=merged.get("third_party_hosts", []),
                pages_crawled=merged.get("pages_crawled", 0),
                truncated=bool(merged.get("truncated")))

            self._validate(asset, ctx.target_url)
            set_run_output(_stage, {
                "mode": mode, "endpoints": len(asset.endpoints),
                "api_paths": len(asset.api_paths),
                "observed": len(asset.observed_requests),
                "findings": len(findings),
                "items": [{"finding": f.get("finding"), "where": f.get("where"),
                           "grade": f.get("grade")} for f in findings[:40]]})
        log.info("recon done [%s]: %d endpoints, %d api paths, %d observed "
                 "(%d high / %d medium / %d low), %d forms, %d pages, %d graded finding(s)%s",
                 mode, len(asset.endpoints), len(asset.api_paths), len(asset.observed_requests),
                 sum(1 for r in observed if r.interest == "high"),
                 sum(1 for r in observed if r.interest == "medium"),
                 sum(1 for r in observed if r.interest == "low"),
                 len(asset.forms), asset.pages_crawled, len(findings),
                 ", TRUNCATED" if asset.truncated else "")
        return asset.model_dump(mode="json")

    # --- per-phase audit dump (opt-in via APTS_RECON_DEBUG_DIR) ----------- #
    @staticmethod
    def _dump(tag: str, data: dict) -> None:
        d = os.environ.get("APTS_RECON_DEBUG_DIR")
        if not d:
            return
        try:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{tag}.json"), "w") as fh:
                json.dump(data, fh, default=str, indent=1)
        except Exception as exc:  # noqa: BLE001
            log.debug("recon phase dump %s failed: %s", tag, exc)

    # --- Phase 0 / 0.5 helpers -------------------------------------------- #
    @staticmethod
    def _cred_args(creds: dict) -> dict:
        """Map the engagement credentials onto crawl_v2's auth kwargs — only the
        keys that are set, so an unauthenticated scan passes nothing."""
        return {k: creds[k] for k in ("headers", "cookies", "basic_auth")
                if creds.get(k)}

    @staticmethod
    def _split_endpoints(endpoints: list[str], target: str) -> tuple[list[str], list[str]]:
        """Split katana's flat URI list into app ROUTES and concrete API/FILE endpoints."""
        host = urlparse(target).hostname or ""
        base = f"{urlparse(target).scheme}://{urlparse(target).netloc}"
        STATIC = (".js", ".css", ".png", ".jpg", ".jpeg", ".ico", ".woff2", ".svg",
                  ".map", ".gif", ".webp")
        NOISE = ("{{", "%7b", "%5c", "\\")
        routes: list[str] = []
        api_file: list[str] = []
        seen_r: set[str] = set()
        for u in endpoints:
            p = urlparse(u)
            if host and p.hostname and p.hostname != host:
                continue
            path = p.path or "/"
            low = path.lower()
            if any(n in u.lower() for n in NOISE) or low.endswith(STATIC):
                continue
            if low.startswith(("/rest", "/api", "/ftp")) or "." in path.rsplit("/", 1)[-1]:
                api_file.append(u)
            else:
                route = base + "/#" + path if not p.fragment else base + "/#" + p.fragment
                if route not in seen_r:
                    seen_r.add(route)
                    routes.append(route)
        if base + "/#/" not in seen_r and base not in seen_r:
            routes.insert(0, base + "/")
        return routes[:60], sorted(set(api_file))[:120]

    @staticmethod
    def _phase1_task(target: str, routes: list[str], api_file_eps: list[str],
                     peek_pages: list[dict], mode: str, pre: str) -> str:
        interactive = []
        for p in peek_pages:
            inputs = p.get("inputs", [])
            if not inputs:
                continue
            interactive.append(
                f"  {p.get('requested','')}  inputs="
                f"{[i.get('type')+':'+i.get('name','') for i in inputs][:8]}"
                f"  on_load_xhr={p.get('xhr_on_load',[])[:4]}")
        api = "\n".join(f"  {u}" for u in api_file_eps[:40]) or "  (none)"
        inter = "\n".join(interactive[:40]) or "  (none with inputs)"
        return (f"{pre}## Phase-0 katana found {len(routes)} route(s), "
                f"{len(api_file_eps)} API/file endpoint(s).\n"
                f"## Phase-0.5 peek — interactive pages (route -> input types + on-load APIs):\n"
                f"{inter}\n\n## Concrete API/file endpoints:\n{api}\n\n"
                f"Your job: CAPTURE the real HTTP requests for the vuln/exploit stages.\n"
                f"- For each interactive page, call crawl_v2(url=<the route>) so its form "
                f"submits and the real request (e.g. a login/search/complaint POST) is "
                f"recorded. Prioritise pages with user input (login, search, forms, upload).\n"
                f"- Use ffuf to find UNLINKED paths/IDs and arjun to find HIDDEN parameters "
                f"on real endpoints.\n"
                f"- Read each OBSERVATION; never repeat a call. Reply DONE when the worthwhile "
                f"surface is captured. You are in {mode.upper()} mode.")

    @staticmethod
    def _peek_findings(peek_pages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for p in peek_pages:
            where = p.get("requested", "")
            types = {i.get("type") for i in p.get("inputs", [])}
            names = {(i.get("name") or "").lower() for i in p.get("inputs", [])}
            if "file" in types:
                out.append({"finding": "file upload input", "where": where, "grade": "high",
                            "category": "file-upload",
                            "reason": "page accepts a file upload — test upload/type/path-traversal"})
            if "password" in types and ("email" in types
                                        or any("email" in n or "user" in n for n in names)):
                out.append({"finding": "authentication form", "where": where, "grade": "high",
                            "category": "authentication",
                            "reason": "credential form — test auth bypass / weak-credential / injection"})
            if "textarea" in types:
                out.append({"finding": "free-text input", "where": where, "grade": "medium",
                            "category": "user-input",
                            "reason": "free-text field — test stored XSS / injection"})
        return out

    @classmethod
    def _attach_requests(cls, findings: list[dict], observed: list) -> list[dict]:
        """Attach each finding's RAW captured request + the PAGE it belongs to."""
        by_path: dict = {}
        for r in observed:
            p = urlparse(r.url).path or "/"
            by_path.setdefault(p, r)

        def _req_dict(req) -> dict:
            return {"method": req.method, "path": urlparse(req.url).path or "/",
                    "url": req.url, "params": _params_of(req),
                    "post_data": req.post_data, "content_type": req.content_type}

        def _is_transport(u: str) -> bool:
            p = urlparse(u).path.lower()
            return "socket.io" in p or p.endswith((".io", "/poll")) or "/socket" in p

        for f in findings:
            wu = urlparse(str(f.get("where", "")))
            cat = str(f.get("category", "")).lower()
            req = None
            if cat == "hidden-parameter":
                f["page"] = None
                f["request"] = None
                continue
            if wu.fragment.startswith("/"):
                page = cls._endpoint_path(wu)
                f["page"] = page
                for r in observed:
                    if cls._endpoint_path(urlparse(r.source_page or "")) == page \
                            and (r.post_data or urlparse(r.url).query) \
                            and not _is_transport(r.url):
                        req = r
                        break
            else:
                epath = wu.path or str(f.get("where", ""))
                req = by_path.get(epath) if epath and epath != "/" else None
                f["page"] = (cls._endpoint_path(urlparse(req.source_page or "")) or epath) \
                    if req is not None else None
            f["request"] = _req_dict(req) if req is not None else None
        return findings

    @classmethod
    def _findings_tree(cls, summary: str, findings: list[dict]) -> dict:
        pages: dict[str, list] = {}
        order: list[str] = []
        rogue: list[dict] = []
        for f in findings:
            page = f.get("page")
            if not page:
                rogue.append(f)
                continue
            if page not in pages:
                pages[page] = []
                order.append(page)
            pages[page].append(f)
        return {"summary": summary, "rogue_findings": rogue,
                "pages": [{"page": p, "findings": pages[p]} for p in order]}

    @staticmethod
    def _group_findings(findings: list[dict]) -> list[dict]:
        order = {"high": 0, "medium": 1, "low": 2, "": 3}
        seen: set[tuple] = set()
        uniq: list[dict] = []
        for f in findings:
            if not isinstance(f, dict):
                continue
            f.setdefault("category", "other")
            k = (f.get("finding", ""), f.get("where", ""))
            if k in seen:
                continue
            seen.add(k)
            uniq.append(f)
        uniq.sort(key=lambda f: (f.get("category", "other"),
                                 order.get(str(f.get("grade", "")).lower(), 3)))
        return uniq

    # --- merge crawls into one surface dict (code-owned, intact) ---------- #
    @staticmethod
    def _merge(crawls: list[dict]) -> dict:
        pages: list[dict] = []
        seen_pages: set[str] = set()
        forms: list[dict] = []
        seen_forms: set[str] = set()
        observed: list[dict] = []
        seen_obs: set[tuple] = set()
        api_paths: set[str] = set()
        third: set[str] = set()
        techs: dict[str, dict] = {}
        truncated = False
        for c in crawls:
            truncated = truncated or bool(c.get("truncated"))
            for p in c.get("pages", []):
                u = p.get("url", "")
                if u and normalize_url(u) not in seen_pages:
                    seen_pages.add(normalize_url(u))
                    pages.append(p)
            for f in c.get("forms", []):
                k = json.dumps({"p": f.get("page"), "m": f.get("method"),
                                "a": f.get("action"),
                                "fl": [x.get("name") for x in f.get("fields", [])]},
                               sort_keys=True)
                if k not in seen_forms:
                    seen_forms.add(k)
                    forms.append(f)
            for r in c.get("observed_requests", []):
                k = (r.get("method"), r.get("url"), r.get("post_data"))
                if k not in seen_obs:
                    seen_obs.add(k)
                    observed.append(r)
            for a in c.get("js_api_paths", []):
                api_paths.add(a)
            for h in c.get("third_party_hosts", []):
                third.add(h)
            for t in c.get("technologies", []):
                techs.setdefault(t["name"], t)
        return {"pages": pages, "forms": forms, "observed_requests": observed,
                "js_api_paths": sorted(api_paths), "third_party_hosts": sorted(third),
                "technologies": list(techs.values()), "pages_crawled": len(pages),
                "truncated": truncated}

    # --- deterministic interest (no per-page model calls) ----------------- #
    @staticmethod
    def _assign_interest(observed: list[ObservedRequest]) -> None:
        for r in observed:
            if _params_of(r):
                r.interest, r.interest_reason = "high", "carries a parameter or body"
            elif (r.resource_type or "").lower() in ("xhr", "fetch") or r.method != "GET":
                r.interest, r.interest_reason = "medium", "app/API call without parameters"
            else:
                r.interest, r.interest_reason = "low", "static/transport"

    # --- the single grade call (model; additive, never touches raw data) --- #
    async def _grade(self, merged: dict, observed: list[ObservedRequest],
                     peek_pages: list[dict], api_file_eps: list[str],
                     preamble: str, config: dict) -> tuple[str, list[dict]]:
        endpoints = sorted({r.get("url", "") and (urlparse(r["url"]).path or "/")
                            for r in merged.get("observed_requests", [])})
        forms = [f"{urlparse(f.get('page','')).path or '/'} "
                 f"[{','.join(x.get('name','') for x in f.get('fields', []))}]"
                 for f in merged.get("forms", [])]
        params = sorted({f"{urlparse(r.url).path or '/'}?{p}"
                        for r in observed if r.interest == "high"
                        for p in _params_of(r)})
        peek_view = [{"page": urlparse(p.get("requested", "")).fragment
                      or urlparse(p.get("requested", "")).path,
                      "inputs": [i.get("type") for i in p.get("inputs", [])],
                      "xhr": p.get("xhr_on_load", [])[:6]}
                     for p in peek_pages if p.get("inputs")][:40]
        view = {
            "target_paths": endpoints[:80], "forms": forms[:20],
            "parameterised_requests": params[:80],
            "api_paths": (merged.get("js_api_paths", []) + api_file_eps)[:100],
            "interactive_pages": peek_view,
            "technologies": [t.get("name") for t in merged.get("technologies", [])],
        }
        grade = prompts.get(RECON_GRADE_PROMPT_ID)
        user = f"{preamble}Attack surface:\n{json.dumps(view, default=str)[:7000]}"
        try:
            model = build_chat_model("recon", max_tokens=MAX_TOKENS)
            with trace_run(name="recon-grade", run_input=user,
                           prompt=grade.trace_prompt):
                msg = await model.ainvoke(
                    agentic.to_lc_messages(grade.text, user), config=config)
            return _parse_grade(agentic.message_text(msg))
        except Exception as exc:  # noqa: BLE001
            log.warning("recon grade call failed: %s — keeping raw surface only", exc)
            return "", []

    # --- validation: the contract must hold ------------------------------- #
    @staticmethod
    def _validate(asset: AssetMap, target: str) -> None:
        if not asset.observed_requests and not asset.endpoints and not asset.api_paths:
            raise WorkFailed(
                f"recon mapped an empty surface for {target} — the crawl reached "
                f"nothing testable (target down, out of scope, or not HTML?)")
        for r in asset.observed_requests:
            if not r.url or not r.method:
                raise WorkFailed("recon produced a malformed observed request "
                                 "(missing method/url) — surface not trustworthy")

    # --- AssetMap builders ------------------------------------------------- #
    @classmethod
    def _endpoints(cls, crawl: dict, extra_routes: list[str] | None = None) -> list[Endpoint]:
        endpoints: list[Endpoint] = []
        seen: set[str] = set()
        for page in crawl.get("pages", []):
            parsed = urlparse(page.get("url", ""))
            path = cls._endpoint_path(parsed)
            if path in seen:
                continue
            seen.add(path)
            endpoints.append(Endpoint(
                path=path, status=page.get("status"),
                params=sorted({p.split("=")[0] for p in parsed.query.split("&") if p}),
                content_type=page.get("content_type", ""),
                response_headers=page.get("response_headers", {}) or {}))
        for url in extra_routes or []:
            path = cls._endpoint_path(urlparse(url))
            if path in seen:
                continue
            seen.add(path)
            endpoints.append(Endpoint(path=path, status=None, params=[],
                                      content_type="", response_headers={}))
        return endpoints

    @staticmethod
    def _endpoint_path(parsed) -> str:
        path = parsed.path or "/"
        fragment = parsed.fragment
        if fragment.startswith("/"):
            prefix = "" if path == "/" else path.rstrip("/")
            return f"{prefix}/#{fragment}"
        return path

    @staticmethod
    def _observed_requests(crawl: dict) -> list[ObservedRequest]:
        return [
            ObservedRequest(
                method=r.get("method", "GET"), url=r.get("url", ""),
                resource_type=r.get("resource_type", ""),
                content_type=r.get("content_type", ""),
                post_data=r.get("post_data"),
                headers=r.get("headers", {}) or {},
                source_page=r.get("source_page", ""))
            for r in crawl.get("observed_requests", [])
        ]

    @staticmethod
    def _forms(crawl: dict) -> list[Form]:
        return [
            Form(page=urlparse(form.get("page", "")).path or "/",
                 method=form.get("method", "GET"), action=form.get("action", ""),
                 fields=[FormField(name=f["name"], type=f.get("type", "text"))
                         for f in form.get("fields", [])])
            for form in crawl.get("forms", [])
        ]

    @classmethod
    def _request_tree(cls, summary: str, merged: dict, observed: list) -> dict:
        pages: dict[str, list] = {}
        order: list[str] = []
        for r in observed:
            if r.interest == "low":
                continue
            label = cls._endpoint_path(urlparse(r.source_page or "")) or "(unknown page)"
            if label not in pages:
                pages[label] = []
                order.append(label)
            pages[label].append({
                "method": r.method, "path": urlparse(r.url).path or "/",
                "params": _params_of(r), "interest": r.interest,
                "reason": r.interest_reason})
        return {"summary": summary,
                "technologies": [t.get("name") for t in merged.get("technologies", [])],
                "pages": [{"page": p, "requests": pages[p]} for p in order]}


def _params_of(req: ObservedRequest) -> list[str]:
    params = [k for k, _ in parse_qsl(urlparse(req.url).query, keep_blank_values=True)]
    if req.post_data and req.post_data.strip():
        try:
            obj = json.loads(req.post_data)
            if isinstance(obj, dict):
                params += list(obj.keys())
        except json.JSONDecodeError:
            params += [k for k, _ in parse_qsl(req.post_data, keep_blank_values=True)]
    seen: set[str] = set()
    return [p for p in params if not (p in seen or seen.add(p))]


def _parse_grade(text: str) -> tuple[str, list[dict]]:
    """Pull {summary, findings} out of the grade reply, tolerant of stray prose."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return "", []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "", []
    if not isinstance(data, dict):
        return "", []
    summary = str(data.get("summary", ""))[:2000]
    raw = data.get("findings") if isinstance(data.get("findings"), list) else []
    findings = []
    for f in raw[:50]:
        if not isinstance(f, dict):
            continue
        findings.append({
            "finding": str(f.get("finding", ""))[:200],
            "where": str(f.get("where", ""))[:200],
            "category": str(f.get("category", "other"))[:40] or "other",
            "grade": str(f.get("grade", "")).lower().strip()
                     if str(f.get("grade", "")).lower().strip() in ("high", "medium", "low") else "",
            "reason": str(f.get("reason", ""))[:200]})
    return summary, findings
