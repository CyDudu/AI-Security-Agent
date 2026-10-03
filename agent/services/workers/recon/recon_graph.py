"""Reconnaissance Phase-1 crawl-decision loop, as an explicit LangGraph ``StateGraph``.

Ported from the langgraph-branch PoC (which drove the model with raw httpx) onto our
`ChatOpenAI` + Langfuse stack, same shape as ``exploit/exploit_graph.py``:

    START -> agent --(tool_calls & step<MAX)--> tools -> agent
                   \\--(else)--------------------> finalize -> END

A code-driven breadth sweep (katana) + behaviour peek already ran; here the model
DECIDES which routes/endpoints to act on, one decision at a time, using crawl_v2 (capture
real form requests), ffuf (unlinked paths / enumerable IDs) and arjun (hidden params).
LangGraph owns *when*; plain Python owns *what* — the de-dup guard (``crawled`` +
normalized set), scope + shallow/targeted crawl caps, and the coaching observations.
``model`` (bound ChatOpenAI) and ``call_tool`` (MCP client wrapper) are INJECTED, so the
same graph runs the real model+tools or test fakes; Langfuse traces it via the callbacks
in ``graph.ainvoke(config=...)``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Awaitable, Callable, TypedDict
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from agent.services.workers.common import common_agentic as agentic

CallTool = Callable[[str, dict], Awaitable[Any]]

# Each agent-loop crawl is SHALLOW + TARGETED: the code-driven katana sweep already did
# the breadth mapping, so a loop crawl maps ONE interesting route, not the whole site.
_ROUTE_MAX_PAGES = int(os.environ.get("APTS_RECON_ROUTE_MAX_PAGES", "4"))
_ROUTE_MAX_DEPTH = int(os.environ.get("APTS_RECON_ROUTE_MAX_DEPTH", "1"))
_TOOLS = ("crawl", "crawl_v2", "katana", "ffuf", "arjun")
_CRAWLERS = ("crawl", "crawl_v2")


def normalize_url(url: str) -> str:
    """Canonical page identity for the de-dup guard. Keeps an SPA hash-route
    (``#/search``) — on a single-page app the hash IS the page, not a fragment."""
    try:
        p = urlparse(url.strip())
    except Exception:  # noqa: BLE001
        return url.strip()
    host = (p.hostname or "").lower()
    port = f":{p.port}" if p.port and p.port not in (80, 443) else ""
    path = (p.path or "/").rstrip("/") or "/"
    frag = p.fragment
    tail = f"#{frag.rstrip('/') or '/'}" if frag.startswith("/") else ""
    query = f"?{p.query}" if p.query else ""
    return f"{p.scheme}://{host}{port}{path}{query}{tail}"


class ReconState(TypedDict, total=False):
    mode: str
    target: str
    domains: list[str]
    creds: dict[str, Any]
    messages: list                 # LangChain messages
    tried: dict[str, str]          # repeat guard: crawl signature -> one-line outcome
    crawled: list[str]             # normalized page URLs already crawled (-> skip_urls)
    crawls: list[dict]             # raw crawl results, accumulated (code-owned, intact)
    enum: list[dict]               # katana/ffuf/arjun results, accumulated
    step: int
    verdict_text: str


def _crawl_digest(crawl: dict, before: set[str]) -> tuple[str, list[str]]:
    """Summarize one crawl result for the observation; return newly-crawled page URLs."""
    pages = [p.get("url", "") for p in crawl.get("pages", []) if p.get("url")]
    new_pages = [u for u in pages if normalize_url(u) not in before]
    mined = crawl.get("mined_routes", []) or []
    uncrawled = [m for m in mined if normalize_url(m) not in before
                 and normalize_url(m) not in {normalize_url(u) for u in pages}]
    forms = crawl.get("forms", []) or []
    obs = crawl.get("observed_requests", []) or []
    digest = {
        "crawled_pages": pages[:20],
        "forms_found": [{"page": urlparse(f.get("page", "")).path or "/",
                         "method": f.get("method"),
                         "fields": [x.get("name") for x in f.get("fields", [])]}
                        for f in forms[:10]],
        "observed_requests": len(obs),
        "uncrawled_mined_routes": uncrawled[:40],
        "truncated": bool(crawl.get("truncated")),
    }
    return json.dumps(digest, default=str)[:2000], new_pages


def _enum_digest(name: str, result: dict) -> str:
    if not isinstance(result, dict):
        return str(result)[:500]
    if result.get("error"):
        return json.dumps({"error": result["error"]})[:500]
    if name == "katana":
        return json.dumps({"endpoints_found": result.get("count"),
                           "sample": result.get("endpoints", [])[:15]}, default=str)[:1500]
    if name == "ffuf":
        return json.dumps({"hits": result.get("count"),
                           "found": [{"in": h.get("input"), "code": h.get("status"),
                                      "url": h.get("url")}
                                     for h in result.get("hits", [])[:20]]}, default=str)[:1800]
    if name == "arjun":
        return json.dumps({"parameters_found": result.get("parameters", [])}, default=str)[:800]
    return json.dumps(result, default=str)[:800]


def _enum_outcome(name: str, result: dict) -> str:
    if not isinstance(result, dict):
        return "ok"
    if result.get("error"):
        return f"error: {str(result['error'])[:60]}"
    if name == "katana":
        return f"{result.get('count', 0)} endpoints"
    if name == "ffuf":
        return f"{result.get('count', 0)} hits"
    if name == "arjun":
        return f"params: {','.join(result.get('parameters', [])) or 'none'}"
    return "ok"


def build_recon_graph(model, call_tool: CallTool, *, max_steps: int, budget_chars: int):
    """Compile the recon crawl-decision loop around an injected bound model + tool driver."""
    compact = agentic.compacting_hook(budget_chars)

    async def agent(state: ReconState, config) -> dict:
        msgs = state["messages"]
        to_model = compact({"messages": msgs}).get("llm_input_messages") or msgs
        try:
            msg = await model.ainvoke(to_model, config=config)
        except Exception as exc:  # noqa: BLE001
            msg = AIMessage(content=(f"DONE — model call failed at step {state['step']} "
                                     f"({type(exc).__name__}: {exc})"))
        return {"messages": msgs + [msg], "step": state["step"] + 1}

    async def tools(state: ReconState) -> dict:
        msg = state["messages"][-1]
        tried = dict(state.get("tried", {}))
        crawled = list(state.get("crawled", []))
        crawls = list(state.get("crawls", []))
        enum = list(state.get("enum", []))
        crawled_set = {normalize_url(u) for u in crawled}
        steps_left = max(0, max_steps - state["step"])
        creds = state.get("creds") or {}
        new_msgs: list = []

        for call in getattr(msg, "tool_calls", None) or []:
            name = call["name"]
            args = dict(call.get("args") or {})
            cid = call.get("id")
            if name not in _TOOLS:
                new_msgs.append(ToolMessage(
                    content=f"Unknown tool {name!r}; available: {', '.join(_TOOLS)}.",
                    tool_call_id=cid))
                continue
            key = agentic.sig(name, args)
            if key in tried:
                new_msgs.append(ToolMessage(
                    content=(f"You already ran this exact call -> {tried[key]}. "
                             "Change the arguments or tool, or reply DONE."),
                    tool_call_id=cid))
                continue

            if name in _CRAWLERS:
                seed = args.get("url", "")
                if seed and normalize_url(seed) in crawled_set:
                    new_msgs.append(ToolMessage(
                        content=(f"Already crawled {seed}. Pick an uncrawled route, use "
                                 "ffuf/arjun/katana to find more, or reply DONE."),
                        tool_call_id=cid))
                    continue
                args.setdefault("allow_domains", list(state.get("domains", []) or []))
                args["skip_urls"] = crawled
                if name == "crawl":
                    args["mine"] = False
                args["max_pages"] = min(int(args.get("max_pages", _ROUTE_MAX_PAGES)
                                            or _ROUTE_MAX_PAGES), _ROUTE_MAX_PAGES)
                args["max_depth"] = min(int(args.get("max_depth", _ROUTE_MAX_DEPTH)
                                            or _ROUTE_MAX_DEPTH), _ROUTE_MAX_DEPTH)
                for k in ("headers", "cookies", "basic_auth"):
                    if creds.get(k) and k not in args:
                        args[k] = creds[k]
                result = await call_tool(name, args)
                crawls.append(result)
                digest, new_pages = _crawl_digest(result, crawled_set)
                for u in new_pages:
                    nu = normalize_url(u)
                    if nu not in crawled_set:
                        crawled_set.add(nu)
                        crawled.append(u)
                tried[key] = (f"{len(new_pages)} new page(s), "
                              f"{len(result.get('observed_requests', []))} request(s)")
                new_msgs.append(ToolMessage(
                    content=(f"OBSERVATION from {name} of {seed}:\n{digest}\n---\n"
                             f"PROGRESS: {steps_left} step(s) left. Crawled so far: "
                             f"{len(crawled)} page(s). Crawl another route, run "
                             "ffuf/arjun/katana, or reply DONE."),
                    tool_call_id=cid))
            else:
                result = await call_tool(name, args)
                enum.append({"tool": name, "args": args, "result": result})
                tried[key] = _enum_outcome(name, result)
                new_msgs.append(ToolMessage(
                    content=(f"OBSERVATION from {name}:\n{_enum_digest(name, result)}\n---\n"
                             f"PROGRESS: {steps_left} step(s) left. Act on a finding "
                             "(crawl a discovered path, note a param), run another tool, "
                             "or reply DONE."),
                    tool_call_id=cid))

        return {"messages": state["messages"] + new_msgs, "tried": tried,
                "crawled": crawled, "crawls": crawls, "enum": enum}

    def finalize(state: ReconState) -> dict:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            text = "DONE — hit the step limit"
        else:
            text = agentic.message_text(last) or "DONE"
        return {"verdict_text": text}

    def route(state: ReconState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None) and state["step"] < max_steps:
            return "tools"
        return "finalize"

    g = StateGraph(ReconState)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("finalize", finalize)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "finalize": "finalize"})
    g.add_edge("tools", "agent")
    g.add_edge("finalize", END)
    return g.compile()
