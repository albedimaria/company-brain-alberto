"""Tools the agent can call: typed wrappers over the Al Dente mock API, the KB
search, plus deterministic Python aggregation (counts/sums/group-by) so the LLM
never does arithmetic.

Every wrapper returns a compact JSON-serializable payload. `run_tool` returns
`(payload, source_ids)`; the agent accumulates `source_ids` into the response
`sources` field (provenance).
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from artifacts import generate_artifact
from config import mock_api
from rag import get_kb_document, kb_search

MAX_LIMIT = 200  # API hard cap
_EUR_FIELDS = ("value_eur", "total_eur", "amount_eur")
_OPEN_STAGES = ("qualification", "negotiation")
_VALID_STAGES = {"qualification", "negotiation", "won", "lost"}


# --------------------------------------------------------------------------- #
# Low-level HTTP                                                              #
# --------------------------------------------------------------------------- #
def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET a JSON endpoint. 404 -> {'_not_found': True} so callers can verify
    premises (trap questions) instead of crashing."""
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    resp = mock_api().get(path, params=clean)
    if resp.status_code == 404:
        return {"_not_found": True}
    resp.raise_for_status()
    return resp.json()


def _list(path: str, params: dict[str, Any], all_pages: bool) -> list[dict[str, Any]]:
    """List endpoint with the {data, pagination} envelope. When `all_pages`,
    page through `pagination.total` (max 200/page) before returning."""
    params = dict(params)
    params.setdefault("limit", MAX_LIMIT if all_pages else 50)
    first = _get(path, {**params, "offset": 0})
    rows: list[dict[str, Any]] = list(first.get("data", []))
    total = first.get("pagination", {}).get("total", len(rows))
    if all_pages:
        while len(rows) < total:
            page = _get(path, {**params, "limit": MAX_LIMIT, "offset": len(rows)})
            batch = page.get("data", [])
            if not batch:
                break
            rows.extend(batch)
    return rows


def _sums(items: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for field in _EUR_FIELDS:
        vals = [r[field] for r in items if isinstance(r.get(field), (int, float))]
        if vals:
            out[field] = round(sum(vals), 2)
    return out


@lru_cache
def _customer_channel_map() -> dict[str, str]:
    rows = _list("/crm/customers", {}, all_pages=True)
    return {r["id"]: r.get("channel", "unknown") for r in rows}


@lru_cache
def _supplier_name_map() -> dict[str, str]:
    rows = _list("/erp/suppliers", {}, all_pages=True)
    return {r["id"]: r.get("name", r["id"]) for r in rows}


def _group(items: list[dict[str, Any]], group_by: str) -> dict[str, Any]:
    """Group rows and compute count + EUR sums per group. Special key
    'customer_channel' joins via the customer's channel."""
    chan = _customer_channel_map() if group_by == "customer_channel" else None
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in items:
        key = (
            chan.get(r.get("customer_id"), "unknown")
            if chan
            else str(r.get(group_by, "unknown"))
        )
        groups.setdefault(key, []).append(r)
    return {k: {"count": len(v), **_sums(v)} for k, v in groups.items()}


# --------------------------------------------------------------------------- #
# CRM                                                                         #
# --------------------------------------------------------------------------- #
def search_customers(search=None, channel=None, status=None, all_pages=True) -> dict:
    items = _list(
        "/crm/customers",
        {"search": search, "channel": channel, "status": status},
        all_pages,
    )
    return {"count": len(items), "customers": items}


def get_customer(customer_id: str) -> dict:
    data = _get(f"/crm/customers/{customer_id}")
    if data.get("_not_found"):
        return {"found": False, "customer_id": customer_id}
    return {"found": True, "customer": data}


def list_opportunities(
    customer_id=None, stage=None, owner=None, group_by=None, all_pages=True
) -> dict:
    # The model sometimes passes a conceptual stage like "open"/"closed", which
    # is not a real stage value and would filter the API down to zero rows.
    # Drop anything outside the real enum: open_count/open_value_eur below are
    # computed in Python over the full set, so ignoring it yields correct totals.
    if stage not in _VALID_STAGES:
        stage = None
    items = _list(
        "/crm/opportunities",
        {"customer_id": customer_id, "stage": stage, "owner": owner},
        all_pages,
    )
    open_items = [r for r in items if r.get("stage") in _OPEN_STAGES]
    out: dict[str, Any] = {
        "count": len(items),
        "sums_eur": _sums(items),
        "open_count": len(open_items),
        "open_value_eur": _sums(open_items).get("value_eur", 0),
        # Always precompute the channel split so the model never has to infer it.
        "by_customer_channel": _group(items, "customer_channel"),
        "opportunities": items,
    }
    if group_by:
        out["grouped_by"] = group_by
        out["groups"] = _group(items, group_by)
    return out


def list_orders(
    customer_id=None,
    status=None,
    date_from=None,
    date_to=None,
    group_by=None,
    all_pages=True,
) -> dict:
    items = _list(
        "/crm/orders",
        {
            "customer_id": customer_id,
            "status": status,
            "from": date_from,
            "to": date_to,
        },
        all_pages,
    )
    out: dict[str, Any] = {
        "count": len(items),
        "sums_eur": _sums(items),
        "by_customer_channel": _group(items, "customer_channel"),
        "orders": items,
    }
    if group_by:
        out["grouped_by"] = group_by
        out["groups"] = _group(items, group_by)
    return out


def list_invoices(customer_id=None, status=None, order_id=None, all_pages=True) -> dict:
    items = _list(
        "/crm/invoices",
        {"customer_id": customer_id, "status": status, "order_id": order_id},
        all_pages,
    )
    return {"count": len(items), "sums_eur": _sums(items), "invoices": items}


# --------------------------------------------------------------------------- #
# Calls                                                                       #
# --------------------------------------------------------------------------- #
def list_calls(
    customer_id=None,
    type=None,
    outcome=None,
    date_from=None,
    date_to=None,
    topic_contains=None,
    all_pages=True,
) -> dict:
    items = _list(
        "/calls",
        {
            "customer_id": customer_id,
            "type": type,
            "outcome": outcome,
            "from": date_from,
            "to": date_to,
        },
        all_pages,
    )
    if topic_contains:
        t = topic_contains.lower()
        items = [
            c
            for c in items
            if t in (c.get("topic", "") + " " + c.get("summary", "")).lower()
        ]
    return {"count": len(items), "calls": items}


def get_call(call_id: str) -> dict:
    data = _get(f"/calls/{call_id}")
    if data.get("_not_found"):
        return {"found": False, "call_id": call_id}
    return {"found": True, "call": data}


def search_transcript(call_id: str, search=None, speaker=None, limit=20) -> dict:
    data = _get(
        f"/calls/{call_id}/transcript",
        {"search": search, "speaker": speaker, "limit": limit},
    )
    if data.get("_not_found"):
        return {"found": False, "call_id": call_id}
    return {
        "found": True,
        "call_id": call_id,
        "matched_segments": data.get("pagination", {}).get("total"),
        "segments": data.get("segments", []),
    }


def count_calls(topic_contains: str, type=None, outcome=None) -> dict:
    """Page the WHOLE call log and count calls whose topic/summary mention a
    term (e.g. a defect). Arithmetic done here, not by the LLM."""
    items = _list("/calls", {"type": type, "outcome": outcome}, all_pages=True)
    t = topic_contains.lower()
    matches = [
        c
        for c in items
        if t in (c.get("topic", "") + " " + c.get("summary", "")).lower()
    ]
    return {
        "term": topic_contains,
        "total_calls_scanned": len(items),
        "match_count": len(matches),
        "matching_call_ids": [c["id"] for c in matches],
    }


# --------------------------------------------------------------------------- #
# ERP                                                                         #
# --------------------------------------------------------------------------- #
def list_production_orders(
    customer_id=None,
    status=None,
    sku=None,
    date_from=None,
    date_to=None,
    all_pages=True,
) -> dict:
    items = _list(
        "/erp/production-orders",
        {
            "customer_id": customer_id,
            "status": status,
            "sku": sku,
            "from": date_from,
            "to": date_to,
        },
        all_pages,
    )
    return {"count": len(items), "production_orders": items}


def list_inventory(
    type=None, below_min=None, search=None, sku=None, all_pages=True
) -> dict:
    items = _list(
        "/erp/inventory",
        {"type": type, "below_min": below_min, "search": search or sku},
        all_pages,
    )
    if sku:
        exact = [r for r in items if r.get("sku") == sku]
        if exact:
            items = exact
    # enrich raw materials with their supplier name (saves a hop on multi-source chains)
    names = _supplier_name_map() if any(r.get("supplier_id") for r in items) else {}
    for r in items:
        if r.get("supplier_id"):
            r["supplier_name"] = names.get(r["supplier_id"], r["supplier_id"])
    return {"count": len(items), "inventory": items}


def list_suppliers(search=None, category=None, all_pages=True) -> dict:
    items = _list("/erp/suppliers", {"search": search, "category": category}, all_pages)
    return {"count": len(items), "suppliers": items}


def get_bom(sku: str) -> dict:
    rows = _list("/erp/bom", {"sku": sku}, all_pages=True)
    if not rows:
        return {"found": False, "sku": sku}
    return {"found": True, "bom": rows[0]}


def list_shipments(
    customer_id=None, order_id=None, status=None, all_pages=True
) -> dict:
    items = _list(
        "/erp/shipments",
        {"customer_id": customer_id, "order_id": order_id, "status": status},
        all_pages,
    )
    return {"count": len(items), "shipments": items}


# --------------------------------------------------------------------------- #
# KB                                                                          #
# --------------------------------------------------------------------------- #
def kb_search_tool(query: str, k: int = 4) -> dict:
    hits = kb_search(query, k=k)
    return {"results": [{"doc_id": d, "content": c} for d, c in hits]}


def get_kb_document_tool(doc_id: str) -> dict:
    content = get_kb_document(doc_id)
    if content is None:
        return {"found": False, "doc_id": doc_id}
    return {"found": True, "doc_id": doc_id, "content": content}


# --------------------------------------------------------------------------- #
# Dispatch + provenance + tool schemas                                        #
# --------------------------------------------------------------------------- #
_SOURCE: dict[str, str] = {
    "search_customers": "crm/customers",
    "get_customer": "crm/customers",
    "list_opportunities": "crm/opportunities",
    "list_orders": "crm/orders",
    "list_invoices": "crm/invoices",
    "list_calls": "calls",
    "get_call": "calls",
    "search_transcript": "calls/transcript",
    "count_calls": "calls",
    "list_production_orders": "erp/production-orders",
    "list_inventory": "erp/inventory",
    "list_suppliers": "erp/suppliers",
    "get_bom": "erp/bom",
    "list_shipments": "erp/shipments",
}

_FUNCS = {
    "search_customers": search_customers,
    "get_customer": get_customer,
    "list_opportunities": list_opportunities,
    "list_orders": list_orders,
    "list_invoices": list_invoices,
    "list_calls": list_calls,
    "get_call": get_call,
    "search_transcript": search_transcript,
    "count_calls": count_calls,
    "list_production_orders": list_production_orders,
    "list_inventory": list_inventory,
    "list_suppliers": list_suppliers,
    "get_bom": get_bom,
    "list_shipments": list_shipments,
    "kb_search": kb_search_tool,
    "get_kb_document": get_kb_document_tool,
    "generate_artifact": generate_artifact,
}


def _doc_sources(payload: dict) -> list[str]:
    ids: list[str] = []
    for r in payload.get("results", []):
        if r.get("doc_id"):
            ids.append(r["doc_id"])
    if payload.get("doc_id") and payload.get("found"):
        ids.append(payload["doc_id"])
    return ids


# tool name -> (list_key, id_field) for extracting real entity ids from list payloads
_LIST_IDS: dict[str, tuple[str, str]] = {
    "search_customers": ("customers", "id"),
    "list_opportunities": ("opportunities", "id"),
    "list_orders": ("orders", "id"),
    "list_invoices": ("invoices", "id"),
    "list_calls": ("calls", "id"),
    "list_production_orders": ("production_orders", "id"),
    "list_inventory": ("inventory", "sku"),
    "list_suppliers": ("suppliers", "id"),
    "list_shipments": ("shipments", "id"),
}

_ENTITY_CAP = 8  # avoid flooding `sources` with 100+ ids on aggregate queries


def _entity_sources(name: str, payload: dict) -> list[str]:
    """Extract the real entity ids a tool actually touched (CUST-/OPP-/CALL-/
    LOT-/SKU/...). Capped so aggregates don't dump the whole result set."""
    ids: list[str] = []
    # single-entity lookups
    if name == "get_customer" and payload.get("found"):
        ids.append(payload["customer"]["id"])
    elif name == "get_call" and payload.get("found"):
        ids.append(payload["call"]["id"])
    elif name == "get_bom" and payload.get("found"):
        ids.append(payload["bom"]["sku"])
    elif name == "search_transcript" and payload.get("found"):
        ids.append(payload["call_id"])
    elif name == "count_calls":
        ids.extend(payload.get("matching_call_ids", []))
    elif name in _LIST_IDS:
        key, field = _LIST_IDS[name]
        ids.extend(r[field] for r in payload.get(key, []) if r.get(field))
    # dedup preserving order, then cap
    seen: list[str] = []
    for i in ids:
        if i not in seen:
            seen.append(i)
    return seen[:_ENTITY_CAP]


def run_tool(name: str, args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Execute a tool. Returns (payload_for_llm, source_ids).

    Sources are the endpoint name (spec example uses these, and they read well
    for aggregates) PLUS the concrete entity ids touched (provenance). KB tools
    return DOC ids only.
    """
    func = _FUNCS.get(name)
    if func is None:
        return {"error": f"unknown tool {name}"}, []
    try:
        payload = func(**args)
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}, []
    except Exception as exc:  # surface API errors to the model, keep loop alive
        return {"error": f"{name} failed: {exc}"}, []

    if name in ("kb_search", "get_kb_document"):
        return payload, _doc_sources(payload)
    endpoint = [_SOURCE[name]] if name in _SOURCE else []
    return payload, endpoint + _entity_sources(name, payload)


def _tool(
    name: str, description: str, properties: dict, required: list[str] | None = None
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }


_STR = {"type": "string"}
_BOOL = {"type": "boolean"}

TOOLS: list[dict] = [
    _tool(
        "search_customers",
        "Search/list CRM customers. Filters are exact, case-sensitive.",
        {
            "search": {**_STR, "description": "name fragment, e.g. 'Primato'"},
            "channel": {"type": "string", "enum": ["GDO", "distributor", "horeca"]},
            "status": {"type": "string", "enum": ["active", "inactive", "prospect"]},
        },
    ),
    _tool(
        "get_customer",
        "Get one customer by id (CUST-####). found=false if it does not exist.",
        {"customer_id": _STR},
        ["customer_id"],
    ),
    _tool(
        "list_opportunities",
        "List opportunities. Returns count, EUR sums, and open_count/open_value_eur (open = qualification+negotiation). Use group_by='customer_channel' to break totals down by GDO/distributor/horeca. When the question is about a specific customer, you MUST pass customer_id.",
        {
            "customer_id": _STR,
            "stage": {
                "type": "string",
                "enum": ["qualification", "negotiation", "won", "lost"],
            },
            "owner": _STR,
            "group_by": _STR,
        },
    ),
    _tool(
        "list_orders",
        "List orders. Returns count + EUR sums. group_by optional ('customer_channel' or a field). When the question is about a specific customer, you MUST pass customer_id.",
        {
            "customer_id": _STR,
            "status": {
                "type": "string",
                "enum": ["open", "in_production", "shipped", "delivered", "cancelled"],
            },
            "date_from": _STR,
            "date_to": _STR,
            "group_by": _STR,
        },
    ),
    _tool(
        "list_invoices",
        "List invoices with EUR sums. When the question is about a specific customer, you MUST pass customer_id.",
        {
            "customer_id": _STR,
            "status": {"type": "string", "enum": ["unpaid", "paid", "overdue"]},
            "order_id": _STR,
        },
    ),
    _tool(
        "list_calls",
        "List call logs (metadata: topic, summary, outcome, related_lot_id). topic_contains filters client-side over topic+summary. When the question is about a specific customer, you MUST pass customer_id.",
        {
            "customer_id": _STR,
            "type": {"type": "string", "enum": ["sales", "support"]},
            "outcome": {
                "type": "string",
                "enum": ["complaint_open", "follow_up", "order_placed", "resolved"],
            },
            "topic_contains": _STR,
        },
    ),
    _tool(
        "get_call",
        "Get one call's metadata by id (CALL-#####).",
        {"call_id": _STR},
        ["call_id"],
    ),
    _tool(
        "search_transcript",
        "Surgically extract transcript segments of a call. ALWAYS pass a search term or speaker; do not download whole transcripts.",
        {
            "call_id": _STR,
            "search": _STR,
            "speaker": {"type": "string", "enum": ["customer", "agent"]},
        },
        ["call_id"],
    ),
    _tool(
        "count_calls",
        "Page the ENTIRE call log and count calls whose topic/summary mention a term (e.g. a defect like 'broken pasta'). Use for 'how many calls about X'.",
        {
            "topic_contains": _STR,
            "type": {"type": "string", "enum": ["sales", "support"]},
            "outcome": {
                "type": "string",
                "enum": ["complaint_open", "follow_up", "order_placed", "resolved"],
            },
        },
        ["topic_contains"],
    ),
    _tool(
        "list_production_orders",
        "List production lots (LOT-####): sku, status, quality_status, linked_order_id.",
        {
            "customer_id": _STR,
            "status": {
                "type": "string",
                "enum": ["planned", "in_progress", "done", "blocked"],
            },
            "sku": _STR,
        },
    ),
    _tool(
        "list_inventory",
        "Inventory for finished goods and raw materials: on_hand, min_stock, below_min, supplier_id (raw). Pass sku to check one item.",
        {
            "type": {"type": "string", "enum": ["finished_good", "raw_material"]},
            "below_min": {"type": "string", "enum": ["true"]},
            "search": _STR,
            "sku": _STR,
        },
    ),
    _tool(
        "list_suppliers",
        "List suppliers (SUP-###) with category.",
        {
            "search": _STR,
            "category": {
                "type": "string",
                "enum": [
                    "semolina",
                    "wheat",
                    "packaging",
                    "labels",
                    "ink",
                    "logistics",
                ],
            },
        },
    ),
    _tool(
        "get_bom",
        "Bill of materials of a finished SKU: components with raw_sku and qty_per_carton.",
        {"sku": _STR},
        ["sku"],
    ),
    _tool(
        "list_shipments",
        "List shipments with status (in_transit/delivered/delayed).",
        {
            "customer_id": _STR,
            "order_id": _STR,
            "status": {
                "type": "string",
                "enum": ["in_transit", "delivered", "delayed"],
            },
        },
    ),
    _tool(
        "kb_search",
        "Search the knowledge base (product specs, allergens, shelf life, returns/quality policies, price list). Returns whole documents with their DOC ids.",
        {"query": _STR, "k": {"type": "integer"}},
        ["query"],
    ),
    _tool(
        "get_kb_document",
        "Fetch one KB document by id (DOC-###) in full.",
        {"doc_id": _STR},
        ["doc_id"],
    ),
    _tool(
        "generate_artifact",
        "Create a DOWNLOADABLE file (pdf/docx/pptx/xlsx) ONLY when the question explicitly asks for that format. Returns artifact_url to pass to final_answer. For HTML/markdown decks do NOT use this - put the HTML inline in final_answer's answer.",
        {
            "format": {"type": "string", "enum": ["pdf", "docx", "pptx", "xlsx"]},
            "title": _STR,
            "sections": {
                "type": "array",
                "description": "ordered sections/slides",
                "items": {
                    "type": "object",
                    "properties": {"heading": _STR, "body": _STR},
                },
            },
            "table": {
                "type": "array",
                "description": "rows for xlsx (array of arrays)",
                "items": {"type": "array", "items": {}},
            },
            "filename": _STR,
        },
        ["format", "title"],
    ),
    _tool(
        "final_answer",
        "Call this LAST with the finished answer. verticale = the dominant source. Set artifact_url only if you created a downloadable file.",
        {
            "answer": _STR,
            "verticale": {"type": "string", "enum": ["crm", "erp", "calls", "kb"]},
            "artifact_url": _STR,
        },
        ["answer", "verticale"],
    ),
]


def tool_json(payload: dict, max_chars: int = 12000) -> str:
    """Serialize a tool payload for the model, capped to keep latency/tokens sane."""
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) > max_chars:
        text = text[:max_chars] + ' ...TRUNCATED (refine your filters / use search)"}'
    return text
