"""The agent loop behind POST /ask.

Deterministic orchestration: the LLM only decides which tools to call and writes
prose; all facts come from tool outputs, all arithmetic is done in Python (see
tools.py). Grounding rules are enforced in the system prompt and by capping the
loop. Source ids are accumulated for provenance; verticale is inferred from them.
"""

from __future__ import annotations

import json
import time

import config
from tools import TOOLS, run_tool, tool_json

MAX_STEPS = 6
LLM_RETRIES = 2
TIMEOUT_SEC = 25

SYSTEM_PROMPT = """You are the Company Brain of Al Dente S.r.l., a dry-pasta maker.
You answer questions about the company by calling tools, then composing a precise answer.

GROUNDING (critical - hallucinations are penalized):
- State ONLY facts present in tool outputs from THIS conversation. Never guess, estimate or infer numbers.
- NEVER invent or guess an id (customer, call, lot, SKU, supplier, document). If you do not have an id, FIND it first with a search/list tool. Do not call a tool with a made-up id.
- To resolve a reference like "the last / latest / most recent / that call (or order, lot)", first list the items (e.g. list_calls with the customer_id) and pick the one with the most recent date. Never assume an id.
- Some questions are TRAPS. If an entity does not exist (get_customer returns found=false) or a figure is not stored in any source (e.g. profit margin / cost), say so explicitly: name what is missing ("there is no customer named X", "cost/margin is not stored in the sources"). A specific "not available" beats any invented value.
- When a phone call and an official document disagree, the official document (price list, policy, spec) is authoritative.

ORCHESTRATION (chains and aggregates):
- Multi-step questions: follow the chain with successive tool calls (e.g. customer -> calls -> lot, or SKU -> get_bom -> list_inventory raw material -> supplier_name). Read ids from one result and feed them to the next call.
- For "grouped by <X>" you MUST pass group_by (use group_by='customer_channel' for GDO/distributor/horeca) and report EACH group's value separately. Never collapse a grouped question into a single total.
- For "how many / total" read the tool's PRECOMPUTED fields and quote them EXACTLY: count, sums_eur, open_count, open_value_eur (open = qualification+negotiation), by_customer_channel. The total value of OPEN opportunities is open_value_eur. Never add up, recompute or redistribute rows yourself, and never report a stage subtotal as the overall total.
- For "how many calls about X" use count_calls. For transcripts use search_transcript with a search term; never request whole transcripts.

OUTPUT:
- Be concise and factual; include the concrete numbers, ids and names you found.
- NEVER use **bold** markdown or any markdown emphasis in answers. Plain text only (no **, no __, no # headings).
- Structure every answer as: one clear summary sentence first, then details on the next line(s). Example:
  "Primato Supermercati ha 4 opportunità aperte per 740.000 €.
  Dettaglio: 446.000 € in negoziazione (2), 294.000 € in qualificazione (2). Canale: GDO."
- Before generating a deck/report/artifact about an entity, FIRST gather every underlying fact with tools (profile, opportunities/deals, orders, production lots, calls). Then compose. Every section MUST contain the real fetched data; never emit empty sections or placeholders. If a section genuinely has no data, state it ("no complaints on record").
- If the question asks for an HTML or markdown deck/slides, put the full HTML inline in the answer.
- If it explicitly asks for a downloadable docx/pptx/pdf/xlsx file, call generate_artifact and pass its artifact_url to final_answer.
- ALWAYS finish by calling final_answer with the answer and the dominant verticale (crm / erp / calls / kb).
"""

_VERTICALE = {"crm": "crm", "calls": "calls", "erp": "erp"}
_LANG = {
    "it": "Italian (italiano)",
    "en": "English",
    "es": "Spanish (español)",
}


def _answer_lang(lang: str | None) -> str:
    return lang if lang in _LANG else "en"


def _system_prompt(lang: str | None) -> str:
    code = _answer_lang(lang)
    label = _LANG[code]
    return (
        SYSTEM_PROMPT
        + f"\n\nLANGUAGE: Write every final answer in {label}. "
        f"Keep ids, SKUs, lot codes and proper nouns as returned by tools; "
        f"all explanatory sentences must be in {label}."
    )


def _infer_verticale(sources: list[str]) -> str:
    counts = {"crm": 0, "erp": 0, "calls": 0, "kb": 0}
    for s in sources:
        if s.startswith("crm/"):
            counts["crm"] += 1
        elif s.startswith("calls"):
            counts["calls"] += 1
        elif s.startswith("erp/"):
            counts["erp"] += 1
        elif s.startswith("DOC-"):
            counts["kb"] += 1
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] else "crm"


def _content(msg) -> str:
    return (
        getattr(msg, "content", None) or getattr(msg, "reasoning_content", None) or ""
    ).strip()


def _chat(messages: list[dict], tool_choice: str = "auto"):
    last_exc: Exception | None = None
    for attempt in range(LLM_RETRIES + 1):
        try:
            return config.llm().chat.completions.create(
                model=config.MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice=tool_choice,
                temperature=0,
            )
        except Exception as exc:  # transient provider errors: backoff then retry
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def _timed_out(start: float) -> bool:
    return time.time() - start > TIMEOUT_SEC


def _force_answer(
    messages: list[dict],
    sources: list[str],
    artifact_url: str | None,
    reason: str,
    lang: str | None = None,
) -> dict:
    code = _answer_lang(lang)
    label = _LANG[code]
    messages.append(
        {
            "role": "user",
            "content": f"{reason} Stop calling tools. Answer now using only what the tools returned. "
            f"If something is missing, say it is not available. Plain text only, no markdown bold. "
            f"Write the answer in {label}.",
        }
    )
    final = _chat(messages, tool_choice="none")
    answer = (
        _content(final.choices[0].message)
        or "I cannot answer this from the available sources."
    )
    return _result(answer, sources, artifact_url)


def run(question: str, lang: str | None = None) -> dict:
    """Run the agent loop and return the /ask payload."""
    messages: list[dict] = [
        {"role": "system", "content": _system_prompt(lang)},
        {"role": "user", "content": question},
    ]
    sources: list[str] = []
    artifact_url: str | None = None
    start = time.time()

    try:
        for step in range(MAX_STEPS):
            if _timed_out(start):
                return _force_answer(
                    messages,
                    sources,
                    artifact_url,
                    "Time limit reached.",
                    lang,
                )
            last_step = step == MAX_STEPS - 1
            resp = _chat(messages, tool_choice="auto")
            msg = resp.choices[0].message
            calls = msg.tool_calls or []

            if not calls:
                answer = (
                    _content(msg) or "I cannot answer this from the available sources."
                )
                return _result(answer, sources, artifact_url)

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.function.name,
                                "arguments": c.function.arguments,
                            },
                        }
                        for c in calls
                    ],
                }
            )

            for call in calls:
                if _timed_out(start):
                    return _force_answer(
                        messages,
                        sources,
                        artifact_url,
                        "Time limit reached.",
                        lang,
                    )
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if name == "final_answer":
                    answer = (
                        args.get("answer")
                        or "I cannot answer this from the available sources."
                    )
                    vert = args.get("verticale") or _infer_verticale(sources)
                    url = args.get("artifact_url") or artifact_url
                    return _result(answer, sources, url, verticale=vert)

                payload, src = run_tool(name, args)
                sources.extend(src)
                if name == "generate_artifact" and payload.get("artifact_url"):
                    artifact_url = payload["artifact_url"]
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": tool_json(payload),
                    }
                )

            if last_step:
                return _force_answer(
                    messages,
                    sources,
                    artifact_url,
                    "Tool budget exhausted.",
                    lang,
                )
    except Exception as exc:  # never 5xx: honest abstention keeps the contract
        return _result(
            f"I cannot answer this right now due to a system error ({type(exc).__name__}).",
            sources,
            artifact_url,
        )

    return _result(
        "I cannot answer this from the available sources.", sources, artifact_url
    )


def _unwrap_fence(answer: str) -> str:
    """If the model wrapped an HTML/markdown artifact in a ``` code fence,
    return the inner content so it renders as a deck, not as code."""
    s = answer.strip()
    if s.startswith("```") and s.endswith("```") and len(s) > 6:
        inner = s[3:-3]
        inner = inner.split("\n", 1)[1] if "\n" in inner[:12] else inner
        return inner.strip()
    return answer


def _result(
    answer: str,
    sources: list[str],
    artifact_url: str | None,
    verticale: str | None = None,
) -> dict:
    seen: list[str] = []
    for s in sources:
        if s not in seen:
            seen.append(s)
    return {
        "answer": _unwrap_fence(answer),
        "sources": seen,
        "verticale": verticale or _infer_verticale(seen),
        "artifact_url": artifact_url,
    }
