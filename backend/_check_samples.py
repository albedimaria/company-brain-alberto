"""Run 12 sample questions against local agent and/or deployed /ask."""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request

import agent

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUESTIONS = [
    "How many open opportunities does Primato Supermercati S.p.A. (CUST-0132) have, and what is their total value?",
    "Is SKU PAS-PEN-500 (Penne Rigate n.73 - 500g box) below its minimum stock? Give the on-hand quantity.",
    "In the last call with NordSpesa S.p.A. (CUST-0137), what was the complaint and which lot did it concern?",
    "What is the shelf life (TMC) and the declared allergens for Spaghetti n.5 - 500g box (SKU PAS-SPA-500)?",
    "Does the complaint from that last NordSpesa S.p.A. call qualify for a return under the quality policy?",
    "Total value of opportunities in the negotiation stage, grouped by customer channel (GDO / distributor / horeca).",
    "What is the profit margin on lot LOT-2026-0658?",
    "What is the status of the order for Supermercati Bianchi?",
    "Generate a 4-slide HTML deck for the sales rep visiting Primato Supermercati S.p.A. (CUST-0132): profile, open deals, order/lot status, recent call complaints.",
    "Which semolina does SKU PAS-SPA-500 use (per its bill of materials), which supplier provides it, and is that raw material below minimum stock?",
    "Across ALL recorded calls (there are 80), count how many quality complaints concern the defect 'broken pasta'. Give the exact number.",
    "GranMercato S.p.A. asked about the price of Fusilli n.98 (PAS-FUS-500). A call mentions one figure and the official 2026 wholesale price list mentions another. Which is the correct list price, and why?",
]

DEPLOY = "https://company-brain-alberto.onrender.com"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def money(s: str) -> str:
    return re.sub(r"[^0-9]", "", s)


def check(i: int, answer: str) -> tuple[bool, str]:
    a = norm(answer)
    raw = answer
    if i == 1:
        ok = ("740" in money(a) or "740000" in money(a)) and re.search(r"\b4\b", a)
        return ok, "4 open ops, 740k EUR"
    if i == 2:
        ok = "462" in a and ("below" in a or "under" in a or "minimum" in a)
        return ok, "below min, 462 on hand"
    if i == 3:
        ok = "broken" in a and "lot-2026-0658" in a
        return ok, "broken pasta, LOT-2026-0658"
    if i == 4:
        ok = ("36" in a and "month" in a) and "gluten" in a
        return ok, "36 months, gluten"
    if i == 5:
        ok = ("yes" in a or "qualif" in a or "cover" in a or "replacement" in a or "credit" in a)
        return ok, "return qualifies"
    if i == 6:
        ok = (
            "3301000" in money(a) or "3,301,000" in raw or "3301" in money(a)
        ) and ("1931000" in money(a) or "1,931,000" in raw) and (
            "3040000" in money(a) or "3,040,000" in raw
        )
        return ok, "GDO 3.301M / dist 1.931M / horeca 3.040M"
    if i == 7:
        ok = any(
            x in a
            for x in (
                "not available",
                "not stored",
                "no profit",
                "cannot",
                "not in",
                "do not have",
                "don't have",
                "is not",
            )
        ) and "margin" in a
        return ok, "margin not available (trap)"
    if i == 8:
        ok = any(
            x in a
            for x in (
                "no customer",
                "not found",
                "does not exist",
                "doesn't exist",
                "not in the crm",
                "no order",
                "cannot find",
            )
        )
        return ok, "Supermercati Bianchi trap"
    if i == 9:
        ok = ("<!doctype" in a.lower() or "<html" in a.lower()) and "primato" in a
        return ok, "inline HTML deck"
    if i == 10:
        ok = "raw-sem-003" in a and "molino" in a
        ok = ok and not re.search(r"below.{0,40}minimum|under.{0,40}minimum", a)
        return ok, "RAW-SEM-003, Molino, not below min"
    if i == 11:
        ok = re.search(r"\b9\b", a) is not None and "broken" in a
        return ok, "9 broken-pasta calls"
    if i == 12:
        ok = "8.07" in raw or "8,07" in raw
        ok = ok and ("doc-015" in a or "price list" in a or "official" in a)
        return ok, "8.07 EUR, official list"
    return False, "?"


def ask_remote(q: str) -> tuple[float, dict]:
    data = json.dumps({"question": q}).encode()
    req = urllib.request.Request(
        f"{DEPLOY}/ask",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=35) as r:
        payload = json.loads(r.read().decode())
    return time.time() - t0, payload


def run_target(name: str, fn):
    print(f"\n{'=' * 60}\nTARGET: {name}\n{'=' * 60}")
    passed = 0
    rows = []
    for i, q in enumerate(QUESTIONS, 1):
        try:
            t0 = time.time()
            if fn is ask_remote:
                dt, r = ask_remote(q)
            else:
                r = agent.run(q)
                dt = time.time() - t0
            ans = r.get("answer", "")
            ok, expect = check(i, ans)
            if ok:
                passed += 1
            status = "PASS" if ok else "FAIL"
            rows.append((i, status, dt, expect, r.get("verticale"), ans[:200]))
            print(f"Q{i:02d} {status} {dt:5.1f}s vert={r.get('verticale')} | {expect}")
            if not ok:
                print(f"     -> {ans[:500]}")
        except Exception as exc:
            rows.append((i, "ERR", 0, str(exc), "", ""))
            print(f"Q{i:02d} ERR  | {exc}")
        time.sleep(1.2)
    print(f"\n{name}: {passed}/12 passed")
    return passed, rows


def main():
    targets = sys.argv[1:] or ["local", "deploy"]
    totals = {}
    for t in targets:
        if t == "local":
            totals[t] = run_target("LOCAL (agent.run)", None)[0]
        elif t in ("deploy", "remote"):
            totals[t] = run_target(f"DEPLOY ({DEPLOY})", ask_remote)[0]
        else:
            print(f"Unknown target: {t}")
    if len(totals) == 2:
        print(
            f"\nSUMMARY local={totals.get('local', '?')}/12 deploy={totals.get('deploy', totals.get('remote', '?'))}/12"
        )


if __name__ == "__main__":
    main()
