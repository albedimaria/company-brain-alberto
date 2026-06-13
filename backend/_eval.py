import sys
import time

import agent

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # noqa: E402 - dev harness

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

idxs = [int(a) for a in sys.argv[1:]] or list(range(1, len(QUESTIONS) + 1))
for i in idxs:
    q = QUESTIONS[i - 1]
    t0 = time.time()
    r = agent.run(q)
    dt = time.time() - t0
    print(
        f"\n===== Q{i} ({dt:.1f}s) verticale={r['verticale']} sources={r['sources']} artifact={r['artifact_url']}"
    )
    ans = r["answer"]
    print(ans[:1200])
