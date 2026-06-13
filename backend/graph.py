"""Knowledge graph for the UI: customers -> products -> raw materials -> suppliers.

Built from real API data, bounded (a sample of finished SKUs and their BOM
chains) and cached in-process so the UI loads instantly and we don't re-hit the
metered API on every page view.
"""

from __future__ import annotations

import tools

NUM_PRODUCTS = 16  # finished SKUs to expand (keeps the graph connected, not huge)
_CACHE: dict | None = None


def _safe(fn, default):
    try:
        return fn()
    except Exception:
        return default


def build_graph() -> dict:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def node(node_id: str, label: str, group: str, title: str = "") -> None:
        if node_id and node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "label": label,
                "group": group,
                "title": title or label,
            }

    def edge(src: str, dst: str, label: str) -> None:
        if src in nodes and dst in nodes:
            edges.append({"from": src, "to": dst, "label": label})

    customers = _safe(lambda: tools.search_customers(all_pages=True)["customers"], [])
    suppliers = _safe(lambda: tools.list_suppliers(all_pages=True)["suppliers"], [])
    raw = _safe(
        lambda: tools.list_inventory(type="raw_material", all_pages=True)["inventory"],
        [],
    )
    finished = _safe(
        lambda: tools.list_inventory(type="finished_good", all_pages=True)["inventory"],
        [],
    )

    cust_name = {c["id"]: c.get("company_name", c["id"]) for c in customers}
    supp_name = {s["id"]: s.get("name", s["id"]) for s in suppliers}
    raw_by_sku = {r["sku"]: r for r in raw}

    product_skus = [f["sku"] for f in finished[:NUM_PRODUCTS]]
    product_set = set(product_skus)
    for f in finished[:NUM_PRODUCTS]:
        node(f["sku"], f["sku"], "product", f.get("description", f["sku"]))

    used_suppliers: set[str] = set()
    for sku in product_skus:
        bom = _safe(lambda s=sku: tools.get_bom(s), {"found": False})
        if not bom.get("found"):
            continue
        for comp in bom["bom"].get("components", []):
            rsku = comp.get("raw_sku")
            if not rsku:
                continue
            rinfo = raw_by_sku.get(rsku, {})
            label = rinfo.get("description") or comp.get("description") or rsku
            node(rsku, rsku, "material", label)
            edge(sku, rsku, "uses")
            sup = rinfo.get("supplier_id")
            if sup:
                node(sup, supp_name.get(sup, sup), "supplier", supp_name.get(sup, sup))
                edge(rsku, sup, "supplied by")
                used_suppliers.add(sup)

    # connect customers via a sample of orders (one page) to keep it bounded
    orders = _safe(lambda: tools.list_orders(all_pages=False)["orders"], [])
    for o in orders:
        cid = o.get("customer_id")
        for line in o.get("lines", []):
            sku = line.get("sku")
            if sku in product_set and cid:
                node(cid, cust_name.get(cid, cid), "customer", cust_name.get(cid, cid))
                edge(cid, sku, "orders")

    return {"nodes": list(nodes.values()), "edges": edges}


def get_graph(refresh: bool = False) -> dict:
    global _CACHE
    if _CACHE is None or refresh:
        _CACHE = build_graph()
    return _CACHE
