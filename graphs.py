
from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from core import Edge, ensure_component_id


def merge_topology(telemetry: pd.DataFrame, topology: pd.DataFrame) -> pd.DataFrame:
    t = telemetry.copy()
    topo = topology.copy()
    t["node_id"] = t["node_id"].astype(str)
    topo["node_id"] = topo["node_id"].astype(str)
    cols = ["node_id"] + [c for c in topo.columns if c != "node_id"]
    out = t.merge(topo[cols].drop_duplicates("node_id"), on="node_id", how="left")
    for c in ["rack_id", "zone", "cooling_loop", "power_domain", "fabric_group", "scheduler_pool", "tenant_id"]:
        if c not in out.columns:
            out[c] = "unknown"
        out[c] = out[c].fillna("unknown").astype(str)
    return out


def topology_summary(topology: pd.DataFrame) -> Dict[str, int]:
    return {c: int(topology[c].nunique()) for c in ["node_id", "rack_id", "zone", "cooling_loop", "power_domain", "fabric_group", "scheduler_pool"] if c in topology.columns}


def _add(edges, a, b, weight, graph, relation):
    if a == b:
        return
    key = tuple(sorted((a, b)) + [graph, relation])
    # Simpler: append; dedupe later by src/dst/graph highest weight.
    edges.append(Edge(str(a), str(b), float(weight), graph, relation))


def build_multigraph_edges(topology: pd.DataFrame, components: pd.DataFrame, same_group_limit: int = 3) -> Dict[str, List[Edge]]:
    comps = ensure_component_id(components)
    topo = topology.copy()
    topo["node_id"] = topo["node_id"].astype(str)
    comps["node_id"] = comps["node_id"].astype(str)
    node_to_components = comps.groupby("node_id")["component_id"].apply(list).to_dict()
    topo_idx = topo.drop_duplicates("node_id").set_index("node_id")
    graphs = {k: [] for k in ["thermal", "cooling", "power", "fabric", "scheduler", "tenant"]}

    def connect_nodes(nodes, graph, weight, relation, limit=None):
        nodes = list(nodes)
        for i, a in enumerate(nodes):
            tail = nodes[i+1:] if limit is None else nodes[i+1:i+1+limit]
            for b in tail:
                for ca in node_to_components.get(a, []):
                    for cb in node_to_components.get(b, []):
                        _add(graphs[graph], ca, cb, weight, graph, relation)

    # Same node edges exist in all relevant graphs.
    for node, ids in node_to_components.items():
        ids = list(sorted(set(ids)))
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                for graph in graphs:
                    _add(graphs[graph], ids[i], ids[j], 1.0, graph, "same_node")

    # Physical row/col adjacency for thermal.
    if {"row", "col"}.issubset(topo_idx.columns):
        rows = pd.to_numeric(topo_idx["row"], errors="coerce")
        cols = pd.to_numeric(topo_idx["col"], errors="coerce")
        nodes = list(topo_idx.index)
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                a, b = nodes[i], nodes[j]
                if pd.isna(rows.loc[a]) or pd.isna(rows.loc[b]) or pd.isna(cols.loc[a]) or pd.isna(cols.loc[b]):
                    continue
                manhattan = abs(rows.loc[a] - rows.loc[b]) + abs(cols.loc[a] - cols.loc[b])
                if manhattan == 1:
                    connect_nodes([a, b], "thermal", 0.90, "adjacent_physical")
                if manhattan == 2:
                    connect_nodes([a, b], "thermal", 0.45, "near_physical")

    for col, graph, weight in [
        ("cooling_loop", "cooling", 0.85),
        ("power_domain", "power", 0.85),
        ("fabric_group", "fabric", 0.80),
        ("scheduler_pool", "scheduler", 0.60),
        ("tenant_id", "tenant", 0.55),
        ("rack_id", "thermal", 0.50),
    ]:
        if col in topo_idx.columns:
            for _, group in topo_idx.groupby(col).groups.items():
                connect_nodes(list(group), graph, weight, f"same_{col}", limit=same_group_limit)

    # Dedupe highest weight per src/dst/graph.
    deduped: Dict[str, List[Edge]] = {}
    for graph, edges in graphs.items():
        best = {}
        for e in edges:
            key = tuple(sorted((e.src, e.dst)))
            if key not in best or e.weight > best[key].weight:
                best[key] = e
        deduped[graph] = list(best.values())
    return deduped


def topology_health(topology: pd.DataFrame, components: pd.DataFrame, graphs: Dict[str, List[Edge]]) -> pd.DataFrame:
    rows = []
    comp_count = max(1, len(components))
    topo_nodes = set(topology["node_id"].astype(str)) if "node_id" in topology.columns else set()
    comp_nodes = set(components["node_id"].astype(str)) if "node_id" in components.columns else set()
    missing = comp_nodes - topo_nodes
    for graph, edges in graphs.items():
        density = len(edges) / comp_count
        warnings = []
        if missing:
            warnings.append(f"{len(missing)} telemetry nodes missing in topology")
        if len(edges) == 0:
            warnings.append("no edges")
        if density > 20:
            warnings.append("graph too dense")
        conf = max(0, min(1, 1 - 0.15*len(warnings) - max(0, density-10)*0.02))
        rows.append({"graph": graph, "components": comp_count, "edges": len(edges), "edge_per_component": round(float(density),3), "warnings": "; ".join(warnings) if warnings else "ok", "confidence": round(float(conf),3)})
    return pd.DataFrame(rows)
