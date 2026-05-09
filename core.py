
"""
FCCT AI Infrastructure Cascade Defense V4 - Core math and data types.

V4 pivot:
- not generic GPU monitoring
- specialized cascade defense for physical/fabric/scheduler layers
- separates fleet stress from true topology-supported cascade risk
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Iterable, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FCCTConfig:
    coherence_radius: int = 1
    growth_height: int = 1
    max_scale: int = 5
    cascade_trigger: float = 0.58
    fleet_stress_trigger: float = 0.60
    min_risky_scale: int = 2
    edge_weight_floor: float = 0.25


@dataclass(frozen=True)
class ScaleCalibration:
    metric: str
    thresholds: Tuple[float, float, float, float, float]
    method: str
    explanation: str
    higher_is_riskier: bool = True


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    weight: float
    graph: str
    relation: str


@dataclass(frozen=True)
class GraphRisk:
    graph: str
    fleet_stress: float
    cascade_risk: float
    headroom: float
    forcing: float
    decay: float
    net_pressure: float
    max_scale: int
    coherent_pairs: int
    risky_components: List[str]
    classification_hint: str


def solve_lambda_r(r: int, h: int = 1, tol: float = 1e-10, max_iter: int = 200) -> float:
    if r < 0:
        raise ValueError("r must be >= 0")
    if h < 1:
        raise ValueError("h must be >= 1")
    lo = 1.0 + 1e-12
    hi = 2.0
    def f(x: float) -> float:
        return x ** (r + h) - x ** r - 1.0
    while f(hi) < 0:
        hi *= 2.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return float((lo + hi) / 2.0)


def component_id(node_id: object, gpu_id: object = "0") -> str:
    return f"{node_id}::gpu{gpu_id}"


def ensure_component_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "component_id" not in out.columns:
        if "gpu_id" not in out.columns:
            out["gpu_id"] = "0"
        out["component_id"] = [
            component_id(n, g) for n, g in zip(out["node_id"].astype(str), out["gpu_id"].astype(str))
        ]
    return out


def assign_scale(values: pd.Series, cal: ScaleCalibration, max_scale: int = 5) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    bins = np.array(cal.thresholds, dtype=float)
    raw = np.digitize(vals.to_numpy(), bins, right=False)
    if not cal.higher_is_riskier:
        raw = max_scale - raw
    raw = np.where(np.isfinite(vals.to_numpy()), raw, 0)
    return pd.Series(np.clip(raw, 0, max_scale), index=values.index, dtype="int64")


def scale_counts(scales: Iterable[int], max_scale: int = 5) -> np.ndarray:
    counts = np.zeros(max_scale + 1, dtype=int)
    for s in scales:
        counts[int(np.clip(int(s), 0, max_scale))] += 1
    return counts


def energy(scales: Iterable[int], lam: float, max_scale: int = 5) -> float:
    counts = scale_counts(scales, max_scale)
    powers = np.array([lam ** k for k in range(max_scale + 1)], dtype=float)
    return float(np.dot(counts, powers))


def normalized_energy(scales: Iterable[int], lam: float, max_scale: int = 5) -> float:
    scale_list = list(scales)
    if not scale_list:
        return 0.0
    return float(energy(scale_list, lam, max_scale) / (len(scale_list) * (lam ** max_scale)))


def combine_scales(row: pd.Series, scale_cols: Dict[str, float], max_scale: int = 5) -> int:
    """
    Weighted + max-safety combination.
    A single critical hardware signal can still push the final scale high.
    """
    weighted = 0.0
    total = 0.0
    max_s = 0
    for col, w in scale_cols.items():
        if col not in row:
            continue
        s = int(row[col]) if pd.notna(row[col]) else 0
        max_s = max(max_s, s)
        weighted += float(w) * s
        total += float(w)
    if total <= 0:
        return 0
    avg = int(round(weighted / total))
    if max_s >= 5:
        return 5
    if max_s >= 4 and avg >= 2:
        return max(3, avg)
    return int(np.clip(max(avg, max_s - 1), 0, max_scale))


def coherent_pair_count(scales: Dict[str, int], edges: List[Edge], r: int, min_scale: int, edge_floor: float) -> Tuple[int, List[str]]:
    count = 0
    risky = set()
    for e in edges:
        if e.weight < edge_floor:
            continue
        if e.src not in scales or e.dst not in scales:
            continue
        a = int(scales[e.src])
        b = int(scales[e.dst])
        if a >= min_scale and b >= min_scale and abs(a - b) <= r:
            count += 1
            risky.add(e.src)
            risky.add(e.dst)
    return count, sorted(risky)


def projected_scales(scales: Dict[str, int], edges: List[Edge], r: int, max_scale: int, edge_floor: float) -> Dict[str, int]:
    proj = dict(scales)
    for e in edges:
        if e.weight < edge_floor:
            continue
        if e.src not in scales or e.dst not in scales:
            continue
        a = int(scales[e.src])
        b = int(scales[e.dst])
        if a > 0 and b > 0 and abs(a - b) <= r:
            jump = min(max(a, b) + 1, max_scale)
            if e.weight >= 0.75 or max(a, b) >= 3:
                proj[e.src] = max(proj[e.src], jump)
                proj[e.dst] = max(proj[e.dst], jump)
    return proj


def compute_graph_risk(
    frame: pd.DataFrame,
    edges: List[Edge],
    graph_name: str,
    config: FCCTConfig,
    prev_score: Optional[float] = None,
) -> GraphRisk:
    f = ensure_component_id(frame)
    if f.empty or "scale" not in f.columns:
        return GraphRisk(graph_name, 0, 0, 1, 0, 0, 0, 0, 0, [], "no_data")

    scale_map = f.set_index("component_id")["scale"].astype(int).to_dict()
    lam = solve_lambda_r(config.coherence_radius, config.growth_height)
    current_score = normalized_energy(scale_map.values(), lam, config.max_scale)
    proj = projected_scales(scale_map, edges, config.coherence_radius, config.max_scale, config.edge_weight_floor)
    cascade_score = normalized_energy(proj.values(), lam, config.max_scale)

    coherent_pairs, risky = coherent_pair_count(
        scale_map, edges, config.coherence_radius, config.min_risky_scale, config.edge_weight_floor
    )
    max_scale = int(max(scale_map.values())) if scale_map else 0
    fleet_stress = float(np.mean([s / config.max_scale for s in scale_map.values()])) if scale_map else 0.0
    forcing = max(0.0, cascade_score - (prev_score if prev_score is not None else current_score))
    decay = max(0.0, (prev_score if prev_score is not None else current_score) - cascade_score)
    net = max(0.0, forcing - decay)
    headroom = max(0.0, 1.0 - cascade_score)

    if coherent_pairs > 0 and cascade_score >= config.cascade_trigger:
        hint = "topology_supported_cascade"
    elif fleet_stress >= config.fleet_stress_trigger and coherent_pairs == 0:
        hint = "fleet_stress_not_cascade"
    elif max_scale >= 5 and coherent_pairs == 0:
        hint = "isolated_point_failure"
    else:
        hint = "watch"

    return GraphRisk(
        graph=graph_name,
        fleet_stress=fleet_stress,
        cascade_risk=cascade_score,
        headroom=headroom,
        forcing=forcing,
        decay=decay,
        net_pressure=net,
        max_scale=max_scale,
        coherent_pairs=coherent_pairs,
        risky_components=risky,
        classification_hint=hint,
    )
