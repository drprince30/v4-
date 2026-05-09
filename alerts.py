
from __future__ import annotations
import pandas as pd
import requests


def grouped_alerts(defense: pd.DataFrame, layer_timeline: pd.DataFrame, trigger: float, cooldown: int=8):
    rows=[]; last=-10**9
    for i,r in defense.reset_index(drop=True).iterrows():
        score=float(r["final_defense_score"])
        if score < trigger or i-last < cooldown: continue
        ts=r["timestamp"]
        layers=layer_timeline[layer_timeline["timestamp"]==ts].sort_values("cascade_risk", ascending=False)
        top=layers.iloc[0]["graph"] if not layers.empty else "unknown"
        sev="critical" if score>=.80 else "high" if score>=.65 else "warning"
        rows.append({"timestamp":ts,"severity":sev,"top_layer":top,"score":round(score,3),"message":f"{sev.upper()} AI infra cascade risk at {ts}: score={score:.3f}, top_layer={top}. Review recommendations."})
        last=i
    return pd.DataFrame(rows)


def payload(alerts: pd.DataFrame):
    if alerts.empty: return {"text":"AI Infrastructure Cascade Defense: no active cascade alerts."}
    return {"text":"AI Infrastructure Cascade Defense alerts:\n" + "\n".join("- "+str(m) for m in alerts["message"].tolist())}


def post_webhook(url, payload_obj, timeout=10):
    try:
        r=requests.post(url,json=payload_obj,timeout=timeout)
        return {"ok":r.status_code<400,"status_code":r.status_code,"text":r.text[:500]}
    except Exception as e:
        return {"ok":False,"error":str(e)}
