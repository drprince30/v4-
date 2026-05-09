
from __future__ import annotations
import pandas as pd


def validate_against_events(defense: pd.DataFrame, telemetry: pd.DataFrame, trigger: float, early_temp: float=70, late_temp: float=82):
    out = defense.copy()
    events = telemetry.groupby("timestamp")["throttle_flag"].sum().reset_index()
    events["event"] = (events["throttle_flag"] > 0).astype(int)
    out = out.merge(events[["timestamp","event"]], on="timestamp", how="left").fillna({"event":0})
    out["fcct_alert"] = (out["final_defense_score"] >= trigger).astype(int)

    def temp_baseline(th):
        rows = []
        for ts, g in telemetry.groupby("timestamp"):
            rows.append({"timestamp":ts, f"temp_{int(th)}_alert": int((pd.to_numeric(g["temp_c"], errors="coerce") >= th).any())})
        return pd.DataFrame(rows)
    out = out.merge(temp_baseline(early_temp), on="timestamp", how="left").merge(temp_baseline(late_temp), on="timestamp", how="left")

    def metrics(col):
        y = (out["event"] > 0).astype(int)
        p = (out[col] > 0).astype(int)
        tp = int(((y==1)&(p==1)).sum()); fp=int(((y==0)&(p==1)).sum()); fn=int(((y==1)&(p==0)).sum()); tn=int(((y==0)&(p==0)).sum())
        prec = tp/(tp+fp) if tp+fp else 0
        rec = tp/(tp+fn) if tp+fn else 0
        return {"alert":col,"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":round(prec,3),"recall":round(rec,3),"alert_frames":int(p.sum()),"event_frames":int(y.sum())}
    summary = pd.DataFrame([metrics("fcct_alert"), metrics(f"temp_{int(early_temp)}_alert"), metrics(f"temp_{int(late_temp)}_alert")])
    return out, summary
