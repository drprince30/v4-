
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import time, requests, pandas as pd

@dataclass(frozen=True)
class MetricMap:
    temp_c: str="DCGM_FI_DEV_GPU_TEMP"
    power_w: str="DCGM_FI_DEV_POWER_USAGE"
    gpu_util: str="DCGM_FI_DEV_GPU_UTIL"
    mem_util: str="DCGM_FI_DEV_MEM_COPY_UTIL"
    throttle_flag: str="DCGM_FI_DEV_CLOCK_THROTTLE_REASONS"
    ecc_errors: str="DCGM_FI_DEV_ECC_DBE_VOL_TOTAL"
    pcie_replay: str="DCGM_FI_DEV_PCIE_REPLAY_COUNTER"
    xid_errors: str="DCGM_FI_DEV_XID_ERRORS"
    nvlink_crc: str="DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL"

def _rows(name, series):
    labels=series.get("metric",{})
    node=labels.get("Hostname") or labels.get("node") or labels.get("instance") or "unknown"
    gpu=labels.get("gpu") or labels.get("GPU") or labels.get("device") or "0"
    out=[]
    for ts,val in series.get("values",[]):
        try: v=float(val)
        except Exception: continue
        out.append({"timestamp_unix":float(ts),"timestamp":time.strftime("t%Y%m%d%H%M%S",time.gmtime(float(ts))),"node_id":str(node).split(":")[0],"gpu_id":str(gpu),name:v})
    return out

def query_range(url, query, start, end, step="60s"):
    r=requests.get(f"{url.rstrip('/')}/api/v1/query_range",params={"query":query,"start":start,"end":end,"step":step},timeout=20)
    r.raise_for_status(); p=r.json()
    if p.get("status")!="success": raise RuntimeError(str(p))
    return p.get("data",{}).get("result",[])

def fetch_prometheus(url, minutes=60, step="60s", metric_map=MetricMap()):
    end=time.time(); start=end-minutes*60
    frames=[]
    for name,query in metric_map.__dict__.items():
        try: res=query_range(url,query,start,end,step)
        except Exception: continue
        rows=[]
        for s in res: rows.extend(_rows(name,s))
        if rows: frames.append(pd.DataFrame(rows))
    if not frames: raise RuntimeError("No Prometheus metrics returned.")
    keys=["timestamp_unix","timestamp","node_id","gpu_id"]
    merged=frames[0]
    for f in frames[1:]: merged=merged.merge(f,on=keys,how="outer")
    if "throttle_flag" in merged: merged["throttle_flag"]=(pd.to_numeric(merged["throttle_flag"],errors="coerce").fillna(0)>0).astype(int)
    return merged.drop(columns=["timestamp_unix"]).sort_values(["timestamp","node_id","gpu_id"]).reset_index(drop=True)

def test_prometheus(url):
    try:
        r=requests.get(f"{url.rstrip('/')}/api/v1/query",params={"query":"up"},timeout=8); r.raise_for_status()
        return {"ok":True,"response":r.json()}
    except Exception as e: return {"ok":False,"error":str(e)}
