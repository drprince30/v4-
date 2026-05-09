
from __future__ import annotations
from http.server import BaseHTTPRequestHandler, HTTPServer
import json, math, time, urllib.parse

METRICS={"DCGM_FI_DEV_GPU_TEMP","DCGM_FI_DEV_POWER_USAGE","DCGM_FI_DEV_GPU_UTIL","DCGM_FI_DEV_MEM_COPY_UTIL","DCGM_FI_DEV_CLOCK_THROTTLE_REASONS","DCGM_FI_DEV_ECC_DBE_VOL_TOTAL","DCGM_FI_DEV_PCIE_REPLAY_COUNTER","DCGM_FI_DEV_XID_ERRORS","DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL"}
def val(m,node,gpu,t):
    phase=t/60+node*.7+gpu*.4; hot=1 if node in [4,5,6,7] else 0; spike=1 if (int(t/60)%80)>35 else 0
    if m=="DCGM_FI_DEV_GPU_TEMP": return 56+6*math.sin(phase/5)+hot*8+spike*hot*18+gpu*.8
    if m=="DCGM_FI_DEV_POWER_USAGE": return 180+25*math.sin(phase/3)+hot*30+spike*hot*80
    if m=="DCGM_FI_DEV_GPU_UTIL": return max(0,min(100,45+20*math.sin(phase/4)+spike*40))
    if m=="DCGM_FI_DEV_MEM_COPY_UTIL": return max(0,min(100,35+18*math.sin(phase/6)+spike*25))
    if m=="DCGM_FI_DEV_CLOCK_THROTTLE_REASONS": return 1 if val("DCGM_FI_DEV_GPU_TEMP",node,gpu,t)>90 else 0
    if m=="DCGM_FI_DEV_ECC_DBE_VOL_TOTAL": return 1 if node==5 and spike else 0
    if m=="DCGM_FI_DEV_PCIE_REPLAY_COUNTER": return 5 if node==6 and spike else 0
    if m=="DCGM_FI_DEV_XID_ERRORS": return 1 if node==7 and spike else 0
    if m=="DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL": return 8 if node in [6,7] and spike else 0
    return 0
class H(BaseHTTPRequestHandler):
    def sendj(self,p):
        d=json.dumps(p).encode(); self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(d))); self.end_headers(); self.wfile.write(d)
    def do_GET(self):
        u=urllib.parse.urlparse(self.path); qs=urllib.parse.parse_qs(u.query)
        if u.path=="/api/v1/query": return self.sendj({"status":"success","data":{"result":[]}})
        if u.path!="/api/v1/query_range": self.send_response(404); self.end_headers(); return
        m=qs.get("query",[""])[0]; start=float(qs.get("start",[time.time()-3600])[0]); end=float(qs.get("end",[time.time()])[0]); step_raw=qs.get("step",["60s"])[0]; step=int(float(step_raw[:-1])) if step_raw.endswith("s") else int(float(step_raw))
        result=[]
        if m in METRICS:
            for node in range(12):
                for gpu in range(4):
                    vals=[]; ts=start
                    while ts<=end:
                        vals.append([ts,str(round(val(m,node,gpu,ts),3))]); ts+=step
                    result.append({"metric":{"__name__":m,"Hostname":f"gpu-node-{node:02d}","gpu":str(gpu),"instance":f"gpu-node-{node:02d}:9400"},"values":vals})
        self.sendj({"status":"success","data":{"resultType":"matrix","result":result}})
if __name__=="__main__":
    print("Mock Prometheus on :9090"); HTTPServer(("0.0.0.0",9090),H).serve_forever()
