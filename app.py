
from __future__ import annotations

import os, uuid
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from telemetry import SCENARIOS, generate_telemetry, load_telemetry_csv, load_topology_csv, validate_topology
from prometheus_connector import fetch_prometheus, test_prometheus
from graphs import merge_topology, topology_summary, topology_health
from calibration import build_calibrations, calibration_confidence
from quality import observability_report, observability_confidence
from core import FCCTConfig, solve_lambda_r
from layers import add_metric_scales, add_layer_scales, compute_multigraph_timelines, aggregate_defense_score, latest_layer_snapshot
from classifier import classify_failure
from recommendations import build_recommendations
from validation import validate_against_events
from alerts import grouped_alerts, payload, post_webhook
from report import generate_report

st.set_page_config(page_title="AI Infrastructure Cascade Defense V4", page_icon="🛡️", layout="wide")
st.title("FCCT AI Infrastructure Cascade Defense Engine V4")
st.caption("Multi-graph cascade defense: cooling, power, fabric, scheduler, tenant, thermal.")

with st.sidebar:
    mode=st.radio("Source",["Sample data","CSV upload","Prometheus/DCGM"],index=0)
    scenario=st.selectbox("Scenario",SCENARIOS,index=3,disabled=(mode!="Sample data"))
    steps=st.slider("Sample steps",80,360,180,disabled=(mode!="Sample data"))
    seed=st.number_input("Seed",value=13,step=1,disabled=(mode!="Sample data"))
    prom_url=st.text_input("Prometheus URL",value=os.getenv("PROMETHEUS_URL","http://localhost:9090"),disabled=(mode!="Prometheus/DCGM"))
    minutes=st.slider("Prometheus minutes",10,240,60,disabled=(mode!="Prometheus/DCGM"))
    step=st.selectbox("Prometheus step",["15s","30s","60s","120s"],index=2,disabled=(mode!="Prometheus/DCGM"))
    r=st.slider("Coherence radius r",0,4,1)
    trigger=st.slider("Cascade trigger",0.05,0.95,0.58)
    edge_floor=st.slider("Edge weight floor",0.0,1.0,0.25)
    min_scale=st.slider("Min risky scale",1,5,2)
    webhook=st.text_input("Webhook URL",value="")

# Load
try:
    if mode=="Sample data":
        telemetry, topology=generate_telemetry(steps=int(steps),seed=int(seed),scenario=scenario)
        source=f"sample:{scenario}"
    elif mode=="CSV upload":
        tf=st.file_uploader("Telemetry CSV",type=["csv"])
        tof=st.file_uploader("Topology CSV",type=["csv"])
        if tf is None:
            st.stop()
        telemetry=load_telemetry_csv(tf)
        if tof: topology=load_topology_csv(tof)
        else:
            nodes=sorted(telemetry["node_id"].astype(str).unique())
            topology=pd.DataFrame({"node_id":nodes,"rack_id":["rack"]*len(nodes),"row":[0]*len(nodes),"col":list(range(len(nodes))),"zone":["zone"]*len(nodes)})
            topology=validate_topology(topology)
        source="uploaded_csv"
    else:
        col1,col2=st.columns(2)
        with col1:
            if st.button("Test Prometheus"): st.json(test_prometheus(prom_url))
        with col2:
            fetch=st.button("Fetch Prometheus telemetry",type="primary")
        if not fetch and "prom_data" not in st.session_state: st.stop()
        if fetch:
            telemetry=fetch_prometheus(prom_url,minutes=int(minutes),step=step); st.session_state["prom_data"]=telemetry
        else: telemetry=st.session_state["prom_data"]
        nodes=sorted(telemetry["node_id"].astype(str).unique())
        topology=pd.DataFrame({"node_id":nodes,"rack_id":[f"rack-{i//4+1}" for i in range(len(nodes))],"row":[i//4 for i in range(len(nodes))],"col":[i%4 for i in range(len(nodes))],"zone":["zone-A" if i%4<2 else "zone-B" for i in range(len(nodes))],"cooling_loop":[f"loop-{1 if i%4<2 else 2}" for i in range(len(nodes))],"power_domain":[f"pdu-{i//4+1}" for i in range(len(nodes))],"fabric_group":[f"fabric-{(i%4)//2}" for i in range(len(nodes))],"scheduler_pool":["pool"]*len(nodes),"tenant_id":[f"tenant-{i%3}" for i in range(len(nodes))]})
        topology=validate_topology(topology); source=f"prometheus:{prom_url}"
except Exception as e:
    st.error(f"Load failed: {e}"); st.exception(e); st.stop()

try:
    merged=merge_topology(telemetry,topology)
    cfg=FCCTConfig(coherence_radius=int(r),cascade_trigger=float(trigger),edge_weight_floor=float(edge_floor),min_risky_scale=int(min_scale))
    obs=observability_report(merged); obs_conf=observability_confidence(obs)
    cals=build_calibrations(merged); cal_conf=calibration_confidence(merged,cals)
    scaled=add_layer_scales(add_metric_scales(merged,cals,cfg),cfg)
    layer_timeline, graph_edges = compute_multigraph_timelines(scaled,topology,cfg)
    defense=aggregate_defense_score(layer_timeline,cfg)
    topo_health=topology_health(topology,scaled[["node_id","gpu_id","component_id"]].drop_duplicates(),graph_edges)
    peak_ts=defense.sort_values("final_defense_score",ascending=False).iloc[0]["timestamp"]
    snapshot=scaled[scaled["timestamp"]==peak_ts]
    shape=classify_failure(layer_timeline,defense,snapshot)
    top_layers=layer_timeline[layer_timeline["timestamp"]==peak_ts].sort_values("cascade_risk",ascending=False)
    recs=build_recommendations(shape,defense[defense["timestamp"]==peak_ts].iloc[0],top_layers,snapshot)
    val,val_summary=validate_against_events(defense,merged,cfg.cascade_trigger)
    alerts=grouped_alerts(defense,layer_timeline,cfg.cascade_trigger)
except Exception as e:
    st.error(f"Compute failed: {e}"); st.exception(e); st.stop()

cols=st.columns(8)
cols[0].metric("Nodes", topology["node_id"].nunique())
cols[1].metric("Components", scaled["component_id"].nunique())
cols[2].metric("λᵣ", f"{solve_lambda_r(cfg.coherence_radius):.4f}")
cols[3].metric("Peak defense", f"{defense['final_defense_score'].max():.3f}")
cols[4].metric("Shape", shape["failure_shape"][:18])
cols[5].metric("Obs conf", obs_conf)
cols[6].metric("Graphs", len(graph_edges))
cols[7].metric("Alerts", len(alerts))

tabs=st.tabs(["Executive", "Layer Risks", "Cascade vs Fleet", "Failure Shape", "Recommendations", "Quality", "Validation", "Alerts", "Report", "Raw"])

with tabs[0]:
    st.subheader("Executive View")
    st.write("V4 separates ordinary GPU health issues from true multi-layer cascade risks.")
    st.json(shape)
    fig,ax=plt.subplots()
    ax.plot(defense["timestamp"],defense["final_defense_score"],label="final defense")
    ax.plot(defense["timestamp"],defense["cascade_score"],label="cascade score")
    ax.plot(defense["timestamp"],defense["fleet_stress_score"],label="fleet stress")
    ax.axhline(cfg.cascade_trigger,linestyle="--",label="trigger")
    ax.tick_params(axis="x",labelrotation=90); ax.legend(); st.pyplot(fig)

with tabs[1]:
    st.subheader("Per-Graph Layer Risks")
    st.dataframe(layer_timeline,use_container_width=True)
    fig,ax=plt.subplots()
    for g in layer_timeline["graph"].unique():
        sub=layer_timeline[layer_timeline["graph"]==g]
        ax.plot(sub["timestamp"],sub["cascade_risk"],label=g)
    ax.tick_params(axis="x",labelrotation=90); ax.legend(); st.pyplot(fig)

with tabs[2]:
    st.subheader("Cascade vs Fleet Stress")
    st.write("Fleet stress means many components stressed. Cascade means topology-supported stress with propagation path.")
    st.dataframe(defense,use_container_width=True)
    fig,ax=plt.subplots()
    ax.scatter(defense["fleet_stress_score"],defense["cascade_score"])
    ax.set_xlabel("Fleet stress"); ax.set_ylabel("Cascade risk"); st.pyplot(fig)

with tabs[3]:
    st.subheader("Failure Shape Classifier")
    st.json(shape)
    st.dataframe(top_layers,use_container_width=True)

with tabs[4]:
    st.subheader("Recommendations")
    st.dataframe(recs,use_container_width=True)

with tabs[5]:
    st.subheader("Observability")
    st.dataframe(obs,use_container_width=True)
    st.subheader("Calibration Confidence")
    st.dataframe(cal_conf,use_container_width=True)
    st.subheader("Topology Health")
    st.dataframe(topo_health,use_container_width=True)

with tabs[6]:
    st.subheader("Validation vs throttling events")
    st.dataframe(val_summary,use_container_width=True)
    fig,ax=plt.subplots()
    for c in ["fcct_alert","temp_70_alert","temp_82_alert","event"]:
        if c in val: ax.plot(val["timestamp"],val[c],label=c)
    ax.tick_params(axis="x",labelrotation=90); ax.legend(); st.pyplot(fig)

with tabs[7]:
    st.subheader("Grouped Alerts")
    st.dataframe(alerts,use_container_width=True)
    pay=payload(alerts); st.json(pay)
    if webhook and st.button("Send webhook"): st.json(post_webhook(webhook,pay))

with tabs[8]:
    st.subheader("Audit Report")
    # layer summary
    summary=layer_timeline.groupby("graph").agg(peak_cascade=("cascade_risk","max"),peak_fleet=("fleet_stress","max"),max_pairs=("coherent_pairs","max")).reset_index()
    html=generate_report(source,shape,summary,layer_timeline,defense,recs,obs,cal_conf,topo_health,val_summary)
    st.download_button("Download HTML audit report",html.encode(),"ai_infra_cascade_defense_v4_report.html","text/html")
    st.components.v1.html(html,height=700,scrolling=True)

with tabs[9]:
    st.subheader("Scaled telemetry")
    st.dataframe(scaled.head(3000),use_container_width=True)
    st.subheader("Graph edges")
    edge_rows=[]
    for g,edges in graph_edges.items():
        for e in edges[:500]:
            edge_rows.append(e.__dict__)
    st.dataframe(pd.DataFrame(edge_rows),use_container_width=True)
