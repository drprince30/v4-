
from __future__ import annotations
import pandas as pd
from html import escape


def table(df, n=25):
    if df is None or df.empty:
        return "<p>No data.</p>"
    return df.head(n).to_html(index=False, escape=True)


def generate_report(source, shape, summary, layer_timeline, defense, recs, obs, cal_conf, topo_health, validation_summary):
    peak = float(defense["final_defense_score"].max()) if not defense.empty else 0
    return f"""
<html><head><title>AI Infrastructure Cascade Defense V4 Report</title>
<style>body{{font-family:Arial;margin:32px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #ddd;padding:6px}}th{{background:#f2f2f2}}.box{{border:1px solid #ddd;border-radius:8px;padding:14px;margin:12px 0}}</style></head>
<body>
<h1>AI Infrastructure Cascade Defense V4 Report</h1>
<div class='box'><b>Source:</b> {escape(str(source))}<br><b>Peak defense score:</b> {peak:.3f}<br><b>Failure shape:</b> {escape(str(shape.get('failure_shape')))}<br><b>Reason:</b> {escape(str(shape.get('reason')))}</div>
<h2>Executive Recommendations</h2>{table(recs,40)}
<h2>Layer Summary</h2>{table(summary,20)}
<h2>Validation Summary</h2>{table(validation_summary,20)}
<h2>Observability Quality</h2>{table(obs,30)}
<h2>Calibration Confidence</h2>{table(cal_conf,30)}
<h2>Topology Health</h2>{table(topo_health,20)}
<h2>Top Defense Frames</h2>{table(defense.sort_values('final_defense_score', ascending=False),20)}
<h2>Top Layer Risks</h2>{table(layer_timeline.sort_values('cascade_risk', ascending=False),30)}
<h2>Limitations</h2>
<ul><li>V4 is shadow-mode and recommendation-mode only.</li><li>It separates cascade risk from ordinary isolated point failures.</li><li>Facility/PDU/CDU data may be unavailable and must be integrated during pilot.</li><li>Automatic action requires separate safety approval.</li></ul>
</body></html>
"""
