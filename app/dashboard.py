"""
Crop Residue Burning & Bioenergy Dashboard
Punjab & Haryana | 47 Districts | 2015–2023 VIIRS Fire Data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# ── Path resolution ────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed")
MAPS = os.path.join(ROOT, "outputs", "maps")

st.set_page_config(
    page_title="Crop Residue & Bioenergy Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1923 0%, #1a2a3a 100%);
}
section[data-testid="stSidebar"] * { color: #e8f0f7 !important; }

div[data-testid="metric-container"] {
    background: #1e2d3d;
    border: 1px solid #2a3f52;
    border-radius: 10px;
    padding: 14px 18px;
}
div[data-testid="metric-container"] label {
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7aa8c8 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    color: #f0f6ff !important;
}

.zone-banner {
    padding: 12px 20px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    margin: 0.5rem 0 1rem 0;
    color: white;
}

.section-hdr {
    font-size: 11px;
    font-weight: 700;
    color: #7aa8c8;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1.2rem 0 0.5rem;
    padding-bottom: 5px;
    border-bottom: 1px solid #2a3f52;
}

.insight-card {
    border-left: 3px solid #0ea5e9;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 13px;
    line-height: 1.6;
    color: #e8f0f7;
    background: #1a2e40;
}
.insight-card.warning { border-left-color: #f59e0b; background: #2a2010; color: #fde68a; }
.insight-card.success { border-left-color: #22c55e; background: #0f2a1a; color: #bbf7d0; }
.insight-card.danger  { border-left-color: #ef4444; background: #2a1010; color: #fecaca; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────
@st.cache_data
def load_all_data():
    brs    = pd.read_csv(os.path.join(DATA, "burning_risk_scores.csv"))
    bsi    = pd.read_csv(os.path.join(DATA, "bsi_scores.csv"))
    bio    = pd.read_csv(os.path.join(DATA, "bioenergy_scores.csv"))
    env    = pd.read_csv(os.path.join(DATA, "env_features.csv"))
    clust  = pd.read_csv(os.path.join(DATA, "district_clusters.csv"))
    fire   = pd.read_csv(os.path.join(DATA, "fire_stats.csv"))
    trends = pd.read_csv(os.path.join(DATA, "fire_trends.csv"))

    for df in [brs, bsi, bio, env, clust, fire, trends]:
        df["district"] = df["district"].str.strip().str.title()

    master = (
        bsi
        .merge(brs[["district", "BRS", "risk_class"]], on="district", how="left")
        .merge(bio[["district", "BPS", "avg_revenue_crore", "avg_NBP_GJ", "avg_CBG_tonnes"]], on="district", how="left")
        .merge(env[["district", "fire_count", "residue", "avg_temp", "rainfall"]], on="district", how="left")
        .merge(clust[["district", "zone"]], on="district", how="left")
    )

    bsi_thresh = master["BSI"].quantile(0.70)
    res_thresh = master["residue"].quantile(0.70)

    def assign_zone(r):
        if r["BSI"] >= bsi_thresh and r["residue"] >= res_thresh:
            return "Plant Zone"
        elif r["BSI"] >= bsi_thresh:
            return "Policy Zone"
        return "Low Priority"

    master["decision_zone"] = master.apply(assign_zone, axis=1)
    master["bsi_thresh"]    = bsi_thresh
    master["res_thresh"]    = res_thresh
    master["bsi_rank"]      = master["BSI"].rank(ascending=False, method="min", na_option="bottom").astype("Int64")

    return master, fire, trends


master, fire_stats, trends = load_all_data()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px'>
        <div style='font-size:2rem'>🌾</div>
        <div style='font-size:15px; font-weight:700; letter-spacing:0.04em'>Crop Residue</div>
        <div style='font-size:10px; color:#7aa8c8; text-transform:uppercase; letter-spacing:0.12em'>Bioenergy Dashboard</div>
    </div>
    <div style='height:1px; background:linear-gradient(90deg,#0ea5e9,#8b5cf6); margin:10px 0 18px'></div>
    """, unsafe_allow_html=True)

    district_list = sorted(master["district"].dropna().unique().tolist())
    default_idx   = district_list.index("Sangrur") if "Sangrur" in district_list else 0
    district = st.selectbox("📍 Select District", district_list, index=default_idx)

    st.markdown("<div style='height:1px; background:#2a3f52; margin:16px 0'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px; color:#7aa8c8; line-height:2'>
    🔴 <b>Plant Zone</b> — BSI ≥ p70 + Residue ≥ p70<br>
    🟠 <b>Policy Zone</b> — High BSI, low residue<br>
    ⚫ <b>Low Priority</b> — Below thresholds<br><br>
    📡 NASA FIRMS VIIRS 2015–2023<br>
    🗺️ Punjab & Haryana | 47 Districts
    </div>
    """, unsafe_allow_html=True)


# ── Selected row ───────────────────────────────────────────────────────────
row    = master[master["district"] == district].iloc[0]
t_row  = trends[trends["district"] == district]
d_fire = fire_stats[fire_stats["district"] == district].sort_values("year")

zone_cfg = {
    "Plant Zone":   ("#dc2626", "🔴 PLANT ZONE — Both BSI and Residue exceed 70th percentile. Priority CBG plant location."),
    "Policy Zone":  ("#ea580c", "🟠 POLICY ZONE — High burning severity, but residue below commercial threshold."),
    "Low Priority": ("#16a34a", "🟢 LOW PRIORITY — Below environmental thresholds. Annual monitoring recommended."),
}
z_color, z_label = zone_cfg.get(row["decision_zone"], ("#666", row["decision_zone"]))

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='font-size:2rem; font-weight:700; color:#f0f6ff; margin-bottom:2px'>{district}</div>
<div style='font-size:13px; color:#7aa8c8; margin-bottom:10px'>
    Punjab & Haryana Crop Residue & Bioenergy Analysis &nbsp;|&nbsp;
    BSI Rank <b style='color:#e8f0f7'>#{int(row['bsi_rank'])}</b> of {len(master)} districts
</div>
""", unsafe_allow_html=True)
st.markdown(f"<div class='zone-banner' style='background:{z_color}'>{z_label}</div>", unsafe_allow_html=True)

# ── KPIs ───────────────────────────────────────────────────────────────────
def fmt(val, s, fb="N/A"):
    try: return s.format(val) if pd.notna(val) else fb
    except: return fb

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🔥 BSI Score",       fmt(row['BSI'],               "{:.1f} / 100"))
c2.metric("⚠️ BRS Score",       fmt(row['BRS'],               "{:.1f} / 100"), delta=str(row.get("risk_class", "")))
c3.metric("🌿 Bioenergy Score", fmt(row['BPS'],               "{:.1f} / 100"))
c4.metric("💰 CBG Revenue",     fmt(row['avg_revenue_crore'], "₹{:.1f} Cr/yr"))
c5.metric("📡 Fire Count",      fmt(row['fire_count'],        "{:,.0f} /yr"))

st.markdown("<div style='height:1px; background:linear-gradient(90deg,#0ea5e9,#8b5cf6); margin:1.2rem 0'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 District Overview",
    "📈 Fire Trend",
    "🌿 Bioenergy",
    "🗺️ All Districts",
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — District Overview
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("<div class='section-hdr'>Score Gauges</div>", unsafe_allow_html=True)

        fig_gauge = make_subplots(
            rows=1, cols=3,
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        )
        for i, (val, title, color) in enumerate([
            (row['BSI'], "BSI", "#ef4444"),
            (row['BRS'], "BRS", "#f97316"),
            (row['BPS'], "BPS", "#22c55e"),
        ]):
            fig_gauge.add_trace(go.Indicator(
                mode="gauge+number",
                value=float(val) if pd.notna(val) else 0,
                title={"text": title, "font": {"size": 13}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#4a6580"},
                    "bar":  {"color": color, "thickness": 0.25},
                    "bgcolor": "#1a2a3a",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  33], "color": "#0f1923"},
                        {"range": [33, 66], "color": "#132030"},
                        {"range": [66,100], "color": "#182838"},
                    ],
                },
                number={"font": {"size": 22, "family": "JetBrains Mono", "color": "#f0f6ff"}, "suffix": "/100"},
            ), row=1, col=i+1)

        fig_gauge.update_layout(
            height=190,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Radar
        st.markdown("<div class='section-hdr'>Feature Profile vs Region Average</div>", unsafe_allow_html=True)
        feat_cols   = ["BSI", "BRS", "BPS", "fire_count", "residue"]
        feat_labels = ["BSI", "BRS", "BPS", "Fire Count", "Residue"]

        def safe_norm(col):
            mn, mx = master[col].min(), master[col].max()
            return (master[col] - mn) / (mx - mn) if mx != mn else master[col] * 0

        d_vals   = [float(safe_norm(c)[master["district"] == district].values[0]) for c in feat_cols]
        avg_vals = [float(safe_norm(c).mean()) for c in feat_cols]

        fig_radar = go.Figure()
        for name, vals, color, fill in [
            (district,     d_vals,   "#ef4444", "rgba(239,68,68,0.15)"),
            ("Region Avg", avg_vals, "#3b82f6", "rgba(59,130,246,0.08)"),
        ]:
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=feat_labels + [feat_labels[0]],
                fill="toself", name=name,
                line=dict(color=color, width=2.5), fillcolor=fill,
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#1a2a3a",
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=8, color="#7aa8c8"), gridcolor="#2a3f52"),
                angularaxis=dict(tickfont=dict(size=11, color="#e8f0f7"), gridcolor="#2a3f52"),
            ),
            showlegend=True,
            legend=dict(y=-0.15, orientation="h", font=dict(size=11, color="#e8f0f7")),
            height=320,
            margin=dict(l=40, r=40, t=10, b=50),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-hdr'>District Insights</div>", unsafe_allow_html=True)

        bsi_v  = float(row['BSI'])
        rev    = float(row['avg_revenue_crore']) if pd.notna(row['avg_revenue_crore']) else 0
        res_kt = float(row['residue']) / 1000 if pd.notna(row['residue']) else 0

        sev_cls = "danger" if bsi_v >= 70 else "warning" if bsi_v >= 40 else "success"
        sev_msg = "🔥 Extremely high burning severity." if bsi_v >= 70 else \
                  "⚠️ Moderate to high burning severity." if bsi_v >= 40 else \
                  "✅ Low burning severity relative to region."
        st.markdown(f"<div class='insight-card {sev_cls}'><b>Burning Severity:</b> {sev_msg} BSI = <b>{bsi_v:.1f}/100</b>, rank <b>#{int(row['bsi_rank'])}</b> of {len(master)}.</div>", unsafe_allow_html=True)

        rev_cls = "success" if rev > 50 else "warning" if rev > 20 else "danger"
        st.markdown(f"<div class='insight-card {rev_cls}'><b>Economic Potential:</b> Estimated CBG revenue <b>₹{rev:.1f} Cr/yr</b> from <b>{res_kt:.1f}K tonnes</b> recoverable residue at ₹46/kg (SATAT).</div>", unsafe_allow_html=True)

        if not t_row.empty:
            tr = t_row.iloc[0]
            slope, sig = float(tr["slope"]), bool(tr["significant"])
            tr_cls = "danger" if slope > 50 else "warning" if slope > 0 else "success"
            sig_txt = "✅ statistically significant" if sig else "not significant at p < 0.05"
            st.markdown(f"<div class='insight-card {tr_cls}'><b>Fire Trend 2015–2023:</b> Sen's slope = <b>{slope:+.1f} fires/yr</b> ({sig_txt}). Direction: <b>{str(tr['trend']).title()}</b>.</div>", unsafe_allow_html=True)

        action = {
            "Plant Zone":   ("danger",  "🏭 <b>Recommended Action:</b> Apply under SATAT scheme immediately. Engage FPOs for residue baling. Target commissioning within 2 years."),
            "Policy Zone":  ("warning", "📋 <b>Recommended Action:</b> Launch PUSA decomposer campaign. Increase MSP for residue. Build supply chain. Reassess in 2–3 years."),
            "Low Priority": ("success", "📡 <b>Recommended Action:</b> Maintain annual NASA FIRMS satellite monitoring. Reassess if fire trend increases."),
        }
        cls, msg = action.get(row["decision_zone"], ("", ""))
        st.markdown(f"<div class='insight-card {cls}'>{msg}</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-hdr'>Environmental Features vs Region</div>", unsafe_allow_html=True)
        feat_tbl = {
            "Fire Count (fires/yr)": "fire_count",
            "Residue (tonnes/yr)":   "residue",
            "Avg Temp (°C)":         "avg_temp",
            "Rainfall (mm)":         "rainfall",
        }
        rows_tbl = []
        for label, col in feat_tbl.items():
            dv  = float(master.loc[master["district"] == district, col].values[0])
            pct = float(master[col].rank(pct=True)[master["district"] == district].values[0]) * 100
            rows_tbl.append({
                "Feature":     label,
                district:      f"{dv:,.1f}",
                "Region Mean": f"{master[col].mean():,.1f}",
                "Percentile":  f"{pct:.0f}th",
            })
        st.dataframe(pd.DataFrame(rows_tbl).set_index("Feature"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Fire Trend
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    col_f1, col_f2 = st.columns([3, 1])

    with col_f1:
        st.subheader(f"Annual Fire Detections — {district}")

        if not d_fire.empty:
            slope   = float(t_row["slope"].values[0])    if not t_row.empty else 0
            sig     = bool(t_row["significant"].values[0]) if not t_row.empty else False
            trend_s = t_row["trend"].values[0]            if not t_row.empty else "—"
            p_val   = float(t_row["p_value"].values[0])  if not t_row.empty else 1.0

            fig_fire = go.Figure()
            fig_fire.add_trace(go.Bar(
                x=d_fire["year"], y=d_fire["fire_count"],
                name="Fire Count", marker_color="#ef4444", opacity=0.75,
            ))
            fig_fire.add_trace(go.Scatter(
                x=d_fire["year"], y=d_fire["fire_count"],
                mode="lines+markers",
                line=dict(color="#fca5a5", width=2),
                marker=dict(size=5), name="Trend line",
            ))
            fig_fire.update_layout(
                title=dict(
                    text=f"Mann-Kendall: <b>{trend_s}</b> | Sen's slope = {slope:+.1f} fires/yr | p = {p_val:.4f} {'✅ significant' if sig else ''}",
                    font=dict(size=12, color="#e8f0f7"),
                ),
                height=360,
                plot_bgcolor="#0f1923", paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Year", tickmode="linear", dtick=1, color="#7aa8c8", gridcolor="#1e2d3d"),
                yaxis=dict(title="VIIRS Fire Detections", color="#7aa8c8", gridcolor="#1e2d3d"),
                legend=dict(orientation="h", y=-0.2, font=dict(color="#e8f0f7")),
                margin=dict(l=50, r=20, t=50, b=60),
            )
            st.plotly_chart(fig_fire, use_container_width=True)

            st.markdown("<div class='section-hdr'>Year-by-Year Statistics</div>", unsafe_allow_html=True)
            tbl_cols = [c for c in ["year", "fire_count", "mean_frp", "fire_days", "onset_doy", "peak_week"] if c in d_fire.columns]
            fire_tbl = d_fire[tbl_cols].copy()
            fire_tbl.columns = [c.replace("_", " ").title() for c in tbl_cols]
            if "Mean Frp" in fire_tbl.columns:
                fire_tbl.rename(columns={"Mean Frp": "Mean FRP (MW)"}, inplace=True)
            st.dataframe(fire_tbl.set_index("Year"), use_container_width=True)
        else:
            st.info(f"No fire data found for {district}.")

    with col_f2:
        st.markdown("<div class='section-hdr'>Mann-Kendall Summary</div>", unsafe_allow_html=True)
        if not t_row.empty:
            r = t_row.iloc[0]
            st.metric("Trend",       str(r["trend"]).title())
            st.metric("Sen's Slope", f"{r['slope']:+.1f} fires/yr")
            st.metric("Kendall's τ", f"{r['tau']:.3f}")
            st.metric("p-value",     f"{r['p_value']:.4f}")
            st.metric("Significant", "Yes ✅" if r["significant"] else "No")
        else:
            st.info("No trend data.")

        st.markdown("<div class='section-hdr'>Significant Trends</div>", unsafe_allow_html=True)
        sig_d = trends[trends["significant"] == True]
        st.metric("Districts with significant trend", len(sig_d))
        for _, sr in sig_d.iterrows():
            arrow = "📈" if sr["slope"] > 0 else "📉"
            st.markdown(f"<span style='color:#e8f0f7; font-size:13px'>{arrow} <b>{sr['district']}</b> {sr['slope']:+.1f}/yr</span>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Bioenergy
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    bio_row = master[master["district"] == district].iloc[0]
    bc1, bc2, bc3 = st.columns(3)
    bc1.metric("⚡ Net Bioenergy",   fmt(bio_row['avg_NBP_GJ'] / 1e6 if pd.notna(bio_row.get('avg_NBP_GJ')) else None, "{:.2f} PJ/yr"))
    bc2.metric("🛢️ CBG Production", fmt(bio_row.get('avg_CBG_tonnes'), "{:,.0f} t/yr"))
    bc3.metric("💰 Annual Revenue", fmt(bio_row.get('avg_revenue_crore'), "₹{:.2f} Cr/yr"))

    st.markdown("<div style='height:1px; background:#2a3f52; margin:1rem 0'></div>", unsafe_allow_html=True)

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.markdown("<div class='section-hdr'>CBG Revenue — Top 20 Districts</div>", unsafe_allow_html=True)
        top20 = (master.dropna(subset=["avg_revenue_crore"])
                 .nlargest(20, "avg_revenue_crore")
                 .sort_values("avg_revenue_crore", ascending=True))
        colors = ["#ef4444" if d == district else "#3b82f6" for d in top20["district"]]
        fig_cbg = go.Figure(go.Bar(
            x=top20["avg_revenue_crore"], y=top20["district"],
            orientation="h", marker_color=colors,
            text=["₹" + str(round(v, 1)) + " Cr" for v in top20["avg_revenue_crore"]],
            textposition="outside", textfont=dict(color="#e8f0f7", size=10),
            hovertemplate="<b>%{y}</b><br>₹%{x:.1f} Cr/yr<extra></extra>",
        ))
        fig_cbg.update_layout(
            height=500,
            plot_bgcolor="#0f1923", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="₹ Crore/yr", color="#7aa8c8", gridcolor="#1e2d3d"),
            yaxis=dict(color="#e8f0f7"),
            margin=dict(l=10, r=80, t=10, b=40),
        )
        st.plotly_chart(fig_cbg, use_container_width=True)

    with col_b2:
        st.markdown("<div class='section-hdr'>Decision Space: BSI × Residue</div>", unsafe_allow_html=True)
        st.caption("Bubble size = CBG Revenue. Selected district marked ★.")

        plot_data = master.dropna(subset=["BSI", "residue"]).copy()
        plot_data["residue_kt"]  = plot_data["residue"] / 1000
        plot_data["bubble_size"] = plot_data["avg_revenue_crore"].fillna(1).clip(lower=1)

        zone_colors = {"Plant Zone": "#ef4444", "Policy Zone": "#f97316", "Low Priority": "#475569"}
        fig_sc = px.scatter(
            plot_data, x="BSI", y="residue_kt",
            color="decision_zone", size="bubble_size", size_max=25,
            color_discrete_map=zone_colors,
            labels={"BSI": "BSI Score", "residue_kt": "Residue (000 t/yr)", "decision_zone": "Zone"},
            hover_data={"district": True, "avg_revenue_crore": ":.1f", "bubble_size": False},
        )
        fig_sc.update_traces(marker=dict(opacity=0.8))

        bsi_t = float(row["bsi_thresh"])
        res_t = float(row["res_thresh"]) / 1000
        fig_sc.add_vline(x=bsi_t, line_dash="dash", line_color="#94a3b8",
                         annotation_text=f"p70 BSI={bsi_t:.1f}", annotation_font_size=10, annotation_font_color="#94a3b8")
        fig_sc.add_hline(y=res_t, line_dash="dash", line_color="#94a3b8",
                         annotation_text=f"p70 Residue={res_t:.1f}K t", annotation_font_size=10, annotation_font_color="#94a3b8")

        sel = master[master["district"] == district]
        if not sel.empty:
            fig_sc.add_trace(go.Scatter(
                x=sel["BSI"], y=sel["residue"] / 1000,
                mode="markers+text", text=[f"★ {district}"],
                textposition="top center",
                textfont=dict(color="#f0f6ff", size=11),
                marker=dict(size=18, color="#fbbf24", symbol="star"),
                name=f"★ {district}",
            ))

        fig_sc.update_layout(
            height=500,
            plot_bgcolor="#0f1923", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#7aa8c8", gridcolor="#1e2d3d"),
            yaxis=dict(color="#7aa8c8", gridcolor="#1e2d3d"),
            legend=dict(font=dict(color="#e8f0f7")),
            margin=dict(l=40, r=40, t=20, b=50),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("<div class='section-hdr'>Policy Recommendation</div>", unsafe_allow_html=True)
    recs = {
        "Plant Zone":   ("danger",  f"🏭 <b>{district} — Priority Plant Investment.</b> BSI = {row['BSI']:.1f}/100, Residue = {row['residue']/1000 if pd.notna(row['residue']) else 0:.1f}K t/yr. Revenue estimate: ₹{row['avg_revenue_crore']:.1f} Cr/yr. Apply under SATAT immediately. Engage FPOs for baling and logistics."),
        "Policy Zone":  ("warning", f"📋 <b>{district} — Policy Intervention First.</b> High BSI ({row['BSI']:.1f}/100) but residue below viability threshold. Launch PUSA decomposer campaign, increase MSP for residue. Reassess in 2–3 years."),
        "Low Priority": ("success", f"📡 <b>{district} — Monitor & Wait.</b> BSI = {row['BSI']:.1f}/100, below thresholds. Maintain annual VIIRS monitoring. Reassess if fire trend increases."),
    }
    cls, msg = recs.get(row["decision_zone"], ("", ""))
    st.markdown(f"<div class='insight-card {cls}' style='font-size:14px; padding:14px 18px'>{msg}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — All Districts
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    zc = master["decision_zone"].value_counts()
    col_z1, col_z2, col_z3 = st.columns(3)
    col_z1.metric("🔴 Plant Zones",  zc.get("Plant Zone", 0))
    col_z2.metric("🟠 Policy Zones", zc.get("Policy Zone", 0))
    col_z3.metric("⚫ Low Priority", zc.get("Low Priority", 0))

    st.markdown("<div style='height:1px; background:#2a3f52; margin:1rem 0'></div>", unsafe_allow_html=True)

    search = st.text_input("🔍 Search district", "")
    all_cols = ["district", "BSI", "BRS", "BPS", "avg_revenue_crore", "fire_count", "residue", "decision_zone"]
    disp = master[all_cols].dropna(subset=["BSI"]).sort_values("BSI", ascending=False).reset_index(drop=True)
    disp.index += 1
    if search:
        disp = disp[disp["district"].str.contains(search, case=False)]
    disp.columns = ["District", "BSI", "BRS", "BPS", "Revenue (₹ Cr)", "Fire Count", "Residue (t)", "Zone"]

    st.dataframe(
        disp.style
            .format({"BSI": "{:.1f}", "BRS": "{:.1f}", "BPS": "{:.1f}",
                     "Revenue (₹ Cr)": "₹{:.1f}", "Fire Count": "{:,.0f}", "Residue (t)": "{:,.0f}"})
            .background_gradient(subset=["BSI"], cmap="YlOrRd")
            .background_gradient(subset=["Revenue (₹ Cr)"], cmap="Greens"),
        use_container_width=True, height=480,
    )

    st.markdown("<div style='height:1px; background:#2a3f52; margin:1rem 0'></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-hdr'>District Maps</div>", unsafe_allow_html=True)
    col_m1, col_m2 = st.columns(2)
    for col, (fname, title, caption) in zip([col_m1, col_m2], [
        ("16_final_decision_map.png", "Final Decision Map",         "🔴 Plant Zone (9) | 🟠 Policy Zone (4) | ⚫ Low Priority (30)"),
        ("15_bsi_map.png",            "Burning Severity Index Map", "PCA-weighted BSI 0–100 across all 47 districts."),
    ]):
        with col:
            st.markdown(f"**{title}**")
            path = os.path.join(MAPS, fname)
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.info(f"Map not found: `{fname}`")
            st.caption(caption)


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#4a6580; font-size:11px; padding:16px 0; font-family:JetBrains Mono'>
    🌾 Crop Residue & Bioenergy Dashboard &nbsp;|&nbsp; NASA FIRMS VIIRS 2015–2023 &nbsp;|&nbsp; Punjab & Haryana
</div>
""", unsafe_allow_html=True)
