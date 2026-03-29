"""
Crop Residue Burning & Bioenergy Dashboard — Enhanced Edition
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
    page_title="Crop Residue & Bioenergy | Punjab & Haryana",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1923 0%, #1a2a3a 100%);
}
section[data-testid="stSidebar"] * {
    color: #e8f0f7 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: #94b8d4 !important;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.2s;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.1);
}
div[data-testid="metric-container"] label {
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
}

/* Zone banner */
.zone-banner {
    padding: 14px 22px;
    border-radius: 10px;
    font-size: 15px;
    font-weight: 600;
    margin: 0.5rem 0 1.2rem 0;
    color: white;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* Section headers */
.section-hdr {
    font-size: 11px;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1.4rem 0 0.5rem;
    padding-bottom: 6px;
    border-bottom: 1px solid #e2e8f0;
}

/* Hero title */
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #0f1923;
    line-height: 1.1;
    margin-bottom: 4px;
}
.hero-sub {
    font-size: 14px;
    color: #64748b;
    margin-bottom: 12px;
}

/* Insight card */
.insight-card {
    background: #f0f9ff;
    border-left: 4px solid #0ea5e9;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13.5px;
    line-height: 1.6;
}
.insight-card.warning {
    background: #fff7ed;
    border-left-color: #f97316;
}
.insight-card.success {
    background: #f0fdf4;
    border-left-color: #22c55e;
}
.insight-card.danger {
    background: #fef2f2;
    border-left-color: #ef4444;
}

/* Score pill */
.score-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
}

/* Tab styling */
button[data-baseweb="tab"] {
    font-family: 'Sora', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

/* Divider */
.fancy-divider {
    height: 2px;
    background: linear-gradient(90deg, #0ea5e9, #8b5cf6, #ec4899);
    border-radius: 2px;
    margin: 1.5rem 0;
}
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
    master["revenue_rank"]  = master["avg_revenue_crore"].rank(ascending=False, method="min", na_option="bottom").astype("Int64")

    # Percentile columns for gauge charts
    for col in ["BSI", "BRS", "BPS", "avg_revenue_crore", "fire_count", "residue"]:
        master[f"{col}_pct"] = master[col].rank(pct=True, na_option="bottom") * 100

    return master, fire, trends


master, fire_stats, trends = load_all_data()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 4px;'>
        <div style='font-size:2.2rem'>🌾</div>
        <div style='font-size:15px; font-weight:700; color:#e8f0f7; letter-spacing:0.04em'>Crop Residue</div>
        <div style='font-size:11px; color:#7aa8c8; text-transform:uppercase; letter-spacing:0.1em'>Bioenergy Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:1px; background:linear-gradient(90deg,#0ea5e9,#8b5cf6); margin:12px 0 20px'></div>", unsafe_allow_html=True)

    district_list = sorted(master["district"].dropna().unique().tolist())
    default_idx   = district_list.index("Sangrur") if "Sangrur" in district_list else 0
    district = st.selectbox("📍 Select District", district_list, index=default_idx)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # Quick compare toggle
    compare_mode = st.toggle("🔁 Compare Two Districts", value=False)
    if compare_mode:
        district2_list = [d for d in district_list if d != district]
        district2 = st.selectbox("📍 Compare With", district2_list)
    else:
        district2 = None

    st.markdown("<div style='height:1px; background:#2a3f52; margin:16px 0'></div>", unsafe_allow_html=True)

    # Filter panel for All Districts tab
    st.markdown("<div style='font-size:11px; color:#7aa8c8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px'>🎛 Global Filters</div>", unsafe_allow_html=True)

    zone_filter = st.multiselect(
        "Decision Zone",
        ["Plant Zone", "Policy Zone", "Low Priority"],
        default=["Plant Zone", "Policy Zone", "Low Priority"],
    )

    bsi_range = st.slider("BSI Range", 0, 100, (0, 100))
    rev_range = st.slider(
        "Revenue Range (₹ Cr/yr)",
        0,
        int(master["avg_revenue_crore"].max()) + 1,
        (0, int(master["avg_revenue_crore"].max()) + 1),
    )

    st.markdown("<div style='height:1px; background:#2a3f52; margin:16px 0'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px; color:#7aa8c8; line-height:1.7'>
    🔴 <b>Plant Zone</b> — BSI ≥ p70 + Residue ≥ p70<br>
    🟠 <b>Policy Zone</b> — High BSI only<br>
    ⚫ <b>Low Priority</b> — Below threshold<br><br>
    📡 <b>Data:</b> NASA FIRMS VIIRS 2015–2023<br>
    🗺️ Punjab & Haryana | 47 Districts
    </div>
    """, unsafe_allow_html=True)


# ── Selected district data ─────────────────────────────────────────────────
row  = master[master["district"] == district].iloc[0]
row2 = master[master["district"] == district2].iloc[0] if district2 else None

# ── Filtered master for charts ─────────────────────────────────────────────
filt = master[
    master["decision_zone"].isin(zone_filter) &
    master["BSI"].between(*bsi_range) &
    master["avg_revenue_crore"].between(*rev_range)
].copy()


# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════
zone_cfg = {
    "Plant Zone":   ("#dc2626", "linear-gradient(135deg,#dc2626,#b91c1c)", "🔴 PLANT ZONE — Build CBG plant now. Both BSI and Residue exceed 70th percentile thresholds."),
    "Policy Zone":  ("#ea580c", "linear-gradient(135deg,#ea580c,#c2410c)", "🟠 POLICY ZONE — High burning severity. Residue supply below commercial threshold. Policy first."),
    "Low Priority": ("#16a34a", "linear-gradient(135deg,#16a34a,#15803d)", "🟢 LOW PRIORITY — Below environmental thresholds. Annual satellite monitoring recommended."),
}
z_color, z_grad, z_label = zone_cfg.get(row["decision_zone"], ("#666", "#666", row["decision_zone"]))

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    state_tag = "Punjab" if row.get("rainfall", 500) < 700 else "Haryana"  # rough heuristic
    bsi_rank_str = f"#{int(row['bsi_rank'])}" if pd.notna(row['bsi_rank']) else "N/A"
    rev_rank_str = f"#{int(row['revenue_rank'])}" if pd.notna(row['revenue_rank']) else "N/A"
    st.markdown(f"""
    <div class='hero-title'>{district}</div>
    <div class='hero-sub'>Punjab & Haryana Crop Residue & Bioenergy Analysis &nbsp;|&nbsp; BSI Rank <b>{bsi_rank_str}</b> of {len(master)} districts &nbsp;|&nbsp; Revenue Rank <b>{rev_rank_str}</b></div>
    """, unsafe_allow_html=True)
    st.markdown(f"<div class='zone-banner' style='background:{z_grad}'>{z_label}</div>", unsafe_allow_html=True)
with col_h2:
    # Mini sparkline of BSI percentile
    pct = float(row["BSI_pct"])
    color = "#dc2626" if pct > 70 else "#ea580c" if pct > 40 else "#16a34a"
    st.markdown(f"""
    <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:18px; text-align:center; margin-top:8px'>
        <div style='font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px'>BSI Percentile</div>
        <div style='font-size:2.8rem; font-weight:700; color:{color}; font-family: JetBrains Mono, monospace'>{pct:.0f}<span style='font-size:1.2rem'>th</span></div>
        <div style='font-size:12px; color:#64748b'>Burning severity vs all districts</div>
    </div>
    """, unsafe_allow_html=True)


# ── KPI Strip ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-hdr'>Key Performance Indicators</div>", unsafe_allow_html=True)

def fmt(val, fmt_str, fallback="N/A"):
    try:
        return fmt_str.format(val) if pd.notna(val) else fallback
    except:
        return fallback

c1, c2, c3, c4, c5, c6 = st.columns(6)

delta2 = None
c1.metric("🔥 BSI Score",      fmt(row['BSI'], "{:.1f}/100"),
          delta=f"#{int(row['bsi_rank'])} ranked" if pd.notna(row.get('bsi_rank')) else None,
          help="PCA-derived Burning Severity Index (NB14). Weights: rainfall 28.3%, temp 27.0%, fire count 22.9%, residue 21.8%.")
c2.metric("⚠️ BRS Score",      fmt(row['BRS'], "{:.1f}/100"),
          delta=str(row.get("risk_class", "")),
          help="Burning Risk Score (NB06): fire frequency 40% + FRP 30% + Mann-Kendall slope 30%.")
c3.metric("🌿 Bioenergy Score", fmt(row['BPS'], "{:.1f}/100"),
          help="Bioenergy Potential Score (NB10): normalised Net Bioenergy Potential in GJ.")
c4.metric("💰 CBG Revenue",    fmt(row['avg_revenue_crore'], "₹{:.1f} Cr/yr"),
          help="Annual Compressed Biogas revenue at ₹46/kg (SATAT scheme).")
c5.metric("📡 Fire Count",     fmt(row['fire_count'], "{:,.0f} fires/yr"),
          help="Mean annual VIIRS fire detections 2015–2023.")
c6.metric("🌾 Residue",        fmt(row['residue']/1000 if pd.notna(row['residue']) else None, "{:.1f}K t/yr"),
          help="Mean annual recoverable crop residue in tonnes.")

# Compare strip
if district2 and row2 is not None:
    st.markdown(f"<div class='section-hdr'>Comparing with {district2}</div>", unsafe_allow_html=True)
    cc1, cc2, cc3, cc4, cc5, cc6 = st.columns(6)

    def delta_str(v1, v2):
        if pd.notna(v1) and pd.notna(v2):
            d = v1 - v2
            return f"{d:+.1f} vs {district2}"
        return None

    cc1.metric("🔥 BSI Score",      fmt(row2['BSI'], "{:.1f}/100"),    delta=delta_str(row['BSI'], row2['BSI']))
    cc2.metric("⚠️ BRS Score",      fmt(row2['BRS'], "{:.1f}/100"),    delta=delta_str(row['BRS'], row2['BRS']))
    cc3.metric("🌿 Bioenergy Score", fmt(row2['BPS'], "{:.1f}/100"),    delta=delta_str(row['BPS'], row2['BPS']))
    cc4.metric("💰 CBG Revenue",    fmt(row2['avg_revenue_crore'], "₹{:.1f} Cr/yr"), delta=delta_str(row['avg_revenue_crore'], row2['avg_revenue_crore']))
    cc5.metric("📡 Fire Count",     fmt(row2['fire_count'], "{:,.0f}"), delta=delta_str(row['fire_count'], row2['fire_count']))
    cc6.metric("🌾 Residue",        fmt(row2['residue']/1000 if pd.notna(row2['residue']) else None, "{:.1f}K t/yr"))

st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Overview",
    "🗺️ Maps & Zones",
    "📈 Fire Trend",
    "🌿 Bioenergy",
    "🔬 Environmental",
    "📊 All Districts",
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW (new!)
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-hdr'>Score Gauges</div>", unsafe_allow_html=True)

    # Gauge row
    g1, g2, g3, g4 = st.columns(4)

    def make_gauge(value, title, color, max_val=100):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value if pd.notna(value) else 0,
            title={"text": title, "font": {"size": 13}},
            gauge={
                "axis": {"range": [0, max_val], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "#f8fafc",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, max_val * 0.33], "color": "#f1f5f9"},
                    {"range": [max_val * 0.33, max_val * 0.66], "color": "#e2e8f0"},
                    {"range": [max_val * 0.66, max_val], "color": "#cbd5e1"},
                ],
                "threshold": {
                    "line": {"color": "#0f172a", "width": 3},
                    "thickness": 0.8,
                    "value": value if pd.notna(value) else 0,
                },
            },
            number={"font": {"size": 26, "family": "JetBrains Mono"}, "suffix": "/100"},
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
        return fig

    with g1:
        st.plotly_chart(make_gauge(row['BSI'], "Burning Severity (BSI)", "#dc2626"), use_container_width=True)
    with g2:
        st.plotly_chart(make_gauge(row['BRS'], "Burning Risk (BRS)", "#ea580c"), use_container_width=True)
    with g3:
        st.plotly_chart(make_gauge(row['BPS'], "Bioenergy Potential (BPS)", "#16a34a"), use_container_width=True)
    with g4:
        max_rev = float(master["avg_revenue_crore"].max())
        fig_rev = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(row['avg_revenue_crore']) if pd.notna(row['avg_revenue_crore']) else 0,
            title={"text": "CBG Revenue (₹ Cr)", "font": {"size": 13}},
            gauge={
                "axis": {"range": [0, max_rev], "tickwidth": 1},
                "bar": {"color": "#7c3aed", "thickness": 0.28},
                "bgcolor": "#f8fafc",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, max_rev * 0.33], "color": "#f1f5f9"},
                    {"range": [max_rev * 0.33, max_rev * 0.66], "color": "#e2e8f0"},
                    {"range": [max_rev * 0.66, max_rev], "color": "#cbd5e1"},
                ],
            },
            number={"font": {"size": 26, "family": "JetBrains Mono"}, "prefix": "₹", "suffix": " Cr"},
        ))
        fig_rev.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_rev, use_container_width=True)

    # Insight panel + radar side by side
    col_ins, col_radar = st.columns([1, 1])

    with col_ins:
        st.markdown("<div class='section-hdr'>AI-Style District Insights</div>", unsafe_allow_html=True)

        zone = row["decision_zone"]
        bsi_v, brs_v = float(row['BSI']), float(row['BRS']) if pd.notna(row['BRS']) else 0
        res_kt = float(row['residue'])/1000 if pd.notna(row['residue']) else 0
        rev = float(row['avg_revenue_crore']) if pd.notna(row['avg_revenue_crore']) else 0
        fc = float(row['fire_count']) if pd.notna(row['fire_count']) else 0

        # Severity insight
        sev_class = "danger" if bsi_v >= 70 else "warning" if bsi_v >= 40 else "success"
        sev_msg = ("🔥 Extremely high burning severity.") if bsi_v >= 70 else \
                  ("⚠️ Moderate to high burning severity.") if bsi_v >= 40 else \
                  ("✅ Low burning severity relative to region.")
        st.markdown(f"<div class='insight-card {sev_class}'><b>Burning Severity:</b> {sev_msg} BSI = <b>{bsi_v:.1f}/100</b>, ranking <b>#{int(row['bsi_rank'])}</b> of {len(master)} districts.</div>", unsafe_allow_html=True)

        # Revenue insight
        rev_class = "success" if rev > 50 else "warning" if rev > 20 else "danger"
        st.markdown(f"<div class='insight-card {rev_class}'><b>Economic Potential:</b> Estimated CBG revenue of <b>₹{rev:.1f} Cr/year</b>. At ₹46/kg SATAT price on <b>{res_kt:.1f}K tonnes</b> recoverable residue annually.</div>", unsafe_allow_html=True)

        # Trend insight
        t_row = trends[trends["district"] == district]
        if not t_row.empty:
            tr = t_row.iloc[0]
            slope = float(tr["slope"])
            sig = bool(tr["significant"])
            trend_cls = "danger" if slope > 50 else "warning" if slope > 0 else "success"
            sig_txt = "✅ statistically significant (p < 0.05)" if sig else "not statistically significant"
            st.markdown(f"<div class='insight-card {trend_cls}'><b>Fire Trend (2015–2023):</b> Sen's slope = <b>{slope:+.1f} fires/yr</b> ({sig_txt}). Trend direction: <b>{str(tr['trend']).title()}</b>.</div>", unsafe_allow_html=True)

        # Zone recommendation
        recs = {
            "Plant Zone":   ("danger",  f"🏭 <b>Action Required:</b> {district} qualifies for immediate SATAT CBG plant application. Both BSI ≥ 70th percentile and residue ≥ 70th percentile. Engage FPOs for baling logistics. Target commissioning within 2 years."),
            "Policy Zone":  ("warning", f"📋 <b>Policy First:</b> High BSI ({bsi_v:.1f}) but residue supply ({res_kt:.1f}K t) below commercial threshold. Launch PUSA decomposer campaign and MSP support. Reassess in 2–3 years."),
            "Low Priority": ("success", f"📡 <b>Monitor & Wait:</b> BSI and residue below thresholds. Maintain annual VIIRS satellite monitoring. Reassess if fire trend increases."),
        }
        cls, msg = recs.get(zone, ("", ""))
        st.markdown(f"<div class='insight-card {cls}'>{msg}</div>", unsafe_allow_html=True)

    with col_radar:
        st.markdown("<div class='section-hdr'>Multi-Dimensional Profile (Normalised)</div>", unsafe_allow_html=True)

        features_radar = ["BSI", "BRS", "BPS", "fire_count", "residue", "avg_revenue_crore"]
        labels_radar   = ["BSI", "BRS", "BPS", "Fire Count", "Residue", "Revenue"]

        def safe_norm(col):
            mn, mx = master[col].min(), master[col].max()
            return (master[col] - mn) / (mx - mn) if mx != mn else master[col] * 0

        dist_vals = [float(safe_norm(c)[master["district"] == district].values[0]) for c in features_radar]
        avg_vals  = [float(safe_norm(c).mean()) for c in features_radar]

        fig_radar = go.Figure()
        traces = [(district, dist_vals, "#dc2626", "rgba(220,38,38,0.15)")]
        if district2 and row2 is not None:
            d2_vals = [float(safe_norm(c)[master["district"] == district2].values[0]) for c in features_radar]
            traces.append((district2, d2_vals, "#7c3aed", "rgba(124,58,237,0.15)"))
        traces.append(("Region Avg", avg_vals, "#3b82f6", "rgba(59,130,246,0.08)"))

        for name, vals, lc, fc in traces:
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=labels_radar + [labels_radar[0]],
                fill="toself",
                name=name,
                line=dict(color=lc, width=2.5),
                fillcolor=fc,
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9)),
                angularaxis=dict(tickfont=dict(size=11)),
            ),
            showlegend=True,
            legend=dict(y=-0.18, orientation="h", font=dict(size=11)),
            height=380,
            margin=dict(l=40, r=40, t=20, b=60),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Rank bar vis
    st.markdown("<div class='section-hdr'>Where Does {d} Stand? — All Score Rankings</div>".replace("{d}", district), unsafe_allow_html=True)

    rank_cols = ["BSI", "BRS", "BPS", "avg_revenue_crore", "fire_count", "residue"]
    rank_labels = ["BSI", "BRS", "BPS", "Revenue", "Fire Count", "Residue"]
    n = len(master)

    pct_vals = [float(row[f"{c}_pct"]) for c in rank_cols]
    bar_colors = ["#dc2626" if p > 70 else "#ea580c" if p > 40 else "#16a34a" for p in pct_vals]

    fig_rank = go.Figure(go.Bar(
        x=rank_labels,
        y=pct_vals,
        marker_color=bar_colors,
        text=[f"{p:.0f}th pct" for p in pct_vals],
        textposition="outside",
        hovertemplate="%{x}: %{y:.1f}th percentile<extra></extra>",
    ))
    fig_rank.add_hline(y=70, line_dash="dash", line_color="#94a3b8", annotation_text="70th pct threshold", annotation_font_size=11)
    fig_rank.update_layout(
        height=280,
        yaxis=dict(range=[0, 115], title="Percentile Rank", ticksuffix="th"),
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_rank, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Maps & Zones
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    col_m1, col_m2 = st.columns(2)
    map_files = {
        "15_bsi_map.png":            ("Burning Severity Index (BSI) Map", "PCA-weighted BSI 0–100. Weights: rainfall 28.3%, avg_temp 27.0%, fire_count 22.9%, residue 21.8%."),
        "15_cluster_map.png":        ("Environmental Zones (K-Means k=3)", "🔴 High Stress/High Opportunity (n=10) — high fire + high residue, dry. 🟡 Moderate (n=17). 🟢 Low Priority (n=20)."),
        "16_final_decision_map.png": ("Final Decision Map", "🔴 Plant Zone (9 districts): BSI ≥ p70 AND Residue ≥ p70. 🟠 Policy Zone (4). ⚫ Low Priority (30)."),
        "15_top5_burning.png":       ("Top-5 Highest BSI Districts", "Sangrur (100), Bathinda (95.1), Muktsar (90), Fazilka (88.5), Sirsa (84.4)."),
    }
    for i, (fname, (title, caption)) in enumerate(map_files.items()):
        col = col_m1 if i % 2 == 0 else col_m2
        with col:
            st.subheader(title)
            path = os.path.join(MAPS, fname)
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.info(f"Map not found: `{fname}` — generate via notebooks.")
            st.caption(caption)
            if i == 1:
                st.markdown("---")

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # Interactive scatter
    st.markdown("<div class='section-hdr'>Interactive Decision Space: BSI × Recoverable Residue</div>", unsafe_allow_html=True)
    st.caption("Each bubble = a district. Size = CBG Revenue. Hover for details. Selected district marked ★.")

    plot_data = filt.dropna(subset=["BSI", "residue"]).copy()
    plot_data["residue_kt"]  = plot_data["residue"] / 1000
    plot_data["bubble_size"] = plot_data["avg_revenue_crore"].fillna(1).clip(lower=1)

    zone_color_map = {"Plant Zone": "#dc2626", "Policy Zone": "#ea580c", "Low Priority": "#94a3b8"}

    fig_scatter = px.scatter(
        plot_data,
        x="BSI", y="residue_kt",
        color="decision_zone",
        size="bubble_size",
        size_max=30,
        text="district",
        color_discrete_map=zone_color_map,
        hover_data={"BRS": ":.1f", "BPS": ":.1f", "avg_revenue_crore": ":.1f", "bubble_size": False, "residue_kt": ":.1f"},
        labels={"BSI": "Burning Severity Index (BSI)", "residue_kt": "Recoverable Residue (000 tonnes/yr)", "decision_zone": "Zone"},
    )
    fig_scatter.update_traces(textposition="top center", marker=dict(opacity=0.8), textfont=dict(size=9))

    bsi_t = float(row["bsi_thresh"])
    res_t = float(row["res_thresh"]) / 1000
    fig_scatter.add_vline(x=bsi_t, line_dash="dash", line_color="#64748b", annotation_text=f"BSI p70={bsi_t:.1f}", annotation_font_size=11)
    fig_scatter.add_hline(y=res_t, line_dash="dash", line_color="#64748b", annotation_text=f"Residue p70={res_t:.1f}K t", annotation_font_size=11)

    sel = master[master["district"] == district]
    if not sel.empty:
        fig_scatter.add_trace(go.Scatter(
            x=sel["BSI"], y=sel["residue"] / 1000,
            mode="markers+text",
            text=[f"★ {district}"],
            textposition="top center",
            marker=dict(size=20, color="#0f172a", symbol="star", opacity=1),
            name=f"★ {district}",
        ))
    if district2 and row2 is not None:
        sel2 = master[master["district"] == district2]
        fig_scatter.add_trace(go.Scatter(
            x=sel2["BSI"], y=sel2["residue"] / 1000,
            mode="markers+text",
            text=[f"◆ {district2}"],
            textposition="top center",
            marker=dict(size=16, color="#7c3aed", symbol="diamond", opacity=1),
            name=f"◆ {district2}",
        ))

    fig_scatter.update_layout(
        height=520, legend_title="Decision Zone",
        margin=dict(l=50, r=50, t=30, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Fire Trend
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    col_f1, col_f2 = st.columns([2, 1])

    with col_f1:
        st.subheader(f"Annual Fire Detections: {district}" + (f" vs {district2}" if district2 else ""))

        def get_fire_data(d):
            return fire_stats[fire_stats["district"] == d].sort_values("year")

        d_fire = get_fire_data(district)
        t_row  = trends[trends["district"] == district]

        if not d_fire.empty:
            slope    = float(t_row["slope"].values[0]) if not t_row.empty else 0
            sig      = bool(t_row["significant"].values[0]) if not t_row.empty else False
            trend_s  = t_row["trend"].values[0] if not t_row.empty else "no trend"
            p_val    = float(t_row["p_value"].values[0]) if not t_row.empty else 1.0

            fig_fire = make_subplots(specs=[[{"secondary_y": True}]])
            fig_fire.add_trace(go.Bar(
                x=d_fire["year"], y=d_fire["fire_count"],
                name=f"{district} Fire Count",
                marker_color="#dc2626", opacity=0.7,
            ), secondary_y=False)

            if "mean_frp" in d_fire.columns:
                fig_fire.add_trace(go.Scatter(
                    x=d_fire["year"], y=d_fire["mean_frp"],
                    name="Mean FRP (MW)",
                    mode="lines+markers",
                    line=dict(color="#7c3aed", width=2.5, dash="dot"),
                    marker=dict(size=7, color="#7c3aed"),
                ), secondary_y=True)

            if district2:
                d2_fire = get_fire_data(district2)
                if not d2_fire.empty:
                    fig_fire.add_trace(go.Scatter(
                        x=d2_fire["year"], y=d2_fire["fire_count"],
                        name=f"{district2} Fire Count",
                        mode="lines+markers",
                        line=dict(color="#7c3aed", width=2),
                        marker=dict(size=6),
                    ), secondary_y=False)

            fig_fire.update_layout(
                title=dict(text=f"Mann-Kendall: {trend_s} | Sen's slope = {slope:+.1f} fires/yr | p = {p_val:.4f} {'✅' if sig else '—'}", font=dict(size=12)),
                height=370,
                xaxis=dict(title="Year", tickmode="linear", dtick=1),
                legend=dict(orientation="h", y=-0.22),
                margin=dict(l=40, r=40, t=50, b=70),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            fig_fire.update_yaxes(title_text="VIIRS Fire Detections", secondary_y=False)
            fig_fire.update_yaxes(title_text="Mean FRP (MW)", secondary_y=True, showgrid=False)
            st.plotly_chart(fig_fire, use_container_width=True)

            # Heatmap of fire_count by year (all districts)
            st.markdown("<div class='section-hdr'>Region-Wide Fire Activity Heatmap</div>", unsafe_allow_html=True)
            pivot = fire_stats.pivot_table(index="district", columns="year", values="fire_count", aggfunc="sum")
            pivot = pivot.fillna(0)
            # Sort by total fire
            pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
            pivot = pivot.head(25)  # top 25

            fig_hm = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[str(c) for c in pivot.columns],
                y=pivot.index.tolist(),
                colorscale="YlOrRd",
                hoverongaps=False,
                hovertemplate="<b>%{y}</b><br>Year: %{x}<br>Fire Count: %{z:,.0f}<extra></extra>",
            ))
            fig_hm.update_layout(
                height=550,
                xaxis=dict(title="Year"),
                yaxis=dict(title="District"),
                margin=dict(l=120, r=20, t=20, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info(f"No fire data found for {district}.")

    with col_f2:
        st.subheader("Mann-Kendall Analysis")
        if not t_row.empty:
            r = t_row.iloc[0]
            sig_color = "#dc2626" if r["significant"] else "#64748b"
            st.markdown(f"""
            <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:16px; margin-bottom:12px'>
                <div style='font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em'>Trend Direction</div>
                <div style='font-size:22px; font-weight:700; color:#0f172a; font-family:JetBrains Mono'>{str(r["trend"]).title()}</div>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Sen's Slope",       f"{r['slope']:+.1f} fires/yr")
            st.metric("Kendall's τ",       f"{r['tau']:.3f}")
            st.metric("p-value",           f"{r['p_value']:.4f}")
            st.metric("Significant?",      "✅ Yes" if r["significant"] else "No (p≥0.05)")
        else:
            st.info("No trend data available.")

        st.markdown("<div class='section-hdr'>BRS Scorecard</div>", unsafe_allow_html=True)
        brs_row = master[master["district"] == district]
        if not brs_row.empty and pd.notna(brs_row["BRS"].values[0]):
            br = brs_row.iloc[0]
            thresholds = [("Critical", 75, "#dc2626"), ("High", 50, "#ea580c"), ("Moderate", 25, "#eab308")]
            fig_brs = go.Figure()
            for label, val, color in thresholds:
                fig_brs.add_hline(y=val, line_dash="dot", line_color=color,
                                  annotation_text=label, annotation_font_size=10, annotation_font_color=color)
            fig_brs.add_trace(go.Bar(x=["BRS Score"], y=[br["BRS"]],
                                     marker_color="#dc2626",
                                     text=[f"{br['BRS']:.1f}"], textposition="outside",
                                     textfont=dict(size=18, family="JetBrains Mono")))
            fig_brs.update_layout(height=250, yaxis=dict(range=[0, 115], title="Score"),
                                  showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                                  paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_brs, use_container_width=True)
            st.markdown(f"**Risk Class:** `{br['risk_class']}`")

        # Trend significance summary across all districts
        st.markdown("<div class='section-hdr'>Significant Trends Region-Wide</div>", unsafe_allow_html=True)
        sig_districts = trends[trends["significant"] == True]
        st.metric("Districts with significant trend", len(sig_districts))
        if not sig_districts.empty:
            for _, sr in sig_districts.iterrows():
                arrow = "📈" if sr["slope"] > 0 else "📉"
                st.markdown(f"**{sr['district']}** {arrow} {sr['slope']:+.1f} fires/yr (p={sr['p_value']:.4f})")


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — Bioenergy
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    # District detail card
    bio_row = master[master["district"] == district].iloc[0]
    bc1, bc2, bc3 = st.columns(3)
    bc1.metric("⚡ Net Bioenergy Potential",
               f"{bio_row['avg_NBP_GJ']/1e6:.2f} PJ/yr" if pd.notna(bio_row.get('avg_NBP_GJ')) else "N/A",
               help="Sum of GEP from rice straw (14.3 GJ/t) and wheat straw (17.5 GJ/t)")
    bc2.metric("🛢️ CBG Production",
               f"{bio_row['avg_CBG_tonnes']:,.0f} t/yr" if pd.notna(bio_row.get('avg_CBG_tonnes')) else "N/A",
               help="22 kg CBG per tonne recoverable residue")
    bc3.metric("💰 Annual Revenue",
               f"₹{bio_row['avg_revenue_crore']:.2f} Cr/yr" if pd.notna(bio_row.get('avg_revenue_crore')) else "N/A",
               help="At SATAT scheme price ₹46/kg CBG")

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.markdown("<div class='section-hdr'>CBG Revenue — Top 20 Districts</div>", unsafe_allow_html=True)
        top20 = (master.dropna(subset=["avg_revenue_crore"])
                 .nlargest(20, "avg_revenue_crore")
                 .sort_values("avg_revenue_crore", ascending=True))

        def bar_color(d):
            if d == district: return "#dc2626"
            if district2 and d == district2: return "#7c3aed"
            return "#3b82f6"

        fig_cbg = go.Figure(go.Bar(
            x=top20["avg_revenue_crore"],
            y=top20["district"],
            orientation="h",
            marker_color=[bar_color(d) for d in top20["district"]],
            text=["₹" + str(round(v, 1)) + " Cr" for v in top20["avg_revenue_crore"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Revenue: ₹%{x:.1f} Cr/yr<extra></extra>",
        ))
        fig_cbg.update_layout(height=520, xaxis=dict(title="Annual CBG Revenue (₹ Cr)"),
                              margin=dict(l=10, r=80, t=10, b=40), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cbg, use_container_width=True)

    with col_b2:
        st.markdown("<div class='section-hdr'>Revenue vs Residue Supply (All Districts)</div>", unsafe_allow_html=True)
        bio_all = master.dropna(subset=["avg_revenue_crore", "residue", "BPS"]).copy()
        bio_all["residue_kt"] = bio_all["residue"] / 1000

        fig_bub = px.scatter(
            bio_all,
            x="residue_kt", y="avg_revenue_crore",
            color="decision_zone",
            size="BPS",
            size_max=28,
            text="district",
            color_discrete_map=zone_color_map,
            labels={"residue_kt": "Recoverable Residue (000 t)", "avg_revenue_crore": "Revenue (₹ Cr/yr)", "decision_zone": "Zone"},
        )
        fig_bub.update_traces(textposition="top center", textfont=dict(size=8), marker=dict(opacity=0.8))
        fig_bub.update_layout(height=520, margin=dict(l=40, r=40, t=20, b=40), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bub, use_container_width=True)

    # Policy recommendation
    st.markdown("<div class='section-hdr'>Policy Recommendation</div>", unsafe_allow_html=True)
    recs_html = {
        "Plant Zone":   ("danger",  f"🏭 <b>{district} — Priority Plant Investment District.</b> BSI = {row['BSI']:.1f}/100, Residue = {row['residue']/1000 if pd.notna(row['residue']) else 'N/A':.1f}K t/yr. Estimated CBG revenue: ₹{row['avg_revenue_crore']:.1f} Cr/yr. <b>Action:</b> Apply immediately under SATAT. Engage FPOs for residue baling. Target commissioning within 2 years."),
        "Policy Zone":  ("warning", f"📋 <b>{district} — Policy Intervention Needed.</b> BSI is high ({row['BSI']:.1f}/100) but residue ({row['residue']/1000 if pd.notna(row['residue']) else 'N/A':.1f}K t/yr) is below commercial viability. <b>Action:</b> PUSA decomposer campaign, increase MSP for residue, support baling infrastructure. Reassess in 2–3 years."),
        "Low Priority": ("success", f"📡 <b>{district} — Below Priority Thresholds.</b> BSI = {row['BSI']:.1f}/100. <b>Action:</b> Maintain annual satellite monitoring via NASA FIRMS. Continue residue collection data. Reassess if burning trend increases."),
    }
    cls, msg = recs_html.get(row["decision_zone"], ("", ""))
    st.markdown(f"<div class='insight-card {cls}' style='font-size:14px; padding:16px 20px'>{msg}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — Environmental Profile
# ══════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader(f"Environmental Feature Profile: {district}")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        # Horizontal bar chart comparing district to min/mean/max
        feat_display = {
            "fire_count": ("Fire Count (fires/yr)", "#dc2626"),
            "residue":    ("Residue (tonnes/yr)",   "#16a34a"),
            "avg_temp":   ("Avg Temp (°C)",          "#ea580c"),
            "rainfall":   ("Rainfall (mm)",           "#3b82f6"),
        }
        fig_feat = go.Figure()
        for col, (label, color) in feat_display.items():
            d_val  = float(master.loc[master["district"] == district, col].values[0])
            mn     = float(master[col].min())
            mx     = float(master[col].max())
            avg    = float(master[col].mean())
            normed = (d_val - mn) / (mx - mn) if mx != mn else 0

            fig_feat.add_trace(go.Bar(
                name=label,
                x=[normed],
                y=[label],
                orientation="h",
                marker_color=color,
                text=[f"{d_val:,.1f}"],
                textposition="outside",
                width=0.5,
            ))
            # Range bar (min-max as thin grey)
            fig_feat.add_trace(go.Bar(
                name=f"{label} range",
                x=[1.0],
                y=[label],
                orientation="h",
                marker_color="#e2e8f0",
                opacity=0.4,
                width=0.5,
                showlegend=False,
            ))

        fig_feat.update_layout(
            barmode="overlay",
            height=320,
            xaxis=dict(title="Normalised Value (0=min, 1=max)", range=[0, 1.3], tickformat=".0%"),
            yaxis=dict(title=""),
            showlegend=False,
            margin=dict(l=10, r=80, t=10, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_feat, use_container_width=True)

        # Feature values table
        st.markdown("<div class='section-hdr'>Raw Feature Values vs Region</div>", unsafe_allow_html=True)
        rows_feat = []
        for col_k, (label, _) in feat_display.items():
            d_val = float(master.loc[master["district"] == district, col_k].values[0])
            rows_feat.append({
                "Feature": label,
                district:       round(d_val, 1),
                "Region Min":   round(float(master[col_k].min()), 1),
                "Region Mean":  round(float(master[col_k].mean()), 1),
                "Region Max":   round(float(master[col_k].max()), 1),
                "Percentile":   f"{float(master[col_k].rank(pct=True)[master['district'] == district].values[0])*100:.0f}th",
            })
        st.dataframe(pd.DataFrame(rows_feat).set_index("Feature"), use_container_width=True)

    with col_e2:
        # Radar with optional comparison
        fig_radar2 = go.Figure()
        features_r2 = ["fire_count", "residue", "avg_temp", "rainfall"]
        labels_r2   = ["Fire Count", "Residue", "Avg Temp", "Rainfall"]

        def get_normed_vals(d):
            return [float(safe_norm(c)[master["district"] == d].values[0]) for c in features_r2]

        d1_vals = get_normed_vals(district)
        avg_vals2 = [float(safe_norm(c).mean()) for c in features_r2]

        for name, vals, lc, fc in [(district, d1_vals, "#dc2626", "rgba(220,38,38,0.18)"),
                                    ("Region Avg", avg_vals2, "#3b82f6", "rgba(59,130,246,0.08)")]:
            fig_radar2.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=labels_r2 + [labels_r2[0]],
                fill="toself", name=name,
                line=dict(color=lc, width=2.5), fillcolor=fc,
            ))

        if district2 and row2 is not None:
            d2_vals2 = get_normed_vals(district2)
            fig_radar2.add_trace(go.Scatterpolar(
                r=d2_vals2 + [d2_vals2[0]], theta=labels_r2 + [labels_r2[0]],
                fill="toself", name=district2,
                line=dict(color="#7c3aed", width=2, dash="dot"), fillcolor="rgba(124,58,237,0.1)",
            ))

        fig_radar2.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            legend=dict(y=-0.15, orientation="h"),
            height=340,
            margin=dict(l=40, r=40, t=10, b=60),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar2, use_container_width=True)

        # K-Means zone
        st.markdown("<div class='section-hdr'>K-Means Zone Assignment</div>", unsafe_allow_html=True)
        zone_val = row.get("zone", "Unknown")
        zone_desc = {
            "High Stress / High Opportunity": ("danger",  "🔴 **High Stress / High Opportunity** — High fire count + high residue + warm, dry climate. Highest burning pressure AND highest feedstock. Priority for immediate intervention."),
            "Moderate Zone":                  ("warning", "🟡 **Moderate Zone** — Intermediate environmental values. Burning present but not extreme. Monitor and build supply chain."),
            "Low Priority Zone":              ("success", "🟢 **Low Priority Zone** — Lower fire count, higher rainfall suppressing burning. Baseline monitoring only."),
        }
        cls2, desc2 = zone_desc.get(str(zone_val), ("", f"Zone: {zone_val}"))
        st.markdown(f"<div class='insight-card {cls2}'>{desc2}</div>", unsafe_allow_html=True)

        # PCA weights
        st.markdown("<div class='section-hdr'>PCA BSI Weights (NB14)</div>", unsafe_allow_html=True)
        pca_df = pd.DataFrame({
            "Feature":            ["Rainfall (inverted)", "Avg Temperature", "Fire Count", "Residue"],
            "|PC1 Loading|":      [0.563, 0.537, 0.454, 0.434],
            "Weight":             ["28.3%", "27.0%", "22.9%", "21.8%"],
            "Direction":          ["↑ dry = more risk", "↑ hot = more risk", "↑ more fires", "↑ more feedstock"],
        })
        st.dataframe(pca_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 6 — All Districts
# ══════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Complete District Rankings & Explorer")

    # Top-line zone summary
    col_s1, col_s2, col_s3 = st.columns(3)
    zone_counts = master["decision_zone"].value_counts()
    col_s1.metric("🔴 Plant Zones",   zone_counts.get("Plant Zone", 0),   help="Both BSI and Residue above 70th percentile.")
    col_s2.metric("🟠 Policy Zones",  zone_counts.get("Policy Zone", 0),  help="High BSI but residue below threshold.")
    col_s3.metric("⚫ Low Priority",  zone_counts.get("Low Priority", 0), help="Below environmental thresholds.")

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # Searchable table
    search = st.text_input("🔍 Search district", "")
    sort_col = st.selectbox("Sort by", ["BSI", "BRS", "BPS", "avg_revenue_crore", "fire_count", "residue"], index=0)
    sort_asc = st.checkbox("Ascending", value=False)

    all_cols = ["district", "BSI", "bsi_rank", "BRS", "risk_class", "BPS",
                "avg_revenue_crore", "fire_count", "residue", "avg_temp", "rainfall", "zone", "decision_zone"]
    display_df = (filt[all_cols]
                  .dropna(subset=["BSI"])
                  .sort_values(sort_col, ascending=sort_asc)
                  .reset_index(drop=True))
    display_df.index += 1

    if search:
        display_df = display_df[display_df["district"].str.contains(search, case=False)]

    display_df.columns = ["District", "BSI", "BSI Rank", "BRS", "BRS Class", "BPS",
                          "Revenue (₹ Cr/yr)", "Fire Count", "Residue (t/yr)",
                          "Avg Temp (°C)", "Rainfall (mm)", "Env Zone", "Decision Zone"]

    st.dataframe(
        display_df.style
            .format({"BSI": "{:.1f}", "BRS": "{:.1f}", "BPS": "{:.1f}",
                     "Revenue (₹ Cr/yr)": "₹{:.1f}", "Fire Count": "{:,.0f}",
                     "Residue (t/yr)": "{:,.0f}", "Avg Temp (°C)": "{:.1f}", "Rainfall (mm)": "{:,.0f}"})
            .background_gradient(subset=["BSI"], cmap="YlOrRd")
            .background_gradient(subset=["BRS"], cmap="Reds")
            .background_gradient(subset=["BPS"], cmap="Greens")
            .background_gradient(subset=["Revenue (₹ Cr/yr)"], cmap="BuGn"),
        use_container_width=True, height=520,
    )

    st.markdown(f"*Showing {len(display_df)} of {len(master)} districts based on active filters.*")
    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # Dual pie charts
    col_z1, col_z2 = st.columns(2)
    with col_z1:
        fig_pie = px.pie(
            zone_counts.reset_index().rename(columns={"index": "Zone", "decision_zone": "Districts"}),
            names="decision_zone", values="count",
            color="decision_zone",
            color_discrete_map=zone_color_map,
            title="Decision Zone Distribution (BSI × Residue)",
            hole=0.45,
        )
        fig_pie.update_traces(textinfo="label+percent+value")
        fig_pie.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_z2:
        env_counts = master["zone"].value_counts()
        env_color_map = {
            "High Stress / High Opportunity": "#dc2626",
            "Moderate Zone": "#f59e0b",
            "Low Priority Zone": "#16a34a",
        }
        fig_pie2 = px.pie(
            env_counts.reset_index().rename(columns={"index": "Zone", "zone": "Districts"}),
            names="zone", values="count",
            color="zone",
            color_discrete_map=env_color_map,
            title="K-Means Environmental Zone Distribution",
            hole=0.45,
        )
        fig_pie2.update_traces(textinfo="label+percent+value")
        fig_pie2.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie2, use_container_width=True)

    # Correlation heatmap
    st.markdown("<div class='section-hdr'>Score Correlation Matrix</div>", unsafe_allow_html=True)
    corr_cols = ["BSI", "BRS", "BPS", "avg_revenue_crore", "fire_count", "residue", "avg_temp", "rainfall"]
    corr_labels = ["BSI", "BRS", "BPS", "Revenue", "Fire Count", "Residue", "Temp", "Rainfall"]
    corr_df = master[corr_cols].dropna().corr()

    fig_corr = go.Figure(go.Heatmap(
        z=corr_df.values,
        x=corr_labels, y=corr_labels,
        colorscale="RdBu", zmid=0,
        text=corr_df.round(2).values,
        texttemplate="%{text}",
        hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
    ))
    fig_corr.update_layout(
        height=400,
        margin=dict(l=80, r=20, t=20, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:12px; padding:8px 0 16px; font-family:JetBrains Mono'>
    🌾 Crop Residue Burning & Bioenergy Dashboard &nbsp;|&nbsp;
    NASA FIRMS VIIRS 2015–2023 &nbsp;|&nbsp;
    Punjab & Haryana Statistical Abstracts &nbsp;|&nbsp;
    NASA POWER API &nbsp;|&nbsp;
    Module C: Environmental K-Means + PCA-BSI
</div>
""", unsafe_allow_html=True)
