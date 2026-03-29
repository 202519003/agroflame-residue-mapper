"""
Crop Residue Burning & Bioenergy Dashboard
Punjab & Haryana | 47 Districts | 2015-2023

Module C (revised): Environmental clustering (K-Means k=3) + PCA-derived
Burning Severity Index (BSI) using fire_count, residue, avg_temp, rainfall.
Decision classification: BSI × Residue at 70th-percentile thresholds.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

# ── Path resolution (works locally AND on Streamlit Cloud) ────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed")
MAPS = os.path.join(ROOT, "outputs", "maps")

st.set_page_config(
    page_title="Crop Residue & Bioenergy Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .zone-banner {
        padding: 10px 18px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 1rem;
        color: white;
    }
    .section-header {
        font-size: 13px;
        font-weight: 600;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.2rem 0 0.4rem 0;
    }
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────
@st.cache_data
def load_all_data():
    # Load each processed CSV
    brs    = pd.read_csv(os.path.join(DATA, "burning_risk_scores.csv"))
    bsi    = pd.read_csv(os.path.join(DATA, "bsi_scores.csv"))
    bio    = pd.read_csv(os.path.join(DATA, "bioenergy_scores.csv"))
    env    = pd.read_csv(os.path.join(DATA, "env_features.csv"))
    clust  = pd.read_csv(os.path.join(DATA, "district_clusters.csv"))
    fire   = pd.read_csv(os.path.join(DATA, "fire_stats.csv"))
    trends = pd.read_csv(os.path.join(DATA, "fire_trends.csv"))

    # Standardise district name casing across all files
    for df in [brs, bsi, bio, env, clust, fire, trends]:
        df["district"] = df["district"].str.strip().str.title()

    # Build master table — one row per district
    master = (
        bsi
        .merge(brs[["district", "BRS", "risk_class"]], on="district", how="left")
        .merge(
            bio[["district", "BPS", "avg_revenue_crore", "avg_NBP_GJ", "avg_CBG_tonnes"]],
            on="district", how="left"
        )
        .merge(
            env[["district", "fire_count", "residue", "avg_temp", "rainfall"]],
            on="district", how="left"
        )
        .merge(clust[["district", "zone"]], on="district", how="left")
    )

    # Decision zone: BSI × Residue at 70th-percentile thresholds
    bsi_thresh = master["BSI"].quantile(0.70)
    res_thresh = master["residue"].quantile(0.70)

    def assign_zone(row):
        if row["BSI"] >= bsi_thresh and row["residue"] >= res_thresh:
            return "Plant Zone"
        elif row["BSI"] >= bsi_thresh:
            return "Policy Zone"
        else:
            return "Low Priority"

    master["decision_zone"] = master.apply(assign_zone, axis=1)
    master["bsi_thresh"]    = bsi_thresh
    master["res_thresh"]    = res_thresh

    # BSI rank (1 = most severe)
    master["bsi_rank"] = master["BSI"].rank(ascending=False, method="min").astype(int)

    return master, fire, trends


master, fire_stats, trends = load_all_data()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌾 Dashboard Controls")
    st.markdown("---")

    district_list = sorted(master["district"].dropna().unique().tolist())
    default_idx   = district_list.index("Sangrur") if "Sangrur" in district_list else 0
    district = st.selectbox("Select District", district_list, index=default_idx)

    st.markdown("---")
    st.markdown("**About Module C**")
    st.caption(
        "This dashboard uses **revised Module C**: instead of socioeconomic data "
        "(NSSO income, Agri Census, FPO, PMFBY), environmental features are used "
        "(fire count, residue, temperature, rainfall) to cluster districts by "
        "K-Means (k=3) and derive a PCA-weighted Burning Severity Index (BSI)."
    )
    st.markdown("---")
    st.markdown("**Decision Logic**")
    st.caption(
        "🔴 **Plant Zone** — BSI ≥ 70th pct AND Residue ≥ 70th pct  \n"
        "🟠 **Policy Zone** — BSI ≥ 70th pct only  \n"
        "⚫ **Low Priority** — Below BSI threshold"
    )
    st.markdown("---")
    st.caption("Data: NASA FIRMS VIIRS 2015–2023 | Punjab & Haryana")

# ── Get selected district row ─────────────────────────────────────────────
row = master[master["district"] == district].iloc[0]

# ── Page header ──────────────────────────────────────────────────────────
st.title(f"🌾 {district}")
st.markdown(f"**Punjab & Haryana Crop Residue Burning & Bioenergy Analysis** | BSI Rank #{int(row['bsi_rank'])} of 47 districts")

# Zone banner
zone_cfg = {
    "Plant Zone":   ("#c0392b", "🔴 PLANT ZONE — Both BSI and Residue above threshold. Build CBG plant now."),
    "Policy Zone":  ("#e67e22", "🟠 POLICY ZONE — High burning severity but residue supply insufficient. Policy intervention first."),
    "Low Priority": ("#27ae60", "⚫ LOW PRIORITY — Below environmental threshold. Annual satellite monitoring."),
}
z_color, z_label = zone_cfg.get(row["decision_zone"], ("#888", row["decision_zone"]))
st.markdown(
    f"<div class='zone-banner' style='background:{z_color}'>{z_label}</div>",
    unsafe_allow_html=True,
)

# ── KPI metric row ────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Key Indicators</div>", unsafe_allow_html=True)
c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric(
    "BSI Score",
    f"{row['BSI']:.1f}/100",
    help="PCA-derived Burning Severity Index (NB14). Weights: rainfall 28.3%, avg_temp 27.0%, fire_count 22.9%, residue 21.8%.",
)
c2.metric(
    "BRS Score",
    f"{row['BRS']:.1f}/100" if pd.notna(row["BRS"]) else "N/A",
    delta=str(row.get("risk_class", "")),
    help="Burning Risk Score (NB06): fire frequency 40% + FRP intensity 30% + Mann-Kendall slope 30%.",
)
c3.metric(
    "Bioenergy Score",
    f"{row['BPS']:.1f}/100" if pd.notna(row["BPS"]) else "N/A",
    help="Bioenergy Potential Score (NB10): normalised Net Bioenergy Potential in GJ.",
)
c4.metric(
    "CBG Revenue",
    f"₹{row['avg_revenue_crore']:.1f} Cr/yr" if pd.notna(row["avg_revenue_crore"]) else "N/A",
    help="Average annual Compressed Biogas revenue at ₹46/kg (SATAT scheme price).",
)
c5.metric(
    "Mean Fire Count",
    f"{row['fire_count']:,.0f}" if pd.notna(row["fire_count"]) else "N/A",
    help="Mean annual VIIRS fire detections 2015–2023.",
)
c6.metric(
    "Recoverable Residue",
    f"{row['residue']/1000:.1f}K t" if pd.notna(row["residue"]) else "N/A",
    help="Mean annual recoverable crop residue in tonnes (NB09: RPR × burn fraction × 70% recovery efficiency).",
)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Maps & Zones",
    "📈 Fire Trend",
    "🌿 Bioenergy",
    "🔬 Environmental Profile",
    "📊 All Districts",
])

# ─────────────────────────────────────────────
# TAB 1 — Maps & Zones
# ─────────────────────────────────────────────
with tab1:
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.subheader("Burning Severity Index (BSI) Map")
        bsi_map_path = os.path.join(MAPS, "15_bsi_map.png")
        if os.path.exists(bsi_map_path):
            st.image(bsi_map_path, use_container_width=True)
            st.caption(
                "PCA-weighted Burning Severity Index (0–100). "
                "Weights derived from PC1 loadings: rainfall 28.3%, avg_temp 27.0%, "
                "fire_count 22.9%, residue 21.8%. Rainfall is inverted (high rainfall suppresses burning)."
            )
        else:
            st.info("BSI map PNG not found. Run NB15 to generate: `15_bsi_map.png`")

    with col_m2:
        st.subheader("Environmental Intervention Zones (K-Means k=3)")
        clust_map_path = os.path.join(MAPS, "15_cluster_map.png")
        if os.path.exists(clust_map_path):
            st.image(clust_map_path, use_container_width=True)
            st.caption(
                "K-Means clustering on 4 standardised environmental features. "
                "🔴 High Stress / High Opportunity (n=10) — high fire + high residue, dry climate. "
                "🟡 Moderate Zone (n=17). "
                "🟢 Low Priority Zone (n=20)."
            )
        else:
            st.info("Cluster map PNG not found. Run NB15 to generate: `15_cluster_map.png`")

    st.markdown("---")

    col_m3, col_m4 = st.columns(2)

    with col_m3:
        st.subheader("Final Decision Map")
        dec_map_path = os.path.join(MAPS, "16_final_decision_map.png")
        if os.path.exists(dec_map_path):
            st.image(dec_map_path, use_container_width=True)
            st.caption(
                "🔴 Plant Zone (9 districts): BSI ≥ 70th pct AND Residue ≥ 70th pct. "
                "🟠 Policy Zone (4 districts): high BSI, lower residue. "
                "⚫ Low Priority (30 districts)."
            )
        else:
            st.info("Decision map PNG not found. Run NB16 to generate: `16_final_decision_map.png`")

    with col_m4:
        st.subheader("Top-5 Highest BSI Districts")
        top5_map_path = os.path.join(MAPS, "15_top5_burning.png")
        if os.path.exists(top5_map_path):
            st.image(top5_map_path, use_container_width=True)
            st.caption(
                "Top 5 by BSI: Sangrur (100.0), Bathinda (95.1), Muktsar (90.0), "
                "Fazilka (88.5), Sirsa (84.4). All are south-west Punjab / north-west Haryana."
            )
        else:
            st.info("Top-5 map PNG not found. Run NB15 to generate: `15_top5_burning.png`")

    st.markdown("---")
    st.subheader("Decision Space: BSI vs Recoverable Residue")
    st.caption(
        "Each point is a district. Dashed lines = 70th-percentile thresholds. "
        "Top-right quadrant = Plant Zone. Selected district shown as ★."
    )

    plot_data = master.dropna(subset=["BSI", "residue"]).copy()
    plot_data["residue_kt"] = plot_data["residue"] / 1000

    zone_color_map = {
        "Plant Zone":   "#c0392b",
        "Policy Zone":  "#e67e22",
        "Low Priority": "#95a5a6",
    }

    fig_scatter = px.scatter(
        plot_data,
        x="BSI",
        y="residue_kt",
        color="decision_zone",
        text="district",
        color_discrete_map=zone_color_map,
        hover_data={"BRS": ":.1f", "BPS": ":.1f", "avg_revenue_crore": ":.1f", "residue_kt": False},
        custom_data=["district", "BRS", "BPS", "avg_revenue_crore"],
        labels={"BSI": "Burning Severity Index (BSI)", "residue_kt": "Recoverable Residue (000 tonnes/yr)"},
    )
    fig_scatter.update_traces(
        textposition="top center",
        marker=dict(size=8, opacity=0.85),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "BSI: %{x:.1f}<br>"
            "Residue: %{y:.1f}K t<br>"
            "BRS: %{customdata[1]:.1f}<br>"
            "BPS: %{customdata[2]:.1f}<br>"
            "Revenue: ₹%{customdata[3]:.1f} Cr/yr<extra></extra>"
        ),
    )

    bsi_t = float(row["bsi_thresh"])
    res_t = float(row["res_thresh"]) / 1000

    fig_scatter.add_vline(x=bsi_t, line_dash="dash", line_color="#666",
                          annotation_text=f"BSI p70 = {bsi_t:.1f}", annotation_position="top right")
    fig_scatter.add_hline(y=res_t, line_dash="dash", line_color="#666",
                          annotation_text=f"Residue p70 = {res_t:.1f}K t", annotation_position="right")

    # Star marker for selected district
    sel = master[master["district"] == district]
    if not sel.empty:
        fig_scatter.add_trace(go.Scatter(
            x=sel["BSI"],
            y=sel["residue"] / 1000,
            mode="markers+text",
            text=[f"★ {district}"],
            textposition="top center",
            marker=dict(size=18, color="black", symbol="star", opacity=0.9),
            name=f"Selected: {district}",
            hovertemplate=f"<b>{district}</b><br>BSI: {sel['BSI'].values[0]:.1f}<extra></extra>",
        ))

    fig_scatter.update_layout(
        height=480,
        legend_title="Decision Zone",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 2 — Fire Trend
# ─────────────────────────────────────────────
with tab2:
    col_f1, col_f2 = st.columns([2, 1])

    with col_f1:
        st.subheader(f"Annual Fire Count 2015–2023: {district}")

        d_fire = fire_stats[fire_stats["district"] == district].sort_values("year")
        t_row  = trends[trends["district"] == district]

        if not d_fire.empty:
            slope = float(t_row["slope"].values[0]) if not t_row.empty else 0.0
            sig   = bool(t_row["significant"].values[0]) if not t_row.empty else False
            trend_str = t_row["trend"].values[0] if not t_row.empty else "no trend"
            p_val     = float(t_row["p_value"].values[0]) if not t_row.empty else 1.0

            subtitle = (
                f"Mann-Kendall: {trend_str} | Sen's slope = {slope:+.1f} fires/yr | "
                f"p = {p_val:.4f} {'✅ significant' if sig else '(not significant at p<0.05)'}"
            )

            fig_fire = go.Figure()
            fig_fire.add_bar(
                x=d_fire["year"], y=d_fire["fire_count"],
                marker_color="#e74c3c", opacity=0.75, name="Fire count",
            )
            fig_fire.add_scatter(
                x=d_fire["year"], y=d_fire["fire_count"],
                mode="lines+markers",
                line=dict(color="#922b21", width=2),
                marker=dict(size=6, color="#922b21"),
                name="Trend line",
            )
            fig_fire.update_layout(
                title=dict(text=subtitle, font=dict(size=12)),
                height=340,
                xaxis=dict(title="Year", tickmode="linear", dtick=1),
                yaxis=dict(title="VIIRS Fire Detections"),
                legend=dict(orientation="h", y=-0.2),
                margin=dict(l=40, r=20, t=50, b=60),
            )
            st.plotly_chart(fig_fire, use_container_width=True)

            # Fire stats table
            st.markdown("<div class='section-header'>Year-by-Year Stats</div>", unsafe_allow_html=True)
            fire_table = d_fire[["year", "fire_count", "mean_frp", "fire_days", "onset_doy", "peak_week"]].copy()
            fire_table.columns = ["Year", "Fire Count", "Mean FRP (MW)", "Fire Days", "Onset DOY", "Peak Week"]
            fire_table["Mean FRP (MW)"] = fire_table["Mean FRP (MW)"].round(2)
            st.dataframe(
                fire_table.set_index("Year"),
                use_container_width=True,
            )
        else:
            st.info(f"No fire data found for {district}.")

    with col_f2:
        st.subheader("Mann-Kendall Summary")

        if not t_row.empty:
            r = t_row.iloc[0]
            st.metric("Trend Direction", str(r["trend"]).title())
            st.metric("Sen's Slope", f"{r['slope']:+.1f} fires/yr")
            st.metric("Kendall's Tau", f"{r['tau']:.3f}")
            st.metric("p-value", f"{r['p_value']:.4f}")
            st.metric("Statistically Significant", "Yes ✅" if r["significant"] else "No")
        else:
            st.info("No trend data available.")

        st.markdown("---")
        st.subheader("BRS Breakdown")
        st.caption("Burning Risk Score components (NB06)")

        brs_row = master[master["district"] == district]
        if not brs_row.empty and pd.notna(brs_row["BRS"].values[0]):
            br = brs_row.iloc[0]
            fig_brs = go.Figure(go.Bar(
                x=["BRS Score"],
                y=[br["BRS"]],
                marker_color="#e74c3c",
                text=[f"{br['BRS']:.1f}"],
                textposition="outside",
            ))
            fig_brs.add_hline(y=75, line_dash="dot", line_color="#c0392b",
                              annotation_text="Critical ≥75")
            fig_brs.add_hline(y=50, line_dash="dot", line_color="#e67e22",
                              annotation_text="High ≥50")
            fig_brs.add_hline(y=25, line_dash="dot", line_color="#f1c40f",
                              annotation_text="Moderate ≥25")
            fig_brs.update_layout(
                height=250,
                yaxis=dict(range=[0, 110], title="Score"),
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_brs, use_container_width=True)
            st.markdown(f"**Risk Class:** `{br['risk_class']}`")


# ─────────────────────────────────────────────
# TAB 3 — Bioenergy
# ─────────────────────────────────────────────
with tab3:
    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.subheader("CBG Revenue — Top 20 Districts")
        st.caption("Annual Compressed Biogas revenue at ₹46/kg (SATAT scheme). Selected district in red.")

        top20 = (
            master.dropna(subset=["avg_revenue_crore"])
            .nlargest(20, "avg_revenue_crore")
            .sort_values("avg_revenue_crore", ascending=True)
        )
        colors = ["#c0392b" if d == district else "#3498db" for d in top20["district"]]

        fig_cbg = go.Figure(go.Bar(
            x=top20["avg_revenue_crore"],
            y=top20["district"],
            orientation="h",
            marker_color=colors,
            text=top20["avg_revenue_crore"].round(1).astype(str) + " Cr",
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Revenue: ₹%{x:.1f} Cr/yr<extra></extra>",
        ))
        fig_cbg.update_layout(
            height=520,
            xaxis=dict(title="Annual CBG Revenue (₹ Crore)"),
            margin=dict(l=10, r=60, t=10, b=40),
        )
        st.plotly_chart(fig_cbg, use_container_width=True)

    with col_b2:
        st.subheader("Bioenergy Potential Score — All Districts")
        st.caption("BPS (0–100) normalised from Net Bioenergy Potential in GJ. Selected district in red.")

        all_bio = master.dropna(subset=["BPS"]).sort_values("BPS", ascending=True)
        colors2 = ["#c0392b" if d == district else "#2ecc71" for d in all_bio["district"]]

        fig_bps = go.Figure(go.Bar(
            x=all_bio["BPS"],
            y=all_bio["district"],
            orientation="h",
            marker_color=colors2,
            hovertemplate="<b>%{y}</b><br>BPS: %{x:.1f}<extra></extra>",
        ))
        fig_bps.update_layout(
            height=900,
            xaxis=dict(title="Bioenergy Potential Score (0–100)"),
            margin=dict(l=10, r=20, t=10, b=40),
        )
        st.plotly_chart(fig_bps, use_container_width=True)

    st.markdown("---")

    # District bioenergy detail card
    st.subheader(f"Bioenergy Details: {district}")
    bio_row = master[master["district"] == district].iloc[0]

    bc1, bc2, bc3 = st.columns(3)
    bc1.metric("Net Bioenergy Potential",
               f"{bio_row['avg_NBP_GJ']/1e6:.2f} PJ/yr" if pd.notna(bio_row.get('avg_NBP_GJ')) else "N/A",
               help="Sum of GEP from rice straw (14.3 GJ/t) and wheat straw (17.5 GJ/t)")
    bc2.metric("CBG Production",
               f"{bio_row['avg_CBG_tonnes']:.0f} t/yr" if pd.notna(bio_row.get('avg_CBG_tonnes')) else "N/A",
               help="At 22 kg CBG per tonne recoverable residue")
    bc3.metric("Annual Revenue",
               f"₹{bio_row['avg_revenue_crore']:.2f} Cr/yr" if pd.notna(bio_row.get('avg_revenue_crore')) else "N/A",
               help="At SATAT scheme price of ₹46/kg CBG")

    st.markdown("---")

    # Policy recommendation
    st.subheader("📋 Policy Recommendation")
    recs = {
        "Plant Zone": (
            f"**{district} is a Priority Plant Investment District.** "
            f"With a BSI of {row['BSI']:.1f}/100 and recoverable residue of "
            f"{row['residue']/1000:.1f}K t/yr, this district combines severe environmental "
            f"burning pressure with abundant feedstock supply. "
            f"Estimated CBG revenue: ₹{row['avg_revenue_crore']:.1f} crore/year. "
            f"**Recommended action:** Apply immediately under SATAT scheme. "
            f"Engage FPOs for residue baling and logistics. Target commissioning within 2 years."
        ),
        "Policy Zone": (
            f"**{district} requires policy intervention before plant investment.** "
            f"BSI is high ({row['BSI']:.1f}/100) indicating severe burning pressure, but "
            f"recoverable residue ({row['residue']/1000:.1f}K t/yr) is below the commercial "
            f"viability threshold. "
            f"**Recommended action:** Launch PUSA decomposer campaign. Increase MSP for residue "
            f"purchase. Support baling infrastructure and FPO formation. "
            f"Revisit plant investment in 2–3 years once supply chain is established."
        ),
        "Low Priority": (
            f"**{district} is currently below priority thresholds for plant investment.** "
            f"Environmental burning pressure is moderate (BSI: {row['BSI']:.1f}/100). "
            f"**Recommended action:** Maintain annual satellite monitoring via NASA FIRMS. "
            f"Continue residue collection data gathering. "
            f"Reassess if burning trend increases significantly in coming seasons."
        ),
    }
    zone = row.get("decision_zone", "Low Priority")
    rec_color = {"Plant Zone": "error", "Policy Zone": "warning", "Low Priority": "info"}
    getattr(st, rec_color.get(zone, "info"))(recs.get(zone, "Recommendation pending."))


# ─────────────────────────────────────────────
# TAB 4 — Environmental Profile
# ─────────────────────────────────────────────
with tab4:
    st.subheader(f"Environmental Feature Profile: {district}")
    st.caption(
        "How this district compares on all 4 environmental features "
        "that drive the BSI score and K-Means zone assignment."
    )

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        # Radar chart — district vs state average
        features_radar = ["fire_count", "residue", "avg_temp", "rainfall"]
        labels_radar   = ["Fire Count", "Residue (t)", "Avg Temp", "Rainfall"]

        def safe_norm(col):
            mn, mx = master[col].min(), master[col].max()
            if mx == mn:
                return master[col] * 0
            return (master[col] - mn) / (mx - mn)

        dist_vals = []
        avg_vals  = []
        for col in features_radar:
            normed = safe_norm(col)
            dist_row = normed[master["district"] == district]
            dist_vals.append(float(dist_row.values[0]) if not dist_row.empty else 0.0)
            avg_vals.append(float(normed.mean()))

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=dist_vals + [dist_vals[0]],
            theta=labels_radar + [labels_radar[0]],
            fill="toself",
            name=district,
            line=dict(color="#e74c3c", width=2),
            fillcolor="rgba(192,57,43,0.2)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=avg_vals + [avg_vals[0]],
            theta=labels_radar + [labels_radar[0]],
            fill="toself",
            name="Region Average",
            line=dict(color="#3498db", width=2, dash="dot"),
            fillcolor="rgba(52,152,219,0.1)",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Feature profile (normalised 0–1)",
            height=380,
            showlegend=True,
            legend=dict(y=-0.15, orientation="h"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_e2:
        # Bar chart: district vs min/mean/max
        st.markdown("<div class='section-header'>Raw Feature Values vs Region</div>", unsafe_allow_html=True)

        feat_display = {
            "fire_count": ("Mean Annual Fire Count", "fires/yr"),
            "residue":    ("Recoverable Residue",    "tonnes/yr"),
            "avg_temp":   ("Average Temperature",    "°C"),
            "rainfall":   ("Total Rainfall",         "mm"),
        }

        rows_feat = []
        for col, (label, unit) in feat_display.items():
            dist_val = master.loc[master["district"] == district, col].values
            dv = float(dist_val[0]) if len(dist_val) > 0 and pd.notna(dist_val[0]) else None
            rows_feat.append({
                "Feature": label,
                "Unit": unit,
                f"{district}": round(dv, 1) if dv else "N/A",
                "Region Min": round(float(master[col].min()), 1),
                "Region Mean": round(float(master[col].mean()), 1),
                "Region Max": round(float(master[col].max()), 1),
            })

        feat_df = pd.DataFrame(rows_feat).set_index("Feature")
        st.dataframe(feat_df, use_container_width=True)

        st.markdown("---")
        st.markdown("<div class='section-header'>K-Means Zone Assignment</div>", unsafe_allow_html=True)

        zone_val = row.get("zone", "Unknown")
        zone_desc = {
            "High Stress / High Opportunity": (
                "🔴 **High Stress / High Opportunity** — High fire count + high residue + "
                "warm, dry climate. Highest burning pressure AND highest feedstock potential. "
                "Priority for immediate intervention."
            ),
            "Moderate Zone": (
                "🟡 **Moderate Zone** — Intermediate values across all features. "
                "Burning present but not extreme. Monitor and build supply chain."
            ),
            "Low Priority Zone": (
                "🟢 **Low Priority Zone** — Lower fire count, lower residue, "
                "or higher rainfall suppressing burning. Baseline monitoring only."
            ),
        }
        st.info(zone_desc.get(str(zone_val), f"Zone: {zone_val}"))

        st.markdown("---")
        st.markdown("<div class='section-header'>PCA BSI Weights Used</div>", unsafe_allow_html=True)
        pca_weights = pd.DataFrame({
            "Feature":  ["Rainfall (inverted)", "Avg Temperature", "Fire Count", "Residue"],
            "PC1 |Loading|": [0.563, 0.537, 0.454, 0.434],
            "Normalised Weight": ["28.3%", "27.0%", "22.9%", "21.8%"],
            "Direction": ["↑ dry = more risk", "↑ hot = more risk", "↑ more fires", "↑ more feedstock"],
        })
        st.dataframe(pca_weights, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# TAB 5 — All Districts
# ─────────────────────────────────────────────
with tab5:
    st.subheader("Complete District Rankings")
    st.caption("Sorted by BSI (highest = most severe environmental burning). Click column headers to re-sort.")

    all_cols = [
        "district", "BSI", "bsi_rank", "BRS", "risk_class",
        "BPS", "avg_revenue_crore",
        "fire_count", "residue",
        "avg_temp", "rainfall",
        "zone", "decision_zone",
    ]
    display_df = master[all_cols].dropna(subset=["BSI"]).sort_values("BSI", ascending=False).reset_index(drop=True)
    display_df.index += 1

    display_df.columns = [
        "District", "BSI", "BSI Rank", "BRS", "BRS Class",
        "BPS", "Revenue (₹ Cr/yr)",
        "Fire Count", "Residue (t/yr)",
        "Avg Temp (°C)", "Rainfall (mm)",
        "Env Zone", "Decision Zone",
    ]

    numeric_cols = ["BSI", "BRS", "BPS", "Revenue (₹ Cr/yr)"]

    # Search filter
    search = st.text_input("🔍 Filter by district name", "")
    if search:
        display_df = display_df[display_df["District"].str.contains(search, case=False)]

    st.dataframe(
        display_df.style
            .format({
                "BSI":              "{:.1f}",
                "BRS":              "{:.1f}",
                "BPS":              "{:.1f}",
                "Revenue (₹ Cr/yr)": "₹{:.1f}",
                "Fire Count":       "{:,.0f}",
                "Residue (t/yr)":   "{:,.0f}",
                "Avg Temp (°C)":    "{:.1f}",
                "Rainfall (mm)":    "{:,.0f}",
            })
            .background_gradient(subset=["BSI"], cmap="YlOrRd")
            .background_gradient(subset=["BRS"], cmap="Reds")
            .background_gradient(subset=["BPS"], cmap="Greens")
            .background_gradient(subset=["Revenue (₹ Cr/yr)"], cmap="BuGn"),
        use_container_width=True,
        height=700,
    )

    st.markdown("---")

    # Zone distribution summary
    col_z1, col_z2 = st.columns(2)
    with col_z1:
        st.subheader("Decision Zone Distribution")
        zone_counts = master["decision_zone"].value_counts().reset_index()
        zone_counts.columns = ["Zone", "Districts"]
        fig_pie = px.pie(
            zone_counts,
            names="Zone", values="Districts",
            color="Zone",
            color_discrete_map={
                "Plant Zone":   "#c0392b",
                "Policy Zone":  "#e67e22",
                "Low Priority": "#95a5a6",
            },
            title="43 districts classified (BSI × Residue thresholds)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_z2:
        st.subheader("Environmental Zone Distribution")
        env_counts = master["zone"].value_counts().reset_index()
        env_counts.columns = ["Zone", "Districts"]
        fig_pie2 = px.pie(
            env_counts,
            names="Zone", values="Districts",
            color="Zone",
            color_discrete_map={
                "High Stress / High Opportunity": "#c0392b",
                "Moderate Zone":                  "#f39c12",
                "Low Priority Zone":              "#27ae60",
            },
            title="K-Means k=3 environmental clustering",
        )
        st.plotly_chart(fig_pie2, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#999; font-size:12px'>"
    "Crop Residue Burning & Bioenergy Dashboard | "
    "Data: NASA FIRMS VIIRS 2015–2023, Punjab & Haryana Statistical Abstracts, NASA POWER | "
    "Module C (revised): Environmental K-Means + PCA-BSI"
    "</div>",
    unsafe_allow_html=True,
)
