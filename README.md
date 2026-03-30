# 🌾 Crop Residue Burning & Bioenergy Dashboard

**Punjab & Haryana | 47 Districts | 2015–2023 VIIRS Fire Data**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 🚀 Live Dashboard

> **[Launch Dashboard →](https://your-app-url.streamlit.app)**  
> *(Replace this link after deploying on Streamlit Cloud)*

---

## 📌 What This Project Does

A complete spatial-ML pipeline to identify **priority districts for Compressed Biogas (CBG) plant investment** by combining satellite fire data, crop production statistics, and environmental variables.

### Three analytical modules:

| Module | Notebooks | Output |
|--------|-----------|--------|
| **A — Fire Risk** | NB01–07 | Burning Risk Score (BRS 0–100) via fire frequency + FRP + Mann-Kendall trend |
| **B — Bioenergy** | NB08–10 | Bioenergy Potential Score (BPS 0–100) + ₹ revenue/yr via RPR + CBG yield |
| **C — Environmental** *(revised)* | NB11–16 | K-Means zones + PCA-derived Burning Severity Index (BSI 0–100) |
| **D — Validation** | NB17 | LOOCV cluster stability: 100% stability, silhouette = 0.41 |

**Module C was revised** from the original socioeconomic approach (NSSO/FPO/PMFBY) to an **environmental feature approach** using fire_count, recoverable residue, avg_temp, and rainfall — making the pipeline entirely satellite + statistical-data-driven.

### Final decision logic (NB16):

```
BSI ≥ 70th percentile  AND  Residue ≥ 70th percentile  →  🔴 Plant Zone
BSI ≥ 70th percentile  only                            →  🟠 Policy Zone
Below thresholds                                        →  ⚫ Low Priority
```

---

## 🔑 Key Findings

- **9 Plant Zone districts** identified: Sangrur, Bathinda, Muktsar, Fazilka, Sirsa, Ferozepur, Fatehabad, Moga, and Mansa
- **Top BSI district**: Sangrur (BSI = 100.0) with 19,313 peak annual fire detections
- **Top revenue potential**: Sangrur ≈ ₹85 Cr/yr, Ludhiana ≈ ₹73 Cr/yr
- **PCA variance explained**: PC1 = 53.6%, PC1+PC2 = 87.0% — 4 environmental features well-captured
- **Mann-Kendall trends**: 2 of 43 districts show statistically significant trends (p < 0.05)
- **LOOCV cluster stability (NB17)**: 47/47 districts (100%) stably assigned — cluster structure is not driven by any individual district

---

## 🗂️ Repository Structure

```
crop-residue-bioenergy-dashboard/
│
├── app/
│   └── dashboard.py              ← Streamlit app (main file)
│
├── data/
│   └── processed/                ← All cleaned CSV files (no raw data)
│       ├── burning_risk_scores.csv   ← NB06 output: BRS + risk_class per district
│       ├── bioenergy_scores.csv      ← NB10 output: BPS + CBG revenue per district
│       ├── bsi_scores.csv            ← NB14 output: BSI 0–100 per district
│       ├── district_clusters.csv     ← NB13 output: K-Means zone per district
│       ├── env_features.csv          ← NB11 output: 4 env features per district
│       ├── fire_stats.csv            ← NB04 output: annual fire stats per district
│       └── fire_trends.csv           ← NB05 output: Mann-Kendall results
│
├── outputs/
│   ├── maps/                     ← Pre-generated map PNGs
│   │   ├── 15_bsi_map.png            ← BSI choropleth (NB15)
│   │   ├── 15_cluster_map.png        ← K-Means zone map (NB15)
│   │   ├── 15_top5_burning.png       ← Top-5 BSI districts highlighted (NB15)
│   │   ├── 16_final_decision_map.png ← Plant/Policy/Low Priority zones (NB16)
│   │   └── 17_cluster_stability_map.png ← LOOCV stability map (NB17) ← NEW
│   ├── plots/
│   │   └── 17_stability_barplot.png  ← Per-district silhouette barplot (NB17) ← NEW
│   └── tables/
│       └── 17_loocv_stability.csv    ← LOOCV per-district results (NB17) ← NEW
│
├── requirements.txt              ← Python dependencies for Streamlit Cloud
├── .gitignore                    ← Excludes raw data, notebooks, pickles
└── README.md                     ← This file
```

---

## ⚡ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/crop-residue-bioenergy-dashboard.git
cd crop-residue-bioenergy-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app/dashboard.py
```

Dashboard opens at **http://localhost:8501**

---

## ☁️ Streamlit Cloud Deployment

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **New app**
3. Select your repo → Branch: `main` → Main file: `app/dashboard.py`
4. Click **Deploy**

> **Note:** `geopandas` is intentionally excluded from `requirements.txt`. Maps are pre-generated PNGs so no spatial dependencies are needed at runtime on Streamlit Cloud.

### Files to commit for NB17 tab to work

After running NB17 locally, commit these three files to your repo:

```
outputs/tables/17_loocv_stability.csv
outputs/maps/17_cluster_stability_map.png
outputs/plots/17_stability_barplot.png
```

The dashboard **gracefully degrades** — if these files are missing, a warning is shown instead of crashing.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **NASA FIRMS VIIRS** | 9 years of active fire detection data (375m resolution) |
| **GeoPandas** | Spatial join of fire points to GADM district polygons |
| **pymannkendall** | Non-parametric trend test on annual fire time series |
| **scikit-learn** | K-Means clustering + PCA for BSI derivation + MinMaxScaler |
| **SciPy** | Hungarian algorithm (linear_sum_assignment) for LOOCV label alignment |
| **Plotly** | Interactive charts in Streamlit dashboard |
| **Streamlit** | Web dashboard deployment |

---

## 📓 Notebook Pipeline (in order)

```
NB01 → Setup spatial (filter GADM to 43 Punjab+Haryana districts)
NB02 → Fire processing (load + filter 8.7M VIIRS records)
NB03 → Spatial join (fire points → district polygons, 1.17M matched)
NB04 → Fire aggregation (district × year: count, FRP, onset, peak week)
NB05 → Trend analysis (Mann-Kendall per district, 2015–2023)
NB06 → BRS score (fire freq 40% + FRP 30% + slope 30%)
NB07 → BRS map (choropleth: continuous + classified)
NB08 → Crop cleaning (Punjab multi-year + Haryana 2022-23, 611 rows)
NB09 → Residue calculation (RPR × burn fraction × 70% recovery)
NB10 → Bioenergy calculation (LHV → GEP → CBG → ₹ revenue → BPS)
NB11 → Environmental feature engineering (fire + residue + NASA POWER weather)
NB12 → StandardScaler (zero mean, unit variance for K-Means + PCA)
NB13 → K-Means k=3 (environmental intervention zones)
NB14 → PCA + BSI (PC1 loadings → data-driven weights → BSI 0–100)
NB15 → Maps visualisation (BSI map + zone map + top-5 map)
NB16 → Final decision map (BSI × residue 70th-pct thresholds)
NB17 → LOOCV cluster stability (100% stable, silhouette = 0.41)   ← NEW
```

---

## 🔬 NB17 — LOOCV Cluster Stability (New)

**Notebook 17** validates the K-Means clustering from NB13 using Leave-One-Out Cross-Validation (LOOCV):

- Each of the 47 districts is iteratively withheld; K-Means (k=3) is re-fitted on the remaining 46; the withheld district is assigned to its nearest centroid
- Cluster label permutation is resolved using the **Hungarian algorithm**
- **Result: 47/47 districts (100.0%) stably assigned** — no boundary districts detected

| Metric | Value |
|--------|-------|
| Cluster stability | 100.0% (47/47) |
| Overall silhouette | 0.4088 |
| Cluster 0 silhouette (n=10) | 0.2999 |
| Cluster 1 silhouette (n=17) | 0.6776 |
| Cluster 2 silhouette (n=20) | 0.2349 |

The dashboard's **🔬 Cluster Stability** tab shows an interactive silhouette bar chart, the NB17 stability map, and a per-district LOOCV results table.

---

## 📚 Data Sources

| Dataset | Source |
|---------|--------|
| Active fire detections | NASA FIRMS VIIRS S-NPP + NOAA-20 (2015–2023) |
| District boundaries | GADM India Level 2 |
| Crop production | Punjab Statistical Abstract, Haryana Statistical Abstract |
| Weather (temp + rainfall) | NASA POWER API |
| RPR & burn fractions | MNRE / ICAR standard values |

---

## 🎓 Academic Context

M.Sc. Semester 2 Research Project | Dehradun, India  
*Crop residue burning in Punjab and Haryana contributes significantly to seasonal air quality crises in north India. This project identifies optimal locations for compressed biogas plants as an economic incentive for farmers to divert residue from burning to bioenergy.*
