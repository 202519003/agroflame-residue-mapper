# Project Proposal — MSc (AA) Project Work 2026

**Group Name:** EmberGrid  
**Members:** Maharshi Patel, Dhruv Soni, Mehul Chaudhary, Yash Daslaniya, Tushar Vadodariya, Gopal Patidar  
**Project Working Title:** AgroFlame — Crop Residue Burning Risk & Bioenergy Potential Mapper (Punjab & Haryana, District-Level)

---

## 1. Problem Statement

### What is the "Spatial Problem"?

Punjab and Haryana together account for over **25 million fire events per year** during the October–November kharif harvest window and the April–May rabi harvest season — the highest agricultural burning concentration in the world. Farmers burn paddy and wheat straw because they lack affordable alternatives, yet this destroys an estimated 35 MT/year of biomass that could power compressed biogas (CBG) or 2G ethanol plants, while also driving the annual Delhi-NCR air quality crisis and drawing Supreme Court and NGT interventions.

The core spatial gap: **no publicly accessible, district-level tool currently integrates burning risk, bioenergy potential, and farmer participation viability into a single prioritisation score.** Policy agencies, energy investors, and agri-tech companies have no data-driven way to identify which of the 44 districts to target first.

AgroFlame addresses this by combining 9 years of VIIRS satellite fire data, verified state crop production statistics, and four socioeconomic proxies into a **Composite Action Priority Score (CAPS)** for every district in Punjab and Haryana.

### Target User

- Energy investors and CBG plant developers evaluating districts with the highest recoverable biomass
- State government officers (Punjab and Haryana Renewable Energy Directorates) identifying where farmer outreach is most urgent
- Researchers and NGOs tracking whether burning trends are improving or worsening at district level

---

## 2. Technical Stack & Libraries

### GUI Framework

**Streamlit** — interactive dashboard without frontend code. Users select a district and instantly see its full burning profile, bioenergy estimates, and priority ranking.

### Core Geospatial Logic

- **GeoPandas** — loading GADM Level-2 district shapefile, spatial join of VIIRS fire points to district polygons
- **Folium** — interactive choropleth maps embedded in the Streamlit dashboard
- **Shapely** — bounding-box clipping and centroid extraction per district

### The "Advanced" Components

**ML Track — Scikit-learn:**

- `KMeans` (k=3) for farmer typology clustering using 4 socioeconomic features
- `StandardScaler` + `PCA` for feature normalisation and importance weighting in the Participation Viability Index (PVI)
- `XGBoost` regressor for bioenergy potential estimation and feature importance ranking

**Web & API Track — Requests:**

- **NASA FIRMS** — VIIRS 375m active fire CSV for South Asia (2015–2023), zero login
- **NASA POWER API** — daily weather data per district centroid via `requests.get()` to `power.larc.nasa.gov/api/temporal/daily/point`
- **pymannkendall** — Mann-Kendall trend test on 9-year per-district fire count series

### Data Sources

| # | Dataset | Source | Format | Module |
|---|---------|--------|--------|--------|
| 1 | VIIRS 375m Active Fire 2015–2023 | firms.modaps.eosdis.nasa.gov | CSV | A |
| 2 | Punjab Statistical Abstract (Crop Production) | esopb.gov.in | Excel | B |
| 3 | Haryana Statistical Abstract (Crop Production) | esaharyana.gov.in | Excel | B |
| 4 | ICAR/MNRE Biomass Atlas (RPR + LHV Coefficients) | mnre.gov.in/biomass-power | PDF → lookup | B |
| 5 | NASA POWER Daily Weather API | power.larc.nasa.gov | JSON via API | A+B |
| 6 | NSSO 77th Round Agricultural Households 2019 | mospi.gov.in | PDF + tables | C |
| 7 | Agricultural Census 2015–16 | agcensus.dacfw.nic.in | PDF tables | C |
| 8 | FPO Registered District Data | data.gov.in | CSV/Excel | C |
| 9 | PMFBY District Insurance Enrolment Reports | pmfby.gov.in | Excel | C |
| 10 | GADM India District Shapefile (Level 2) | gadm.org | GeoJSON/SHP | All |
| 11 | FAO FAOSTAT India Crop Production (validation) | faostat.fao.org | CSV | Validation |

---

## 3. Proposed GUI Architecture

### Input Section

- **District Dropdown** — `st.selectbox` for any of the 44 Punjab/Haryana districts
- **Year Range Slider** — `st.slider` to filter the 9-year fire time series (2015–2023)
- **Weight Customiser** — `st.number_input` fields to adjust BRS/BPS/PVI weights in CAPS formula (defaults: 35%, 40%, 25%)

### Processing Section

When the user hits **"Generate Profile"**:

1. Filters the fire dataframe to selected district and year range
2. Pulls precomputed BRS, BPS, PVI, and CAPS scores from master district dataframe
3. Renders selected district polygon highlighted on Folium map
4. Computes bioenergy revenue potential in ₹ crore/year dynamically

### Output / Visualisation

- **Folium Choropleth Map** — full 44-district view coloured by CAPS score; selected district highlighted with popup
- **Fire Trend Line Chart** — 9-year fire count time series with Mann-Kendall trend direction
- **Bioenergy Bar Chart** — rice straw vs wheat straw recoverable biomass and ₹ revenue potential
- **Score Card Panel** — BRS, BPS, PVI, CAPS as `st.metric` cards with delta vs state average
- **Policy Recommendation Box** — classifies district as: Priority Investment Zone / Policy Intervention Zone / Monitoring Zone

---

## 4. GitHub Repository Setup

**Repo URL:** `https://github.com/dhruvhokgames-web/agroflame-residue-mapper`

**Initial Folder Structure (confirmed):**

```
agroflame-residue-mapper/
├── data/
│   ├── fire/               # VIIRS CSV files (2015–2023, Punjab+Haryana)
│   ├── crop/               # Punjab + Haryana Statistical Abstract Excel files
│   ├── socioeconomic/      # NSSO, Agri Census, FPO, PMFBY cleaned CSVs
│   └── spatial/            # GADM Level-2 GeoJSON + OSM Shapefiles
├── docs/
│   ├── writeup.pdf         # 3–5 page project report
│   └── architecture.png    # System architecture diagram
├── src/
│   ├── gui/
│   │   └── dashboard.py    # Streamlit app layout and event handling
│   ├── logic/
│   │   ├── burning_risk.py  # Module A: VIIRS processing, BRS computation
│   │   ├── bioenergy.py     # Module B: Residue estimation, GEP/NBP/revenue
│   │   ├── participation.py # Module C: K-Means clustering, PVI scoring
│   │   └── caps.py          # Module D: Composite score integration
│   └── utils/
│       ├── api_fetcher.py   # NASA POWER API calls per district centroid
│       ├── spatial_utils.py # GeoPandas helpers, spatial join functions
│       └── preprocessor.py  # Data cleaning, normalisation, merging
├── main.py                  # Entry point: launches Streamlit dashboard
├── requirements.txt         # All pip dependencies
└── PROPOSAL.md              # This file
```

---

## 5. Preliminary Task Distribution

| Member | Primary Responsibility | Secondary Responsibility |
|--------|----------------------|--------------------------|
| Member 1 | **Module A — Burning Risk** (VIIRS pipeline, spatial join, Mann-Kendall trend, BRS) | Data download & folder setup |
| Member 2 | **Module B — Bioenergy Potential** (Statistical Abstract cleaning, RPR/LHV coefficients, NBP/revenue) | Validation against PAU/HAU estimates |
| Member 3 | **Module C — Participation Viability** (NSSO + Agri Census + FPO + PMFBY, K-Means, PVI) | Documentation & README.md |
| Member 4 | **Module D — CAPS Integration** (BRS + BPS + PVI merge, district ranking, zone classification) | Presentation visuals |
| Member 5 | **GUI — Streamlit Dashboard** (dropdown, sliders, Folium map, score cards, policy box) | UI/UX testing |
| Member 6 | **API & System Lead** (NASA POWER fetcher, Git management, requirements.txt, integration testing) | Code review & final submission |

---


