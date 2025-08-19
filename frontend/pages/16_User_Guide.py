# frontend/pages/13_User_Guide.py
from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="User Guide", page_icon="📘", layout="wide")
st.title("📘 How to use this platform")

st.markdown("""
Welcome! This guide explains what each page does and what the main **settings** mean.

---

## 🔼 Upload & Standardize
- **What it does:** reads your spreadsheet(s), runs the NLP standardizer (locations, perpetrators, chiefs, etc.), and produces a **Processed** dataset.
- **Serialized ID (HTV…)**: a stable ID per unique victim. We count unique victims by this ID (not by rows).
- **Locations (NLP)**: single‑word tokens (e.g., “Sudan”, “Tripoli”). We always take the **first token** per step.

**Tips**
- Keep only one victim per “Unique ID”.
- Long sentences in time/nationality/location columns are auto‑cleaned into standardized fields.

---

## 🔗 Merge Datasets
- **What it does:** combines multiple **Processed** datasets into a **Merged** one with deduplication by `Serialized ID` + data consistency checks.
- **History/Delete:** you can keep multiple versions; the latest is offered first on other pages.

---

## 🔍 Query Builder & Insights
- Works only on **Processed/Merged** datasets.
- **Gender = Any:** shows all.
- **Locations (searchable):** filters by standardized locations.
- **Perpetrators/Chiefs filters:** comma‑separated list; match‑any.
- **Results table — list columns:** choose what you want to see; downloads in CSV/JSON.

**Charts**
- **Gender pie**: fraction of victims (not rows).
- **Nationality bar**: top nationalities by unique victims.
- **Route lengths (binned)**: histogram of number of **steps** per victim.  
  *“Stops (binned)”* means we group victims by how many steps they have (e.g., 6–10, 11–15).

---

## 🕸️ Network Graphs
- **Network** tab: victim/location/perpetrator nodes with visits/links.
  - Hover to see **node type**, **name**, and quick counts (e.g., “Location — Visits: 41”).
  - Zoom/pan; use fullscreen; export HTML/PNG.
- **Victim Route Hierarchy** tab: per‑victim step‑by‑step flow.
  - Each node is a **step** containing the location name; edges show direction.
  - We’ll also show perpetrators/chiefs at a step when present.

---

## 📈 Trend Graph Explorer
- **Trend by**: *Nationality* (from standardized field) **or** *Location* (first token).
- **Year**: we try “Date of Interview → year”; if missing, use “Left Home Country Year”.
- **Smoothing**: 3/5‑year rolling mean to reduce spikiness (0 = off).
- **Cumulative**: running totals over years.
- **Y‑axis scale**: linear or log.

**Counts are by unique victims (Serialized ID), not by rows.**

---

## 🔮 Predictive Analytics
Two models:

1) **Next Locations (n‑gram)**  
   - Learns “A → B” transitions from past routes; predicts next N locations for a victim.  
   - **Score**: relative confidence from the model (higher = more likely within the learned transitions).  
   - You can **save** a run to overlay the predicted path on the **GIS** page.

2) **Perpetrators (baseline)**  
   - Learns co‑occurrence patterns (victims ↔ perpetrators) and suggests top‑K perpetrators.  
   - If a victim has very few steps or rare names, you may see *“No predictions produced”* — try another victim or add more data.

---

## ⏱️ Temporal Forecast (ETA)
- Combines the **next‑locations model** with a learned **Time‑to‑travel** estimator.
- ETA learning uses medians of:
  1. **Transition medians**: typical days for A→B.
  2. **Location medians**: typical days to reach a location (if A→B is unknown).
  3. **Global median**: typical days across the whole dataset.
- **Fallback days (if unknown):** final default if nothing was learned for a step.
- **Start date:** if provided, we compute **arrival dates** by cumulatively adding step ETAs.

**Interpreting the table**
- **Step**: 1…N predicted hops.
- **Predicted Location**: standardized location token.
- **ETA (days / weeks)**: typical time to reach that location.
- **Cumulative days** & **Arrival date** (if you set a start date).

---

## 🗺️ Map / GIS Visualizer
- Dark theme; markers sized by visits/degree.
- Hover shows **incoming/outgoing counts**; legend explains color by type.
- Overlay **predicted paths** saved from *Predictive Analytics*.
- Fullscreen + PNG export.

---

## ⚙️ Settings & Admin Pages
- **Upload History**: lists RAW/Processed/Merged datasets in the registry.
- **Admin File Manager**: pick a dataset, preview first rows, and **download full CSV**.
- **Query History**: saved queries and model runs (next‑locations, perpetrators, ETA). You can preview & download each record.

---

## Common Questions

**Q: Why do I only see “Unknown” genders or many NaNs initially?**  
A: It means the original fields were free text (or missing). The **NLP standardizer** cleans them; make sure you’re working with the **Processed/Merged** datasets on analytics pages.

**Q: “Stops (binned)” looks odd.**  
A: That chart groups *victims* by how many steps they had; if many victims have short routes, most counts will sit in small bins.

**Q: What’s a good “smoothing” value?**  
A: Try **3‑year** smoothing to reduce noise; use **5‑year** when you care more about trend than year‑to‑year variability.

**Q: When predictions say “insufficient history”?**  
A: The model needs enough past steps for that victim (and overlapping patterns with others). Try a different victim or add more processed data.

---
""")
