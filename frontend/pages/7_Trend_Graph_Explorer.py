import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="📈 Trend Graph Explorer", layout="wide")
st.title("📈 Country-wise Victim Trends Over Time")

if st.session_state.get("role") not in ["Admin", "Researcher", "Viewer"]:
    st.warning("Access denied.")
    st.stop()

# Accept merged and individual datasets
datasets = {}
if "merged_df" in st.session_state:
    datasets["Merged Dataset"] = st.session_state["merged_df"]

for key in st.session_state.keys():
    if key.startswith("dataset_") and isinstance(st.session_state[key], pd.DataFrame):
        datasets[key.replace("dataset_", "")] = st.session_state[key]
if "uploaded_df" in st.session_state:
    datasets["Uploaded Dataset"] = st.session_state["uploaded_df"]

if not datasets:
    st.info("Please upload or merge datasets first.")
    st.stop()

st.subheader("📂 Select Dataset(s) to Analyze")
selected_sets = st.multiselect(
    "Choose one or more datasets", 
    options=list(datasets.keys()), 
    default=list(datasets.keys())[:1]
)
if not selected_sets:
    st.warning("Please select at least one dataset.")
    st.stop()

# Merge selected
df = pd.concat([datasets[name] for name in selected_sets], ignore_index=True)

if "Left Home Country Year" not in df.columns or "Nationality of Victim" not in df.columns:
    st.error("Required fields missing in selected dataset(s).")
    st.stop()

# Clean and prepare
cleaned = df.dropna(subset=["Left Home Country Year", "Nationality of Victim"])
cleaned["Year"] = cleaned["Left Home Country Year"].astype(int)
trend = cleaned.groupby(["Nationality of Victim", "Year"]).size().unstack(fill_value=0)

selected_country = st.selectbox("Select Country to View Trends", sorted(trend.index))
data = trend.loc[selected_country]

fig, ax = plt.subplots()
data.plot(ax=ax, marker='o')
ax.set_title(f"Victim Trend: {selected_country}")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Victims")
st.pyplot(fig)
