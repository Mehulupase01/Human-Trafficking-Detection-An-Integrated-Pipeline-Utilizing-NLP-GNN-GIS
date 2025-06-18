import streamlit as st
import streamlit.components.v1 as components
from backend.api.graph_hierarchy import build_trafficker_hierarchy

st.set_page_config(page_title="📊 Trafficker Hierarchy Viewer", layout="centered")
st.title("📊 Trafficker Hierarchy Graph")

if st.session_state.get("role") not in ["Admin", "Researcher"]:
    st.warning("Only Admin and Researcher roles can view the trafficker hierarchy.")
    st.stop()

if "uploaded_df" not in st.session_state:
    st.info("Please upload a dataset first.")
    st.stop()

df = st.session_state["uploaded_df"]

if st.button("Generate Hierarchy Graph"):
    path = build_trafficker_hierarchy(df)
    st.image(path, caption="Trafficker Hierarchy", use_column_width=True)
    st.success("Hierarchy graph generated successfully!")
