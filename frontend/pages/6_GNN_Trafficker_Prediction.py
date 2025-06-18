import streamlit as st
import pandas as pd
from backend.models.gnn_trafficker_predict import prepare_gnn_graph, run_gnn_prediction

st.set_page_config(page_title="🧠 GNN: Trafficker Prediction", layout="wide")
st.title("🧠 GNN Prediction: Likely Traffickers")

if st.session_state.get("role") not in ["Admin", "Researcher"]:
    st.warning("Only Admin and Researcher roles can use this feature.")
    st.stop()

if "uploaded_df" not in st.session_state:
    st.info("Please upload a dataset first.")
    st.stop()

st.markdown("""
This model classifies graph nodes to predict which ones are likely human traffickers based on victim connections.
""")

data = prepare_gnn_graph(st.session_state["uploaded_df"])
predictions = run_gnn_prediction(data)

# Filter by predicted traffickers
predicted_traffickers = [node for node, label in predictions.items() if label == 1]

if predicted_traffickers:
    st.success("GNN classified the following nodes as likely traffickers:")
    st.write(predicted_traffickers)
else:
    st.warning("No trafficker predictions were identified by the GNN.")
