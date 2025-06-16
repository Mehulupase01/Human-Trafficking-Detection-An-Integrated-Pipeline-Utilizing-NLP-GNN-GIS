import streamlit as st
import pandas as pd
import os

# Auth imports
from backend.api.auth import login_user, signup_user
from backend.api.user_roles import get_user_role

# Upload & NLP imports
from backend.api.upload import process_upload
from backend.api.ontology import create_and_export_rdf
from backend.api.nlp import run_nlp_pipeline

# Config
st.set_page_config(page_title="Trafficking Analytics App", layout="centered")
st.title("🚨 Human Trafficking Analytics Platform")

# Menu selection
menu = ["Login", "SignUp"]
choice = st.sidebar.selectbox("Menu", menu)

# Authentication section
if choice == "Login":
    email = st.text_input("Email")
    password = st.text_input("Password", type='password')

    if st.button("Login"):
        result = login_user(email, password)
        if "error" in result:
            st.error(f"Login failed: {result['error']}")
        else:
            st.success("Logged in successfully!")
            role = get_user_role(email)
            st.session_state["email"] = email
            st.session_state["role"] = role

elif choice == "SignUp":
    email = st.text_input("Email")
    password = st.text_input("Password", type='password')
    confirm_password = st.text_input("Confirm Password", type='password')

    if st.button("Sign Up"):
        if password != confirm_password:
            st.warning("Passwords do not match!")
        else:
            result = signup_user(email, password)
            if "error" in result:
                st.error(f"Signup failed: {result['error']}")
            else:
                st.success("Signed up successfully! Please check your email to verify.")

# Role-Based Dashboard
if "role" in st.session_state:

    st.markdown(f"**Logged in as:** `{st.session_state['email']}` | **Role:** `{st.session_state['role']}`")
    st.divider()

    role = st.session_state["role"]

    # Data Upload Interface (Phase 2)
    if role in ["Admin", "Data Owner"]:
        st.subheader("📤 Upload Cleaned Victim Interview Dataset")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"], key="upload")

        if uploaded_file:
            df, status = process_upload(uploaded_file)
            if df is not None:
                st.success(status)
                st.dataframe(df.head())
                st.session_state["uploaded_df"] = df
            else:
                st.error(status)

    # NLP Entity Extraction (Phase 3)
    if role in ["Admin", "Researcher"]:
        st.subheader("🧠 NLP Entity Extraction & Structuring")
        uploaded_file_nlp = st.file_uploader("Upload Dataset for NLP", type=["csv", "xlsx"], key="nlp")

        if uploaded_file_nlp:
            if uploaded_file_nlp.name.endswith(".csv"):
                df_nlp = pd.read_csv(uploaded_file_nlp)
            else:
                df_nlp = pd.read_excel(uploaded_file_nlp)

            st.write("Running NLP pipeline...")
            results = run_nlp_pipeline(df_nlp)
            st.success("Extraction complete!")
            st.dataframe(pd.DataFrame(results))
            
            
# RDF Export Section (Phase 4)
if "role" in st.session_state and st.session_state["role"] in ["Admin", "Researcher"]:
    st.subheader("🧬 Ontology Generation & RDF Export")

    if "uploaded_df" in st.session_state:
        run_export = st.button("Generate & Export Ontology RDF File")

        if run_export:
            df = st.session_state["uploaded_df"]
            structured = run_nlp_pipeline(df)
            path = create_and_export_rdf(structured)
            st.success("Ontology generated successfully!")
            with open(path, "rb") as file:
                st.download_button("📥 Download RDF File", file, file_name="ontology_export.owl")
    else:
        st.info("Please upload and process a dataset first (Phase 2/3).")



from backend.api.graph import run_graph_pipeline
import streamlit.components.v1 as components

if "role" in st.session_state and st.session_state["role"] in ["Admin", "Researcher"]:
    st.subheader("🌐 Social Network Graph Visualization")

    if "uploaded_df" in st.session_state:
        show_graph = st.button("Generate & View Network Graph")

        if show_graph:
            df = st.session_state["uploaded_df"]
            structured = run_nlp_pipeline(df)
            graph_path = run_graph_pipeline(structured)
            st.success("Graph generated successfully!")
            components.html(open(graph_path, "r", encoding="utf-8").read(), height=600)
    else:
        st.info("Please upload and process a dataset first (Phase 2/3).")
        

from backend.api.gis import create_gis_map

if "role" in st.session_state and st.session_state["role"] in ["Admin", "Researcher"]:
    st.subheader("🗺️ GIS Map of Trafficking Routes")

    if "uploaded_df" in st.session_state:
        if st.button("Generate GIS Map"):
            df = st.session_state["uploaded_df"]
            structured = run_nlp_pipeline(df)
            map_path = create_gis_map(structured)
            st.success("Map generated successfully!")
            components.html(open(map_path, "r", encoding="utf-8").read(), height=600)
    else:
        st.info("Please upload and process a dataset first (Phase 2/3).")
        
from backend.api.predict import run_prediction_pipeline

if "role" in st.session_state and st.session_state["role"] in ["Admin", "Researcher"]:
    st.subheader("🔮 Predict Next Trafficking Location")

    if "uploaded_df" in st.session_state:
        if st.button("Run Predictive Model"):
            df = st.session_state["uploaded_df"]
            structured = run_nlp_pipeline(df)
            predictions = run_prediction_pipeline(structured)

            if isinstance(predictions, str):
                st.warning(predictions)
            else:
                st.success("Prediction complete!")
                st.dataframe(predictions)
    else:
        st.info("Please upload and process a dataset first (Phase 2/3).")


from backend.api.query import apply_filters

if "role" in st.session_state and st.session_state["role"] in ["Admin", "Researcher"]:
    st.subheader("🔍 Query Builder & Filter Interface")

    if "uploaded_df" in st.session_state:
        df = st.session_state["uploaded_df"]
        nationality = st.selectbox("Filter by Nationality", options=[""] + sorted(df["Nationality of Victim"].dropna().unique().tolist()))
        gender = st.selectbox("Filter by Gender", options=["", "Male", "Female"])
        year_range = st.slider("Left Home Country (Year Range)", min_value=int(df["Left Home Country Year"].min()), max_value=int(df["Left Home Country Year"].max()), value=(2010, 2020))
        location = st.text_input("Contains Location (partial match)")

        run_query = st.button("Apply Filters")
        if run_query:
            filtered_df = apply_filters(df, nationality or None, gender or None, year_range, location or None)
            st.success(f"{len(filtered_df)} record(s) matched.")
            st.dataframe(filtered_df)
    else:
        st.info("Please upload and process a dataset first (Phase 2/3).")


from backend.api.metrics import run_metrics_pipeline
import streamlit.components.v1 as components
import base64

if "role" in st.session_state and st.session_state["role"] in ["Admin", "Researcher"]:
    st.subheader("📊 Dashboard: Summary Metrics & Visualizations")

    if "uploaded_df" in st.session_state:
        df = st.session_state["uploaded_df"]
        total, nationality_dist, gender_dist, chart = run_metrics_pipeline(df)

        st.metric("Total Unique Victims", total)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Nationality Distribution**")
            st.json(nationality_dist)
        with col2:
            st.markdown("**Gender Distribution**")
            st.json(gender_dist)

        st.markdown("**Histogram: Year Left Home Country**")
        st.image(chart, use_column_width=True)
    else:
        st.info("Please upload and process a dataset first (Phase 2/3).")