from dotenv import load_dotenv
load_dotenv()
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import base64
import streamlit.components.v1 as components

# Auth imports
from backend.api.auth import login_user, signup_user
from backend.api.user_roles import get_user_role

# App pipeline modules
from backend.api.upload import process_upload
from backend.api.ontology import create_and_export_rdf
from backend.api.nlp import run_nlp_pipeline
from backend.api.graph import run_graph_pipeline
from backend.api.gis import create_gis_map
from backend.api.predict import run_prediction_pipeline
from backend.api.query import apply_filters
from backend.api.metrics import run_metrics_pipeline
from backend.models.gnn_model import build_gnn_data, train_gnn

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
    st.subheader("📝 Create New Account")

    col1, col2 = st.columns(2)
    first_name = col1.text_input("First Name")
    last_name = col2.text_input("Last Name")

    email = st.text_input("Email Address")
    password = st.text_input("Password", type='password')
    confirm_password = st.text_input("Confirm Password", type='password')

    col3, col4 = st.columns(2)
    org_type = col3.selectbox("Organization Type", ["NGO", "Company", "University", "Government", "Other"])
    country = col4.selectbox("Country", ["Kenya", "Morocco", "Tunisia", "Libya", "Egypt", "Other"])

    org_name = st.text_input("Organization Name")

    st.markdown("---")
    st.markdown("🔐 Or sign up using:")
    col_g, col_m = st.columns(2)
    with col_g:
        st.button("🔵 Google Login (Coming Soon)")
    with col_m:
        st.button("🟣 Microsoft Login (Coming Soon)")

    st.markdown("---")

    if "signed_up_email" not in st.session_state:
        if st.button("Sign Up"):
            if password != confirm_password:
                st.warning("Passwords do not match!")
            elif not email or not first_name or not org_name:
                st.warning("Please fill in all fields.")
            else:
                metadata = {
                    "first_name": first_name,
                    "last_name": last_name,
                    "organization_type": org_type,
                    "organization": org_name,
                    "country": country,
                }
                from backend.api.auth import signup_user
                result = signup_user(email, password, metadata)
                if "error" in result:
                    st.error(f"Signup failed: {result['error']}")
                else:
                    st.success("Sign-up successful. A verification OTP has been sent to your email.")
                    st.session_state["signed_up_email"] = email
    else:
        from backend.api.auth import verify_email_otp
        st.info(f"Enter the OTP sent to `{st.session_state['signed_up_email']}` to complete verification.")
        otp = st.text_input("Enter OTP")
        if st.button("Verify OTP"):
            result = verify_email_otp(st.session_state["signed_up_email"], otp)
            if "success" in result:
                st.success("Email verified successfully. Please log in.")
                del st.session_state["signed_up_email"]
            else:
                st.error("Invalid or expired OTP.")


# Role-Based Dashboard
if "role" in st.session_state:
    st.markdown(f"**Logged in as:** `{st.session_state['email']}` | **Role:** `{st.session_state['role']}`")
    st.divider()

    role = st.session_state["role"]

    # Unified Upload → Auto NLP Pipeline
    if role in ["Admin", "Data Owner"]:
        st.subheader("📤 Upload Cleaned Victim Interview Dataset")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"], key="upload")

        if uploaded_file:
            df, status = process_upload(uploaded_file)
            if df is not None:
                st.success("Upload successful. Running NLP pipeline...")
                st.session_state["uploaded_df"] = df
                structured = run_nlp_pipeline(df)
                st.session_state["structured_data"] = structured
                st.success("NLP extraction complete.")
                st.dataframe(pd.DataFrame(structured).head())
            else:
                st.error(status)

    # Optional NLP panel for Admins
    if role == "Admin":
        st.subheader("🧪 [Optional] NLP Entity Extraction Tester")
        uploaded_test_file = st.file_uploader("Upload Dataset for NLP", type=["csv", "xlsx"], key="nlp_test")

        if uploaded_test_file:
            if uploaded_test_file.name.endswith(".csv"):
                df_test = pd.read_csv(uploaded_test_file)
            else:
                df_test = pd.read_excel(uploaded_test_file)

            st.write("Running NLP pipeline...")
            result = run_nlp_pipeline(df_test)
            st.success("Test complete.")
            st.dataframe(pd.DataFrame(result))

    if role in ["Admin", "Researcher"]:
        st.subheader("🧬 Ontology Generation & RDF Export")
        if "uploaded_df" in st.session_state:
            if st.button("Generate & Export Ontology RDF File"):
                df = st.session_state["uploaded_df"]
                structured = run_nlp_pipeline(df)
                path = create_and_export_rdf(structured)
                st.success("Ontology generated successfully!")
                with open(path, "rb") as file:
                    st.download_button("📥 Download RDF File", file, file_name="ontology_export.owl")
        else:
            st.info("Please upload and process a dataset first (Phase 2/3).")

        st.subheader("🌐 Social Network Graph Visualization")

        dataset_source = st.radio("Select Dataset Source", ["Uploaded Dataset", "Merged Dataset"])

        df = None
        if dataset_source == "Uploaded Dataset" and "uploaded_df" in st.session_state:
            df = st.session_state["uploaded_df"]
        elif dataset_source == "Merged Dataset" and "merged_df" in st.session_state:
            df = st.session_state["merged_df"]

        if df is not None:
            show_graph = st.button("Generate & View Network Graph")
            if show_graph:
                structured = run_nlp_pipeline(df)
                graph_path = run_graph_pipeline(structured)
                st.success("Graph generated successfully!")
                components.html(open(graph_path, "r", encoding="utf-8").read(), height=600)
        else:
            st.info("Please upload or merge a dataset first.")


        st.subheader("🗺️ GIS Map of Trafficking Routes")
        if df is not None:
            if st.button("Generate GIS Map"):
                structured = run_nlp_pipeline(df)
                map_path = create_gis_map(structured)
                st.success("Map generated successfully!")
                components.html(open(map_path, "r", encoding="utf-8").read(), height=600)
        else:
            st.info("Please upload or merge a dataset first.")


        st.subheader("🔮 Predict Next Trafficking Location")
        if df is not None:
            if st.button("Run Predictive Model"):
                structured = run_nlp_pipeline(df)
                predictions = run_prediction_pipeline(structured)
                if isinstance(predictions, str):
                    st.warning(predictions)
                else:
                    st.success("Prediction complete!")
                    st.dataframe(predictions)
        else:
            st.info("Please upload or merge a dataset first.")

        st.subheader("🔍 Query Builder & Filter Interface")
        if df is not None:
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
            st.info("Please upload or merge a dataset first.")


        st.subheader("📊 Dashboard: Summary Metrics & Visualizations")
        if df is not None:
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
            st.info("Please upload or merge a dataset first.")


        st.subheader("🧠 GNN Node Classification (Victim / Location / Perpetrator)")
        if df is not None:
            if st.button("Run GNN Classification"):
                structured = run_nlp_pipeline(df)
                data, encoder = build_gnn_data(structured)
                model, output = train_gnn(data, num_classes=len(encoder.classes_))
                st.success("GNN trained and nodes classified!")
                predictions = output.argmax(dim=1).numpy()
                labels = [encoder.inverse_transform([pred])[0] for pred in predictions]
                node_map = {f"Node {i}": label for i, label in enumerate(labels)}
                st.json(node_map)
        else:
            st.info("Please upload or merge a dataset first.")

# 📁 Downloadable Upload Format Template
st.markdown("#### 📄 Download Upload Format Template")

sample = pd.DataFrame(columns=[
    "Unique ID",
    "Interviewer Name",
    "Date of Interview",
    "Gender of Victim",
    "Nationality of Victim",
    "Left Home Country Year",
    "Borders Crossed",
    "City / Locations Crossed",
    "Final Location",
    "Name of the Perpetrators involved",
    "Hierarchy of Perpetrators",
    "Human traffickers/ Chief of places",
    "Time Spent in Location / Cities / Places"
])

st.download_button(
    "📥 Download Template (CSV)",
    sample.to_csv(index=False),
    file_name="trafficking_upload_template.csv"
)

st.markdown("---")

# 📜 Consent & FAIR Compliance
with st.expander("📜 Data Consent & FAIR Compliance Notice"):
    st.markdown("""
    - By uploading, you confirm the data adheres to FAIR and ethical principles.
    - Personally identifiable information must be anonymized.
    - Qualitative abuse details should be excluded.
    - System access is controlled by user roles.
    - Uploaded data may be cleaned, validated, and merged.

    *Your contribution enables responsible and collaborative anti-trafficking research.*
    """)
