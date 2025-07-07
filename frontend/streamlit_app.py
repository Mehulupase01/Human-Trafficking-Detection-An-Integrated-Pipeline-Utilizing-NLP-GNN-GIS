from dotenv import load_dotenv
load_dotenv()
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import base64
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Auth
from backend.api.auth import login_user, signup_user
from backend.api.user_roles import get_user_role

# Modules
from backend.api.upload import process_upload
from backend.api.ontology import create_and_export_rdf
from backend.api.nlp import run_nlp_pipeline
from backend.api.graph import run_graph_pipeline
from backend.api.gis import create_gis_map
from backend.api.predict import run_prediction_pipeline
from backend.api.query import apply_filters
from backend.api.metrics import run_metrics_pipeline
from backend.models.gnn_model import build_gnn_data, train_gnn
from backend.api.ontology import create_and_export_rdf

st.set_page_config(page_title="Trafficking Analytics App", layout="centered")
st.title("🚨 Human Trafficking Analytics Platform")

# Auth UI
menu = ["Login", "SignUp"]
choice = st.sidebar.selectbox("Menu", menu)

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
    st.button("🔵 Google Login (Coming Soon)")
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

# ======================== MAIN ===========================
if "role" in st.session_state:
    st.markdown(f"**Logged in as:** `{st.session_state['email']}` | **Role:** `{st.session_state['role']}`")
    st.divider()
    role = st.session_state["role"]

    # UPLOAD FLOW
    if role in ["Admin", "Data Owner"]:
        st.subheader("📤 Upload Cleaned Victim Interview Dataset")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"], key="upload")

        if uploaded_file:
            df, status = process_upload(uploaded_file)
            if df is not None:
                st.success("Upload successful. Running NLP pipeline...")
                st.session_state["uploaded_df"] = df
                structured = st.session_state.get("structured_data", run_nlp_pipeline(df))
                st.session_state["structured_data"] = structured
                st.success("NLP extraction complete.")
                st.dataframe(pd.DataFrame(structured).head())
            else:
                st.error(status)

    # OPTIONAL TEST PANEL
    if role == "Admin":
        st.subheader("🧪 [Optional] NLP Entity Extraction Tester")
        test_file = st.file_uploader("Upload Dataset for NLP", type=["csv", "xlsx"], key="nlp_test")
        if test_file:
            df_test = pd.read_csv(test_file) if test_file.name.endswith(".csv") else pd.read_excel(test_file)
            result = run_nlp_pipeline(df_test)
            st.success("NLP extraction completed.")
            st.dataframe(pd.DataFrame(result))

    # ALL POST-NLP MODULES (shared data source toggle)
    if role in ["Admin", "Researcher"]:
        st.subheader("🧬 Ontology Generation & RDF Export")
        if st.button("Generate & Export Ontology RDF File"):
            structured = st.session_state.get("structured_data", run_nlp_pipeline(df))
            rdf_path = create_and_export_rdf(structured)
            if rdf_path:
                with open(rdf_path, "rb") as f:
                    st.download_button("📥 Download RDF Ontology", f, file_name="ontology_export.owl")
                st.success("Ontology successfully generated and exported.")
            else:
                st.error("Failed to generate ontology.")

        st.subheader("🌐 Social Network Graph Visualization")
        dataset_source = st.radio("Select Dataset Source", ["Uploaded Dataset", "Merged Dataset"])
        df = None
        if dataset_source == "Uploaded Dataset" and "uploaded_df" in st.session_state:
            df = st.session_state["uploaded_df"]
        elif dataset_source == "Merged Dataset" and "merged_df" in st.session_state:
            df = st.session_state["merged_df"]

        if df is not None:
            if st.button("Generate & View Network Graph"):
                structured = st.session_state.get("structured_data", run_nlp_pipeline(df))
                path = run_graph_pipeline(structured)
                st.success("Graph generated.")
                components.html(open(path, "r", encoding="utf-8").read(), height=600)

            st.subheader("🗺️ GIS Map of Trafficking Routes")
            if st.button("Generate GIS Map"):
                structured = st.session_state.get("structured_data", run_nlp_pipeline(df))
                map_path = create_gis_map(structured)
                st.success("Map created.")
                components.html(open(map_path, "r", encoding="utf-8").read(), height=600)

            st.subheader("🔮 Predict Next Trafficking Location")
            if st.button("Run Predictive Model"):
                structured = st.session_state.get("structured_data", run_nlp_pipeline(df))
                preds = run_prediction_pipeline(structured)
                st.success("Prediction complete.")
                st.dataframe(preds)

            st.subheader("🔍 Query Builder & Filter Interface")
            nationality = st.selectbox("Filter by Nationality", options=[""] + sorted(df["Nationality of Victim"].dropna().unique()))
            gender = st.selectbox("Filter by Gender", options=["", "Male", "Female"])
            year_range = st.slider("Left Home Country (Year Range)",
                min_value=int(df["Left Home Country Year"].min()),
                max_value=int(df["Left Home Country Year"].max()),
                value=(2010, 2020)
            )
            location = st.text_input("Contains Location (partial match)")
            if st.button("Apply Filters"):
                filtered = apply_filters(df, nationality or None, gender or None, year_range, location or None)
                st.success(f"{len(filtered)} records matched.")
                st.dataframe(filtered)

            st.subheader("📊 Dashboard: Summary Metrics & Visualizations")
            total, nat_dist, gen_dist, chart = run_metrics_pipeline(df)
            st.metric("Total Unique Victims", total)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Nationality Distribution**")
                st.json(nat_dist)
            with col2:
                st.markdown("**Gender Distribution**")
                st.json(gen_dist)
            st.image(chart, use_container_width=True)

            st.subheader("🧠 GNN Node Classification (Victim / Location / Perpetrator)")
            if st.button("Run GNN Classification"):
                structured = st.session_state.get("structured_data", run_nlp_pipeline(df))
                data, encoder = build_gnn_data(structured)
                model, out = train_gnn(data, num_classes=len(encoder.classes_))
                preds = out.argmax(dim=1).numpy()
                labels = [encoder.inverse_transform([p])[0] for p in preds]
                st.json({f"Node {i}": l for i, l in enumerate(labels)})

# 📁 Template Download
st.markdown("#### 📄 Download Upload Format Template")
sample = pd.DataFrame(columns=[
    "Unique ID", "Interviewer Name", "Date of Interview",
    "Gender of Victim", "Nationality of Victim", "Left Home Country Year",
    "Borders Crossed", "City / Locations Crossed", "Final Location",
    "Name of the Perpetrators involved", "Hierarchy of Perpetrators",
    "Human traffickers/ Chief of places", "Time Spent in Location / Cities / Places"
])
st.download_button("📥 Download Template (CSV)", sample.to_csv(index=False), file_name="trafficking_upload_template.csv")

# 📜 Consent
with st.expander("📜 Data Consent & FAIR Compliance Notice"):
    st.markdown("""
- By uploading, you confirm the data adheres to FAIR and ethical principles.
- Personally identifiable information must be anonymized.
- Qualitative abuse details should be excluded.
- System access is controlled by user roles.
- Uploaded data may be cleaned, validated, and merged.

*Your contribution enables responsible and collaborative anti-trafficking research.*
""")
