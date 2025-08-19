# frontend/streamlit_app.py
from __future__ import annotations

# --- .env + import path like your original ---
from dotenv import load_dotenv
load_dotenv()
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- std libs / UI ---
import datetime as dt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import streamlit as st
import pandas as pd
import base64  # kept for parity with your original (used in some variants)
import streamlit.components.v1 as components  # optional, used by some pages

# --- backend registry + optional gazetteer ---
from backend.core import dataset_registry as registry
try:
    from backend.geo.gazetteer import list_gazetteers, set_active_gazetteer, get_active_gazetteer
    _HAS_GAZ = True
except Exception:
    _HAS_GAZ = False

# --- optional auth (kept compatible with your original) ---
try:
    from backend.api.auth import login_user, signup_user, verify_email_otp
    from backend.api.user_roles import get_user_role
    _HAS_AUTH = True
except Exception:
    _HAS_AUTH = False

# ---------------- Page setup ----------------
st.set_page_config(
    page_title="Trafficking Analytics App",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================== SIDEBAR: GLOBAL CONTROLS =====================
with st.sidebar:
    st.header("⚙️ Global Controls")

    # Account (optional auth)
    if _HAS_AUTH:
        st.subheader("👤 Account")
        if "email" not in st.session_state or "role" not in st.session_state:
            mode = st.radio("Sign in or create an account", ["Login", "SignUp"], horizontal=True)
            if mode == "Login":
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.button("Login", use_container_width=True):
                    result = login_user(email, password)
                    if "error" in result:
                        st.error(f"Login failed: {result['error']}")
                    else:
                        st.success("Logged in successfully!")
                        st.session_state["email"] = email
                        st.session_state["role"] = get_user_role(email)
                        st.rerun()
            else:  # SignUp
                col1, col2 = st.columns(2)
                first_name = col1.text_input("First Name")
                last_name  = col2.text_input("Last Name")
                email = st.text_input("Email Address")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                col3, col4 = st.columns(2)
                org_type = col3.selectbox("Organization Type", ["NGO", "Company", "University", "Government", "Other"])
                country  = col4.text_input("Country", value="")
                org_name = st.text_input("Organization Name")
                if "signed_up_email" not in st.session_state:
                    if st.button("Sign Up", use_container_width=True):
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
                    st.info(f"Enter the OTP sent to `{st.session_state['signed_up_email']}` to complete verification.")
                    otp = st.text_input("Enter OTP")
                    if st.button("Verify OTP", use_container_width=True):
                        result = verify_email_otp(st.session_state["signed_up_email"], otp)
                        if "success" in result:
                            st.success("Email verified successfully. Please log in.")
                            del st.session_state["signed_up_email"]
                        else:
                            st.error("Invalid or expired OTP.")
        else:
            st.success(f"Logged in: `{st.session_state['email']}`\n\nRole: `{st.session_state['role']}`")
            if st.button("Logout", use_container_width=True):
                for k in ["email", "role", "signed_up_email"]:
                    st.session_state.pop(k, None)
                st.rerun()
        st.divider()

    # Data dir + artifact counts
    data_dir = os.getenv("APP_DATA_DIR", "(not set)")
    st.caption(f"**APP_DATA_DIR**: `{data_dir}`")

    def _count(kind: str) -> int:
        try:
            return len(registry.list_datasets(kind=kind))
        except Exception:
            return 0

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Processed", _count("processed"))
        st.metric("Merged", _count("merged"))
        st.metric("Saved Queries", _count("saved_query"))
    with c2:
        st.metric("Pred Runs", _count("prediction_run") + _count("perp_prediction_run"))
        st.metric("Eval Reports", _count("evaluation_report"))
        st.metric("Gazetteers", _count("gazetteer"))

    st.divider()

    # Gazetteer selector (optional)
    if _HAS_GAZ:
        st.subheader("🗺️ Active Gazetteer")
        try:
            gzs = list_gazetteers()
        except Exception:
            gzs = []
        if gzs:
            try:
                active = get_active_gazetteer()
            except Exception:
                active = None
            def _fmt_gz(e: dict) -> str:
                return f"{e.get('name','(unnamed)')} • {e.get('id','')}"
            idx = 0
            if active:
                for i, g in enumerate(gzs):
                    if g.get("id") == active:
                        idx = i; break
            choice = st.selectbox("Select gazetteer", options=gzs, index=idx, format_func=_fmt_gz)
            if st.button("Activate Gazetteer", use_container_width=True):
                try:
                    set_active_gazetteer(choice.get("id"))
                    st.success(f"Activated: {choice.get('name')} • {choice.get('id')}")
                except Exception as e:
                    st.error(f"Failed to activate gazetteer: {e}")
        else:
            st.caption("No gazetteers found. Add one from the GIS/Geo tools page.")

    st.divider()

    # Cache & state
    st.subheader("🧼 Cache & State")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Clear cache", use_container_width=True):
            try:
                st.cache_data.clear()
                st.success("Cleared @cache_data.")
            except Exception as e:
                st.error(f"Cache clear failed: {e}")
    with colB:
        if st.button("Reset session", use_container_width=True):
            st.session_state.clear()
            st.success("Session state reset. Refresh the page.")

    st.divider()

    # Quick links to pages
    st.subheader("🔗 Quick Links")
    # inside the LINKS list in streamlit_app.py
    LINKS = [
        ("pages/0_Upload_Standardize.py", "📤 Upload & Standardize"),  # ⟵ add this line
        ("pages/1_Merge_Datasets.py", "🧩 Merge Datasets"),
        ("pages/3_Query_Insights.py", "🔎 Query & Insights"),
        ("pages/4_Network_Graphs.py", "🕸️ Network Graphs"),
        ("pages/5_Temporal_Forecast.py", "⏱️ Temporal Forecast (ETA)"),
        ("pages/6_GNN_Trafficker_Prediction.py", "🔮 Predictive Analytics"),
        ("pages/8_Map_GIS_Visualizer.py", "🗺️ GIS Map & Spatio-Temporal"),
        ("pages/13_Ontology_TTL_Merge.py", "🧩 Ontology (TTL) + Merge"),
        ("pages/14_Summary_Dashboard.py", "📊 Summary Dashboard"),
        ("pages/15_Automated_Evaluations.py", "🧪 Automated Evaluations"),
        ("pages/16_User_Guide.py", "📘 User Guide"),
    ]

    if hasattr(st, "page_link"):
        for path, label in LINKS:
            st.page_link(path, label=label, icon=None, use_container_width=True)
    else:
        for path, label in LINKS:
            st.markdown(f"- [{label}]({path})")
            
            

from _debug import debug_enabled  # at top with other imports

# ... inside with st.sidebar: ...
st.subheader("🐞 Debug")
dbg = st.toggle("Enable debug traces", value=os.environ.get("APP_DEBUG","0")=="1")
os.environ["APP_DEBUG"] = "1" if dbg else "0"
st.caption("When on, pages will show full Python tracebacks.")


# ===================== MAIN: HOME HUB =====================
st.title("🚨 Human Trafficking Analytics Platform")
if _HAS_AUTH and "role" in st.session_state and "email" in st.session_state:
    st.caption(f"**Logged in as:** `{st.session_state['email']}` | **Role:** `{st.session_state['role']}`")
else:
    st.caption("You can browse pages directly; if your deployment requires auth, sign in from the sidebar.")

st.markdown("""
**Welcome!** Use the pages in the sidebar (or links above) to merge datasets, explore, visualize networks & maps, run predictions, export ontology TTL, and evaluate quality.
""")

st.divider()

# --------- Recent artifacts ---------
st.subheader("🗂️ Recent Artifacts")

def _list_recent(kind: str, n: int = 6):
    try:
        items = registry.list_datasets(kind=kind)
    except Exception:
        items = []
    def _ts(x):
        v = x.get("created_at")
        try:
            return dt.datetime.fromisoformat(v.replace("Z","")) if isinstance(v, str) else dt.datetime.min
        except Exception:
            return dt.datetime.min
    items = sorted(items, key=_ts, reverse=True)[:n]
    return items

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Merged datasets**")
    rec = _list_recent("merged")
    if rec:
        for e in rec:
            st.write(f"• **{e.get('name','Merged')}** — `{e.get('id','')}` · {e.get('created_at','')}")
    else:
        st.caption("None yet.")
with col2:
    st.markdown("**Prediction runs**")
    preds = _list_recent("prediction_run") + _list_recent("perp_prediction_run")
    if preds:
        for e in preds[:6]:
            st.write(f"• **{e.get('name','Prediction')}** — `{e.get('id','')}` · {e.get('created_at','')}")
    else:
        st.caption("None yet.")
with col3:
    st.markdown("**Evaluation reports**")
    rec = _list_recent("evaluation_report")
    if rec:
        for e in rec:
            st.write(f"• **{e.get('name','Evaluation')}** — `{e.get('id','')}` · {e.get('created_at','')}")
    else:
        st.caption("None yet.")

st.divider()

# --------- Getting started ---------
st.subheader("🚀 Getting Started")
gs1, gs2, gs3 = st.columns(3)
with gs1:
    st.markdown("**1) Merge & Standardize**")
    st.write("Combine processed datasets with true de-duplication and conflict resolution.")
    if hasattr(st, "page_link"):
        st.page_link("pages/1_Merge_Datasets.py", label="Open Merge Datasets →", use_container_width=True)
with gs2:
    st.markdown("**2) Explore & Filter**")
    st.write("Use processed fields only — Any-gender, searchable locations, saved queries.")
    if hasattr(st, "page_link"):
        st.page_link("pages/3_Query_Insights.py", label="Open Query & Insights →", use_container_width=True)
with gs3:
    st.markdown("**3) Analyze & Visualize**")
    st.write("Graphs, GIS overlays, predictions, and evaluations in dedicated pages.")
    if hasattr(st, "page_link"):
        st.page_link("pages/6_GNN_Trafficker_Prediction.py", label="Open Predictive Analytics →", use_container_width=True)

st.divider()

# --------- Upload template (kept from your original) ---------
st.markdown("#### 📄 Download Upload Format Template")
sample = pd.DataFrame(columns=[
    "Unique ID", "Interviewer Name", "Date of Interview",
    "Gender of Victim", "Nationality of Victim", "Left Home Country Year",
    "Borders Crossed", "City / Locations Crossed", "Final Location",
    "Name of the Perpetrators involved", "Hierarchy of Perpetrators",
    "Human traffickers/ Chief of places", "Time Spent in Location / Cities / Places"
])
st.download_button("📥 Download Template (CSV)", sample.to_csv(index=False), file_name="trafficking_upload_template.csv")

# --------- Consent notice (kept from your original) ---------
with st.expander("📜 Data Consent & FAIR Compliance Notice"):
    st.markdown("""
- By uploading, you confirm the data adheres to FAIR and ethical principles.
- Personally identifiable information must be anonymized.
- Qualitative abuse details should be excluded.
- System access is controlled by user roles.
- Uploaded data may be cleaned, validated, and merged.

*Your contribution enables responsible and collaborative anti-trafficking research.*
""")
