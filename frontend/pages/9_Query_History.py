import streamlit as st
import pandas as pd

st.set_page_config(page_title="🧾 Query History", layout="wide")
st.title("🧾 Query History (Admin & Researcher Only)")

# Role restriction
if st.session_state.get("role") not in ["Admin", "Researcher"]:
    st.warning("Access denied: This page is only for Admins and Researchers.")
    st.stop()

# Initialize query history storage if not present
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# Display query history
if st.session_state.query_history:
    st.markdown("### 🔍 Logged Queries")
    df = pd.DataFrame(st.session_state.query_history)
    st.dataframe(df)

    # Optional: delete queries
    indices_to_delete = st.multiselect("Select queries to delete (by index)", df.index.tolist())
    if st.button("Delete Selected Queries"):
        st.session_state.query_history = [
            q for i, q in enumerate(st.session_state.query_history) if i not in indices_to_delete
        ]
        st.success("Selected queries deleted successfully.")
        st.experimental_rerun()
else:
    st.info("No query history logged yet.")
