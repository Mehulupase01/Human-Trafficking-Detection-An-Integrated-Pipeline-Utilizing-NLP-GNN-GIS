import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(page_title="👥 User Manager", layout="wide")
st.title("👥 Admin: Role Assignment & User Manager")

if st.session_state.get("role") != "Admin":
    st.warning("Access denied: Admins only.")
    st.stop()

st.markdown("### 🔍 Current Users")
# Simulated user table
users = pd.DataFrame({
    "Email": ["a@x.com", "b@y.com", "c@z.com"],
    "Role": ["Researcher", "Viewer", "Data Owner"]
})
st.dataframe(users)

st.markdown("### 🛠 Modify User Role")
selected_user = st.selectbox("Select User", users["Email"])
new_role = st.selectbox("Assign New Role", ["Admin", "Researcher", "Viewer", "Data Owner"])
if st.button("Update Role"):
    st.success(f"Role for {selected_user} updated to {new_role}. (Simulated)")

st.markdown("### ❌ Delete User")
delete_user = st.selectbox("Select User to Delete", users["Email"])
if st.button("Delete User"):
    st.warning(f"User {delete_user} deleted. (Simulated)")

st.markdown("### 📤 Bulk User Import (CSV)")
st.markdown("Upload a CSV with columns: `Email`, `Role`")

sample = """Email,Role
john@example.com,Researcher
jane@example.com,Viewer
"""
st.download_button("📄 Download Sample CSV", sample, file_name="bulk_users_template.csv")

bulk_file = st.file_uploader("Upload User CSV", type="csv", key="bulk")
if bulk_file:
    df_bulk = pd.read_csv(bulk_file)
    st.dataframe(df_bulk)
    if st.button("Import Users"):
        st.success(f"{len(df_bulk)} users imported successfully. (Simulated)")
