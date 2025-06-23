import streamlit as st
import os
import json
from datetime import datetime
from backend.api.auth import send_otp_email, update_password, update_email_verified
from backend.api.user_roles import get_user_profile, save_user_profile

st.set_page_config(page_title="⚙️ Settings", layout="wide")
st.title("⚙️ User Settings Panel")

# Access control
if "email" not in st.session_state or "role" not in st.session_state:
    st.warning("Please login first.")
    st.stop()

user_email = st.session_state["email"]
role = st.session_state["role"]

# Load profile (name, org, country, etc.)
profile = get_user_profile(user_email)

st.subheader("👤 Profile Information")
st.markdown(f"**Email:** `{user_email}`")
st.markdown(f"**Role:** `{role}`")
st.text_input("First Name", value=profile.get("first_name", ""), key="first_name")
st.text_input("Last Name", value=profile.get("last_name", ""), key="last_name")
st.text_input("Organization", value=profile.get("organization", ""), key="organization")
st.text_input("Country", value=profile.get("country", ""), key="country")

if st.button("Save Profile"):
    save_user_profile(user_email, {
        "first_name": st.session_state["first_name"],
        "last_name": st.session_state["last_name"],
        "organization": st.session_state["organization"],
        "country": st.session_state["country"]
    })
    st.success("Profile updated!")

# Profile picture
st.subheader("🖼️ Profile Picture")
img_path = f"frontend/profile_pics/{user_email.replace('@', '_at_')}.png"
if os.path.exists(img_path):
    st.image(img_path, width=120)

upload = st.file_uploader("Upload new profile image (PNG)", type=["png"])
if upload:
    os.makedirs("frontend/profile_pics", exist_ok=True)
    with open(img_path, "wb") as f:
        f.write(upload.read())
    st.success("Profile picture updated!")
    st.experimental_rerun()

# Change email
st.subheader("📧 Change Email")
new_email = st.text_input("New Email Address")
if st.button("Send Verification Email"):
    otp = send_otp_email(new_email)
    st.session_state["pending_email"] = new_email
    st.session_state["email_otp"] = otp
    st.success("Verification email sent. Enter OTP below to confirm.")

entered_otp = st.text_input("Enter OTP for email verification")
if st.button("Verify & Change Email"):
    if entered_otp == st.session_state.get("email_otp"):
        update_email_verified(user_email, st.session_state["pending_email"])
        st.success("Email updated successfully. Please log in again.")
        st.session_state.clear()
        st.experimental_rerun()
    else:
        st.error("Invalid OTP")

# Change password
st.subheader("🔑 Change Password")
old_password = st.text_input("Old Password", type="password")
new_password = st.text_input("New Password", type="password")
confirm = st.text_input("Confirm New Password", type="password")
if st.button("Update Password"):
    if new_password != confirm:
        st.error("Passwords do not match")
    elif not old_password or not new_password:
        st.error("Please enter all fields")
    else:
        result = update_password(user_email, old_password, new_password)
        if result.get("success"):
            st.success("Password updated.")
        else:
            st.error(result.get("error", "Failed to update."))