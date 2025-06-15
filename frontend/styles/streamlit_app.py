# 4. frontend/streamlit_app.py
import streamlit as st
from backend.api.auth import login_user, signup_user
from backend.api.user_roles import get_user_role

st.set_page_config(page_title="Trafficking Analytics App", layout="centered")

st.title("🔐 Secure Login Portal")

menu = ["Login", "SignUp"]
choice = st.selectbox("Menu", menu)

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

            if role == "Admin":
                st.info("Welcome Admin. You can assign roles and view all datasets.")
            elif role == "Researcher":
                st.info("Welcome Researcher. You can run analytics and visualize networks.")
            elif role == "Data Owner":
                st.info("Welcome Data Owner. You can upload datasets.")
            else:
                st.info("Welcome Viewer. You have read-only access.")

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