import os
from backend.storage.firebase_config import auth
USE_LOCAL_AUTH = os.getenv("USE_LOCAL_AUTH", "true").lower() == "true"

# 🔒 Local test users
local_users = {
    "mehulupase@gmail.com": {"password": "Mehulupase01#", "role": "Admin"},
    "upasemehul@gmail.com": {"password": "Abcd@1234", "role": "Researcher"},
    "scaletheotherside@gmail.com": {"password": "Wxyz@1234", "role": "Data Owner"}
}

# 🔐 Firebase fallback
if not USE_LOCAL_AUTH:
    from backend.storage.firebase_config import auth

def login_user(email, password):
    if USE_LOCAL_AUTH:
        user = local_users.get(email)
        if user and user["password"] == password:
            return {"email": email, "role": user["role"]}
        return {"error": "Invalid credentials"}
    else:
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            return {"email": email}
        except Exception as e:
            return {"error": str(e)}

def signup_user(email, password):
    if USE_LOCAL_AUTH:
        return {"error": "Sign-up is disabled in local mode."}
    else:
        try:
            user = auth.create_user_with_email_and_password(email, password)
            auth.send_email_verification(user['idToken'])
            return user
        except Exception as e:
            return {"error": str(e)}
