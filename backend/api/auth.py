import os
import json
import hashlib
import random
from datetime import datetime

USE_LOCAL_AUTH = os.getenv("USE_LOCAL_AUTH", "true").lower() == "true"

# 🔐 Firebase setup
if not USE_LOCAL_AUTH:
    from backend.storage.firebase_config import auth, db

# 🔒 Local test users
USERS_FILE = "database/users.json"
VERIFIED_FILE = "database/verified_emails.json"
local_users = {
    "mehulupase@gmail.com": {"password": "Mehulupase01#", "role": "Admin", "verified": True},
    "upasemehul@gmail.com": {"password": "Abcd@1234", "role": "Researcher", "verified": True},
    "scaletheotherside@gmail.com": {"password": "Wxyz@1234", "role": "Data Owner", "verified": True}
}

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

# 🔐 LOCAL MODE
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    os.makedirs("database", exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def load_verified():
    if os.path.exists(VERIFIED_FILE):
        with open(VERIFIED_FILE, "r") as f:
            return json.load(f)
    return {}

def save_verified(data):
    os.makedirs("database", exist_ok=True)
    with open(VERIFIED_FILE, "w") as f:
        json.dump(data, f, indent=2)

def send_otp_email(email):
    otp = str(random.randint(100000, 999999))
    print(f"[LOCAL MODE] OTP for {email}: {otp}")
    verified = load_verified()
    verified[email] = {"otp": otp, "timestamp": str(datetime.now())}
    save_verified(verified)
    return otp

def verify_email_otp(email, otp):
    verified = load_verified()
    if email in verified and verified[email]["otp"] == otp:
        users = load_users()
        if email in users:
            users[email]["verified"] = True
            save_users(users)
            del verified[email]
            save_verified(verified)
            return {"success": True}
    return {"error": "Invalid or expired OTP"}

# 🟢 Firebase Signup with Metadata
def signup_user(email, password, metadata=None):
    if USE_LOCAL_AUTH:
        users = load_users()
        if email in users or email in local_users:
            return {"error": "User already exists."}
        users[email] = {
            "password": hash_pw(password),
            "role": "Pending",
            "verified": False,
            "metadata": metadata or {}
        }
        save_users(users)
        send_otp_email(email)
        return {"success": True}
    else:
        try:
            user = auth.create_user_with_email_and_password(email, password)
            auth.send_email_verification(user['idToken'])
            uid = user['localId']
            db.child("users").child(uid).set({
                "email": email,
                "role": "Pending",
                "verified": False,
                "metadata": metadata or {}
            })
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}

def login_user(email, password):
    if USE_LOCAL_AUTH:
        user = local_users.get(email)
        if user and user["password"] == password:
            if not user.get("verified", False):
                return {"error": "Email not verified"}
            return {"email": email, "role": user["role"]}
        users = load_users()
        u = users.get(email)
        if u and u["password"] == hash_pw(password):
            if not u.get("verified", False):
                return {"error": "Email not verified"}
            return {"email": email, "role": u.get("role", "Pending")}
        return {"error": "Invalid credentials"}
    else:
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            return {"email": email}
        except Exception as e:
            return {"error": str(e)}

def update_password(email, old_pw, new_pw):
    if USE_LOCAL_AUTH:
        users = load_users()
        u = users.get(email)
        if u and u["password"] == hash_pw(old_pw):
            u["password"] = hash_pw(new_pw)
            save_users(users)
            return {"success": True}
        return {"error": "Old password incorrect"}
    else:
        return {"error": "Password update not supported in Firebase UI yet"}

def update_email_verified(old_email, new_email):
    if USE_LOCAL_AUTH:
        users = load_users()
        if old_email in users:
            users[new_email] = users.pop(old_email)
            users[new_email]["verified"] = True
            save_users(users)
            return True
    else:
        return False
