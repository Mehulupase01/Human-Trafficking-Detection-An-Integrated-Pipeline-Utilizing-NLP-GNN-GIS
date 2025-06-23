import os
import json
# Use this to switch between offline and Firebase mode
USE_LOCAL_AUTH = os.getenv("USE_LOCAL_AUTH", "true").lower() == "true"

PROFILE_DIR = "database/user_profiles"
os.makedirs(PROFILE_DIR, exist_ok=True)

def get_user_profile(email):
    filename = os.path.join(PROFILE_DIR, f"{email.replace('@', '_at_')}.json")
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {}

def save_user_profile(email, data):
    filename = os.path.join(PROFILE_DIR, f"{email.replace('@', '_at_')}.json")
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


# 🔐 Offline/local fallback roles
local_user_roles = {
    "mehulupase@gmail.com": "Admin",
    "upasemehul@gmail.com": "Researcher",
    "scaletheotherside@gmail.com": "Data Owner"
}

def get_user_role(email):
    # Always respect hardcoded users (even when online)
    if email in local_user_roles:
        return local_user_roles[email]

    # Online mode fallback
    if not USE_LOCAL_AUTH:
        # ⚠️ Extend this logic: fetch from DB or default role
        return "Viewer"  # fallback for unknown Firebase users

    return "Viewer"  # Fallback for unknown local users
