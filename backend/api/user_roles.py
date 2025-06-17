import os

# Use this to switch between offline and Firebase mode
USE_LOCAL_AUTH = os.getenv("USE_LOCAL_AUTH", "true").lower() == "true"

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
