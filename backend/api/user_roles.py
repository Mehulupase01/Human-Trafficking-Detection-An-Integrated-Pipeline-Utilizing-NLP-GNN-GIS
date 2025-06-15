# 3. backend/api/user_roles.py
user_roles = {
    "admin@example.com": "Admin",
    "researcher@example.com": "Researcher",
    "viewer@example.com": "Viewer",
    "dataowner@example.com": "Data Owner"
}

def get_user_role(email):
    return user_roles.get(email, "Viewer")