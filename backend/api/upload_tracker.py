# /backend/api/upload_tracker.py
import datetime

UPLOAD_LOG = []  # In-memory list (could be DB in real app)

def log_upload(file_name, user_email):
    UPLOAD_LOG.append({
        "file": file_name,
        "user": user_email,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def get_upload_history():
    return UPLOAD_LOG
