# frontend/_bootstrap.py
from __future__ import annotations
import os, sys
from dotenv import load_dotenv

load_dotenv()

# Ensure project root (parent of frontend/) is on PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Guarantee a single shared data dir for the registry when local
if not os.environ.get("APP_DATA_DIR"):
    os.environ["APP_DATA_DIR"] = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(os.environ["APP_DATA_DIR"], exist_ok=True)
