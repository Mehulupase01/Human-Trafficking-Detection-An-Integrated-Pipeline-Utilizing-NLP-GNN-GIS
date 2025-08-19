# frontend/_debug.py
from __future__ import annotations
import os, traceback
import streamlit as st

def debug_enabled() -> bool:
    return os.environ.get("APP_DEBUG", "0") == "1"

def show_exception(e: Exception, note: str = ""):
    if debug_enabled():
        st.exception(e)
    else:
        msg = f"{note}\n{type(e).__name__}: {e}" if note else f"{type(e).__name__}: {e}"
        st.error(msg)

def guard(fn):
    """Decorator: run a function and render full trace in debug mode."""
    def _inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            show_exception(e, note=f"While running {fn.__name__}")
            return None
    return _inner
