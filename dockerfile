# ---- base ----
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    APP_DATA_DIR=/app/data \
    STREAMLIT_SERVER_PORT=8501

WORKDIR ${APP_HOME}

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# ---- deps ----
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- app ----
COPY backend ./backend
COPY frontend ./frontend

# Streamlit default config
RUN mkdir -p ~/.streamlit && printf "\
[server]\nheadless = true\nport = 8501\n\n\
[browser]\ngatherUsageStats = false\n" > ~/.streamlit/config.toml

EXPOSE 8501

# Default: run Streamlit app (you can change the entrypoint if your main file differs)
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.address=0.0.0.0"]
