.PHONY: install test run docker-build docker-run

install:
\tpip install -r requirements.txt

test:
\tpytest -q

run:
\tstreamlit run frontend/streamlit_app.py

docker-build:
\tdocker build -t thesis-app:latest .

docker-run:
\tdocker run --rm -p 8501:8501 -e APP_DATA_DIR=/app/data -v ${PWD}/data:/app/data thesis-app:latest
