# /backend/api/nlp.py

from backend.nlp.entity_extraction import extract_entities
from backend.nlp.topic_modeling import get_topics
import numpy as np
import re

def normalize_perpetrators(text):
    if not text or isinstance(text, float) or str(text).lower().strip() in ["no", "none", "not specified", "yes", "yes but", "n/a"]:
        return ""
    # Replace ' and ' with comma for standardization
    text = re.sub(r"\s+and\s+", ", ", text.strip())
    # Remove trailing punctuation
    text = re.sub(r"[\.|;]+$", "", text)
    return text

def normalize_time_spent(text):
    if not isinstance(text, str) or text.strip().lower() in ["n/a", "not applicable", ""]:
        return ""
    text = text.lower().strip()

    # Try extracting numbers and convert to rough days
    if "month" in text:
        match = re.search(r"(\d+)", text)
        if match:
            return int(match.group(1)) * 30
    elif "week" in text:
        match = re.search(r"(\d+)", text)
        if match:
            return int(match.group(1)) * 7
    elif "day" in text:
        match = re.search(r"(\d+)", text)
        if match:
            return int(match.group(1))
        elif "3/4" in text:
            return 4
    return ""

def clean_hierarchy(text):
    if not text or isinstance(text, float) or str(text).strip().lower() in ["not applicable", "not aplicable", "n/a"]:
        return ""
    return text.strip()

def run_nlp_pipeline(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]  # Clean headers

    structured_rows = []

    for _, row in df.iterrows():
        # Build composite narrative for location extraction
        text = " | ".join([
            str(row.get("City / Locations Crossed", "")),
            str(row.get("Borders Crossed", "")),
            str(row.get("Name of the Perpetrators involved", "")),
            str(row.get("Human traffickers/ Chief of places", "")),
            str(row.get("Hierarchy of Perpetrators", ""))
        ])

        entities = extract_entities(text)
        locations = [loc for loc in entities.get("locations", []) if str(loc).lower() != "nan"]

        structured_rows.append({
            "Unique ID": row.get("Unique ID", ""),
            "Interviewer Name": row.get("Interviewer Name", ""),
            "Date of Interview": row.get("Date of Interview", ""),
            "Gender of Victim": row.get("Gender of Victim", ""),
            "Nationality of Victim": row.get("Nationality of Victim", ""),
            "Left Home Country Year": row.get("Left Home Country Year", ""),
            "Borders Crossed": row.get("Borders Crossed", ""),
            "City / Locations Crossed": row.get("City / Locations Crossed", ""),
            "Final Location": row.get("Final Location", ""),
            "Name of the Perpetrators involved": normalize_perpetrators(row.get("Name of the Perpetrators involved", "")),
            "Hierarchy of Perpetrators": clean_hierarchy(row.get("Hierarchy of Perpetrators", "")),
            "Human traffickers/ Chief of places": row.get("Human traffickers/ Chief of places", ""),
            "Time Spent in Location / Cities / Places": normalize_time_spent(row.get("Time Spent in Location / Cities / Places", "")),
            "Locations (NLP)": ", ".join(locations)
        })

    return structured_rows
