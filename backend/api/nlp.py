from backend.nlp.entity_extraction import extract_entities
from backend.nlp.topic_modeling import get_topics
import re
import pandas as pd

def clean_time_spent(value):
    """Convert time descriptions to numeric days."""
    value = str(value).lower().strip()
    if value in ["", "nan", "none", "not sure", "not specified"]:
        return ""

    value = value.replace("approximately", "").replace("about", "").strip()

    if "month" in value:
        match = re.search(r"\d+", value)
        return str(int(match.group()) * 30) if match else ""

    if "day" in value:
        match = re.search(r"\d+", value)
        return str(int(match.group())) if match else ""

    if "/" in value:
        parts = value.split("/")
        try:
            return str(round(sum([float(p) for p in parts]) / len(parts)))
        except:
            return ""

    match = re.search(r"\d+", value)
    return str(match.group()) if match else ""

def clean_perpetrators(text):
    """Convert to comma-separated names, empty if 'No' or invalid."""
    if pd.isna(text) or str(text).strip().lower() in ["no", "none", "nan"]:
        return ""
    text = re.sub(r"\s+and\s+", ", ", str(text).strip(), flags=re.IGNORECASE)
    return text

def clean_hierarchy(value):
    """Remove 'not applicable' or blank out if no valid data."""
    if pd.isna(value) or str(value).strip().lower() in ["not applicable", "none", "nan"]:
        return ""
    return str(value).strip()

def clean_location_list(locations):
    """Remove 'nan' and deduplicate."""
    if not isinstance(locations, list):
        return ""
    clean = [loc for loc in locations if str(loc).lower() != "nan"]
    return list(set(clean))

def run_nlp_pipeline(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]  # Clean headers

    structured_rows = []

    for _, row in df.iterrows():
        # Combine text for richer NLP signal
        text = " | ".join([
            str(row.get("City / Locations Crossed", "")),
            str(row.get("Borders Crossed", "")),
            str(row.get("Name of the Perpetrators involved", "")),
            str(row.get("Human traffickers/ Chief of places", "")),
            str(row.get("Hierarchy of Perpetrators", ""))
        ])

        entities = extract_entities(text)

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
            "Name of the Perpetrators involved": clean_perpetrators(row.get("Name of the Perpetrators involved", "")),
            "Hierarchy of Perpetrators": clean_hierarchy(row.get("Hierarchy of Perpetrators", "")),
            "Human traffickers/ Chief of places": row.get("Human traffickers/ Chief of places", ""),
            "Time Spent in Location / Cities / Places": clean_time_spent(row.get("Time Spent in Location / Cities / Places", "")),
            "Locations (NLP)": clean_location_list(entities.get("locations", []))
        })

    return structured_rows
