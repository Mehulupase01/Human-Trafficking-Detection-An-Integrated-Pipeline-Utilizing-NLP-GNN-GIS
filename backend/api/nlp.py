from backend.nlp.entity_extraction import extract_entities
from backend.nlp.topic_modeling import get_topics

def run_nlp_pipeline(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    structured_rows = []

    for _, row in df.iterrows():
        # Concatenate all narrative-style fields for NLP
        text = " | ".join([
            str(row.get("City / Locations Crossed", "")),
            str(row.get("Borders Crossed", "")),
            str(row.get("Name of the Perpetrators involved", "")),
            str(row.get("Human traffickers/ Chief of places", "")),
            str(row.get("Hierarchy of Perpetrators", ""))
        ])
        entities = extract_entities(text)

        structured_rows.append({
            "Victim ID": row.get("Unique ID", "Unknown"),
            "Gender": row.get("Gender of Victim", ""),
            "Nationality": row.get("Nationality of Victim", ""),
            "Left Home Country Year": row.get("Left Home Country Year", ""),
            "Cities": row.get("City / Locations Crossed", ""),
            "Borders Crossed": row.get("Borders Crossed", ""),
            "Final Location": row.get("Final Location", ""),
            "Perpetrators": row.get("Name of the Perpetrators involved", ""),
            "Hierarchy": row.get("Hierarchy of Perpetrators", ""),
            "Chiefs": row.get("Human traffickers/ Chief of places", ""),
            "Time Spent": row.get("Time Spent in Location / Cities / Places", ""),
            # NLP Extracted:
            "Names (NLP)": entities.get("names", []),
            "Locations (NLP)": entities.get("locations", []),
            "Organizations (NLP)": entities.get("organizations", [])
        })

    return structured_rows