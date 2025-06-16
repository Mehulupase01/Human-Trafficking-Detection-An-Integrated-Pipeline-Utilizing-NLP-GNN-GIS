from backend.nlp.entity_extraction import extract_entities
from backend.nlp.topic_modeling import get_topics

def run_nlp_pipeline(dataframe):
    structured_rows = []
    for _, row in dataframe.iterrows():
        narrative = row.get("City / Locations Crossed", "")
        result = extract_entities(narrative)
        structured_rows.append({
            "Victim ID": row["Unique ID (Victim)"],
            "Names": result["names"],
            "Locations": result["locations"],
            "Organizations": result["organizations"]
        })
    return structured_rows