from backend.nlp.entity_extraction import extract_entities
from backend.nlp.topic_modeling import get_topics

def run_nlp_pipeline(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]  # Clean headers

    structured_rows = []

    for _, row in df.iterrows():
        # Combine narrative fields for NLP
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
            "Name of the Perpetrators involved": row.get("Name of the Perpetrators involved", ""),
            "Hierarchy of Perpetrators": row.get("Hierarchy of Perpetrators", ""),
            "Human traffickers/ Chief of places": row.get("Human traffickers/ Chief of places", ""),
            "Time Spent in Location / Cities / Places": row.get("Time Spent in Location / Cities / Places", ""),
            # NLP-enhanced insights
            "Names (+NLP)": entities.get("names", []),
            "Locations (+NLP)": entities.get("locations", []),
            "Organizations (+NLP)": entities.get("organizations", [])
        })

    return structured_rows