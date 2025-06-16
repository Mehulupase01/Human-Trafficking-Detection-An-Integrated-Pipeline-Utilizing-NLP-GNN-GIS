from bertopic import BERTopic

def get_topics(texts: List[str]) -> List[str]:
    model = BERTopic()
    topics, _ = model.fit_transform(texts)
    return model.get_topic_info()

# 3. backend/api/nlp.py
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