import spacy
from spacy.tokens import DocBin
from typing import List, Dict

nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str) -> Dict[str, list]:
    if not isinstance(text, str) or not text.strip():
        return {"names": [], "locations": [], "organizations": []}

    doc = nlp(text)
    people = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
    gpe = [ent.text.strip() for ent in doc.ents if ent.label_ == "GPE"]
    orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]

    def clean(lst):
        return sorted(set(x for x in lst if str(x).strip().lower() not in ["", "nan", "none"]))

    return {
        "names": clean(people),
        "locations": clean(gpe),
        "organizations": clean(orgs)
    }
