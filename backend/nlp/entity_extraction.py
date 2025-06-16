import spacy
from spacy.tokens import DocBin
from typing import List, Dict

nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str) -> Dict[str, List[str]]:
    doc = nlp(text)
    people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    gpe = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return {
        "names": list(set(people)),
        "locations": list(set(gpe)),
        "organizations": list(set(orgs))
    }
