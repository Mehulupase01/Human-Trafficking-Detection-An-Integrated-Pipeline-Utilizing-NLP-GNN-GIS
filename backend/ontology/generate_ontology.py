from owlready2 import *
from rdflib import Graph
import uuid
import os

def build_ontology(data, ontology_name="TraffickingOntology"):
    onto_path.append("./")
    onto = get_ontology(f"http://example.org/{ontology_name}#{uuid.uuid4()}")

    with onto:
        class Victim(Thing): pass
        class Location(Thing): pass
        class Perpetrator(Thing): pass
        class passedThrough(ObjectProperty):
            domain = [Victim]
            range = [Location]
        class interactedWith(ObjectProperty):
            domain = [Victim]
            range = [Perpetrator]

        for row in data:
            v = Victim(f"Victim_{row['Victim ID']}")
            for loc in row["Locations"]:
                l = Location(loc.replace(" ", "_"))
                v.passedThrough.append(l)
            for p in row["Names"]:
                perp = Perpetrator(p.replace(" ", "_"))
                v.interactedWith.append(perp)

    return onto

def export_ontology(onto, file_path="ontology_output.owl"):
    onto.save(file=file_path, format="rdfxml")
    return os.path.abspath(file_path)
