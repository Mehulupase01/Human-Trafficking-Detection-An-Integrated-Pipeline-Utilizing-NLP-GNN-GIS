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
            try:
                v = Victim(f"Victim_{row['Unique ID']}")

                # Handle locations
                locations_raw = row.get("City / Locations Crossed", "")
                if isinstance(locations_raw, str):
                    for loc in locations_raw.split(","):
                        loc = loc.strip()
                        if loc:
                            l = Location(loc.replace(" ", "_"))
                            v.passedThrough.append(l)

                # Handle perpetrators
                perps_raw = row.get("Name of the Perpetrators involved", "")
                if isinstance(perps_raw, str):
                    for perp in perps_raw.replace(" and ", ",").split(","):
                        perp = perp.strip()
                        if perp and perp.lower() != "no":
                            p = Perpetrator(perp.replace(" ", "_"))
                            v.interactedWith.append(p)

            except Exception as e:
                print(f"[Ontology Warning] Skipping row due to error: {e}")

    return onto

def export_ontology(onto, file_path="ontology_output.owl"):
    onto.save(file=file_path, format="rdfxml")
    return os.path.abspath(file_path)
