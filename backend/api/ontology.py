from backend.ontology.generate_ontology import build_ontology, export_ontology

def create_and_export_rdf(structured_data):
    onto = build_ontology(structured_data)
    file_path = export_ontology(onto)
    return file_path