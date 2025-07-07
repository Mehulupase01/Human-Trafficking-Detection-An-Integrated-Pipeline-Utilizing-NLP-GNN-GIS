from backend.ontology.generate_ontology import build_ontology, export_ontology

def create_and_export_rdf(structured_data, filename="ontology_output.owl"):
    try:
        ontology = build_ontology(structured_data)
        path = export_ontology(ontology, file_path=filename)
        return path
    except Exception as e:
        return f"Ontology generation failed: {str(e)}"
