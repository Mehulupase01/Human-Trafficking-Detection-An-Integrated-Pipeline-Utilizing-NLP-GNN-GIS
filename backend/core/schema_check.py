import pandas as pd

EXPECTED_COLUMNS = [
    "Unique ID", "Interviewer Name", "Date of Interview",
    "Gender of Victim", "Nationality of Victim", "Left Home Country Year",
    "Borders Crossed", "City / Locations Crossed", "Final Location",
    "Name of the Perpetrators involved", "Hierarchy of Perpetrators",
    "Human traffickers/ Chief of places", "Time Spent in Location / Cities / Places"
]

def validate_schema(df):
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"
    return True, "Schema is valid."