from backend.api.upload import process_upload
from io import StringIO
import pandas as pd

def test_valid_csv_schema():
    csv = """Sr. no.,Unique ID (Victim),Interviewer Name,Date of Interview,Gender of Victim,Nationality of Victim,Left Home Country Year,Borders Crossed,City / Locations Crossed,Final Location,Name of the Perpetrators involved,Hierarchy of Perpetrators,Human traffickers/ Chief of places,Time Spent in Location / Cities / Places
    1,101,John Doe,2020-01-01,Female,Eritrea,2017,Border1,"Tripoli, Sabha",Libya,"Ahmed, Yusef",Top-Mid,"Chief A",1 year"""
    file = StringIO(csv)
    file.name = "test.csv"
    df, msg = process_upload(file)
    assert df is not None
    assert "Sr. no." in df.columns