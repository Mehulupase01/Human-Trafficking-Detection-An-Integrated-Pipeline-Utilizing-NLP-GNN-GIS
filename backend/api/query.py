import pandas as pd

def apply_filters(df, nationality=None, gender=None, year_range=None, location=None):
    filtered = df.copy()
    if nationality:
        filtered = filtered[filtered['Nationality of Victim'] == nationality]
    if gender:
        filtered = filtered[filtered['Gender of Victim'] == gender]
    if year_range:
        start, end = year_range
        filtered = filtered[(filtered['Left Home Country Year'] >= start) & (filtered['Left Home Country Year'] <= end)]
    if location:
        filtered = filtered[filtered['City / Locations Crossed'].str.contains(location, case=False, na=False)]
    return filtered