import pandas as pd

def apply_filters(df, nationality=None, gender=None, year_range=(2000, 2025), location_match=None):
    df_filtered = df.copy()

    if nationality:
        df_filtered = df_filtered[df_filtered["Nationality of Victim"] == nationality]

    if gender:
        df_filtered = df_filtered[df_filtered["Gender of Victim"] == gender]

    df_filtered = df_filtered[
        (df_filtered["Left Home Country Year"] >= year_range[0]) &
        (df_filtered["Left Home Country Year"] <= year_range[1])
    ]

    if location_match:
        df_filtered = df_filtered[df_filtered["City / Locations Crossed"].str.contains(location_match, case=False, na=False)]

    return df_filtered
