import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

def predict_time_location(df: pd.DataFrame):
    if "Left Home Country Year" not in df or "City / Locations Crossed" not in df:
        return "Required columns missing."

    df = df.dropna(subset=["Left Home Country Year", "City / Locations Crossed"])

    # Encode locations
    le = LabelEncoder()
    df["Location_Code"] = le.fit_transform(df["City / Locations Crossed"].astype(str))

    # Train model to predict year based on location code
    X = df[["Location_Code"]]
    y = df["Left Home Country Year"].astype(int)
    model = LinearRegression().fit(X, y)

    # Now predict: given a location, what year would it be trafficked to?
    unique_locs = df["City / Locations Crossed"].unique()
    loc_df = pd.DataFrame({"Location": unique_locs})
    loc_df["Location_Code"] = le.transform(loc_df["Location"])
    loc_df["Predicted_Year"] = model.predict(loc_df[["Location_Code"]]).round().astype(int)

    return loc_df.sort_values("Predicted_Year")