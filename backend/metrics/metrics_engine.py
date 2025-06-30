import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

def compute_basic_metrics(df):
    total_victims = df["Unique ID"].nunique()
    nationality_count = df["Nationality of Victim"].value_counts().to_dict()
    gender_dist = df["Gender of Victim"].value_counts().to_dict()
    years = df["Left Home Country Year"].dropna().astype(int)
    return total_victims, nationality_count, gender_dist, years

def plot_year_histogram(years):
    plt.figure(figsize=(8, 4))
    years.hist(bins=10)
    plt.title("Victims Leaving Home Country - Year Distribution")
    plt.xlabel("Year")
    plt.ylabel("Count")

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    return buffer
