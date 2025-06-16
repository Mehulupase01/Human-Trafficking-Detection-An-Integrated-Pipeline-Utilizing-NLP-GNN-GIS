import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO


def compute_basic_metrics(df):
    total_victims = df["Unique ID (Victim)"].nunique()
    nationality_count = df["Nationality of Victim"].value_counts().to_dict()
    gender_dist = df["Gender of Victim"].value_counts().to_dict()
    years = df["Left Home Country Year"].dropna().astype(int)
    return total_victims, nationality_count, gender_dist, years


def plot_year_histogram(years):
    fig, ax = plt.subplots()
    ax.hist(years, bins=10, color="#4c72b0", edgecolor="white")
    ax.set_title("Distribution of Year Left Home Country")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Victims")
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf