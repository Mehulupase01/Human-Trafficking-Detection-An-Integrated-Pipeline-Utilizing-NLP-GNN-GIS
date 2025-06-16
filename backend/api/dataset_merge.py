import pandas as pd
from functools import reduce

def merge_datasets(file_list):
    all_dfs = []
    for file in file_list:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        all_dfs.append(df)

    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df = merged_df.sort_values(by="Left Home Country Year", ascending=True)
    return merged_df