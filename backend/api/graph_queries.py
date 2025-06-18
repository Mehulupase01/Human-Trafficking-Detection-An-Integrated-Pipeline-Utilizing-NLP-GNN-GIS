import pandas as pd
from collections import defaultdict

def build_victim_trafficker_map(df: pd.DataFrame):
    vt_map = defaultdict(set)
    tv_map = defaultdict(set)

    for _, row in df.iterrows():
        victim_id = row.get("Unique ID")
        if pd.isna(victim_id):
            continue

        # Combine both perpetrator fields
        names = []
        if isinstance(row.get("Name of the Perpetrators involved"), str):
            names += [n.strip() for n in row["Name of the Perpetrators involved"].split("and")]
        if isinstance(row.get("Human traffickers/ Chief of places"), str):
            names += [n.strip() for n in row["Human traffickers/ Chief of places"].split("and")]

        for name in names:
            if name:
                vt_map[victim_id].add(name)
                tv_map[name].add(victim_id)

    return vt_map, tv_map

def get_victims_by_trafficker(tv_map, trafficker_name):
    return list(tv_map.get(trafficker_name, []))

def get_traffickers_by_victim(vt_map, victim_id):
    return list(vt_map.get(victim_id, []))

def build_victim_trajectory(df: pd.DataFrame):
    trajectories = defaultdict(list)
    df_sorted = df.sort_values(by=["Unique ID", "Left Home Country Year"])

    for _, row in df_sorted.iterrows():
        vid = row["Unique ID"]
        loc = row.get("City / Locations Crossed")
        if pd.notna(vid) and pd.notna(loc):
            trajectories[vid].append(loc)

    return dict(trajectories)

def get_countries_crossed(df: pd.DataFrame):
    countries = defaultdict(set)
    for _, row in df.iterrows():
        vid = row.get("Unique ID")
        borders = row.get("Borders Crossed")
        if pd.notna(vid) and pd.notna(borders):
            for pair in borders.split("/"):
                countries[vid].add(pair.strip())
    return dict(countries)

def get_origin_and_destination(df: pd.DataFrame):
    origin = {}
    destination = {}
    for vid, group in df.groupby("Unique ID"):
        first = group.head(1)
        last = group.tail(1)
        origin[vid] = first["City / Locations Crossed"].values[0] if not first.empty else None
        destination[vid] = last["Final Location"].values[0] if not last.empty else None
    return origin, destination
