import pandas as pd
from collections import defaultdict

def build_victim_trafficker_map(df):
    vt_map = defaultdict(list)
    tv_map = defaultdict(list)

    for _, row in df.iterrows():
        vid = str(row["Unique ID"])
        raw_names = str(row["Name of the Perpetrators involved"])
        names = [n.strip() for n in raw_names.split(",") if n.strip()]
        for name in names:
            vt_map[vid].append(name)
            tv_map[name].append(vid)

    return vt_map, tv_map

def get_traffickers_by_victim(vt_map, victim_id):
    return vt_map.get(victim_id, [])

def get_victims_by_trafficker(tv_map, trafficker_name):
    return tv_map.get(trafficker_name, [])

def build_victim_trajectory(df):
    trajectories = {}
    for _, row in df.iterrows():
        vid = str(row["Unique ID"])
        locations = [l.strip() for l in str(row["City / Locations Crossed"]).split(",") if l.strip()]
        trajectories[vid] = locations
    return trajectories

def get_countries_crossed(df):
    crosses = {}
    for _, row in df.iterrows():
        vid = str(row["Unique ID"])
        countries = [c.strip() for c in str(row["Borders Crossed"]).split(",") if c.strip()]
        crosses[vid] = countries
    return crosses

def get_origin_and_destination(df):
    origins = {}
    destinations = {}
    for _, row in df.iterrows():
        vid = str(row["Unique ID"])
        origins[vid] = row.get("Nationality of Victim", "")
        destinations[vid] = row.get("Final Location", "")
    return origins, destinations

