# backend/api/ontology.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import re

import pandas as pd
from rdflib import Graph, Namespace, URIRef, BNode, Literal
from rdflib.namespace import RDF, RDFS, XSD

from backend.core import dataset_registry as registry
from backend.api.graph_queries import concat_processed_frames
from backend.geo.geo_utils import resolve_locations_to_coords

# Expected processed columns
COL_SID = "Serialized ID"
COL_UID = "Unique ID"
COL_LOC = "Location"
COL_ROUTE = "Route_Order"
COL_GENDER = "Gender of Victim"
COL_NATION = "Nationality of Victim"
COL_PERPS = "Perpetrators (NLP)"
COL_CHIEFS = "Chiefs (NLP)"

def _slug(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"[^\w\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "unk"

def default_mapping(base_uri: str) -> Dict[str, Dict[str, str]]:
    """
    Returns default classes/predicates under the given base URI.
    Override any URI via the JSON mapping uploaded in the UI.
    """
    B = base_uri.rstrip("#/") + "#"
    return {
        "classes": {
            "Victim":     B + "Victim",
            "Trafficker": B + "Trafficker",
            "Chief":      B + "Chief",
            "Location":   B + "Location",
            "Visit":      B + "Visit",
        },
        "predicates": {
            "hasSerializedId": B + "hasSerializedId",
            "hasUniqueId":     B + "hasUniqueId",
            "hasGender":       B + "hasGender",
            "hasNationality":  B + "hasNationality",
            "hasTrafficker":   B + "hasTrafficker",
            "hasChief":        B + "hasChief",
            "hasVisit":        B + "hasVisit",      # Victim -> Visit
            "atLocation":      B + "atLocation",    # Visit -> Location
            "visitOrder":      B + "visitOrder",    # Visit -> xsd:int
            "label":           str(RDFS.label),
            "lat":             "http://www.w3.org/2003/01/geo/wgs84_pos#lat",
            "lon":             "http://www.w3.org/2003/01/geo/wgs84_pos#long",
        }
    }

def _get(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

def build_graph_from_processed(
    df: pd.DataFrame,
    base_uri: str = "https://example.org/htn#",
    mapping_override: Optional[Dict] = None,
    include_geo: bool = True,
) -> Tuple[Graph, Dict[str, int]]:
    """
    Build an RDF graph from standardized processed data.
    - Victims, Locations, Traffickers, Chiefs as resources
    - Visit node per (victim, hop) with order and atLocation
    - Optionally attach geo lat/lon to Locations
    """
    # Prepare mapping
    cfg = default_mapping(base_uri)
    if mapping_override:
        # deep merge (shallow keys only)
        for k in ("classes", "predicates"):
            if k in mapping_override and isinstance(mapping_override[k], dict):
                cfg[k].update(mapping_override[k])

    C = {k: URIRef(v) for k, v in cfg["classes"].items()}
    P = {k: URIRef(v) for k, v in cfg["predicates"].items()}

    g = Graph()
    BASE = Namespace(base_uri.rstrip("#/") + "#")
    GEO = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")

    # Declare classes (optional but nice)
    for cls in C.values():
        g.add((cls, RDF.type, RDFS.Class))

    # Pre-resolve coords if requested
    coords_map = {}
    if include_geo and _get(df, COL_LOC):
        unique_locs = sorted({str(x) for x in df[COL_LOC].dropna().astype(str).tolist()})
        coords_map = resolve_locations_to_coords(unique_locs)

    # Build entities
    victim_count = 0
    location_count = 0
    perp_count = 0
    chief_count = 0
    visit_count = 0

    # Caches for created resources
    VICT = {}
    LOCS = {}
    PERPS = {}
    CHIEFS = {}

    # Victim records (attributes)
    if not {COL_SID, COL_ROUTE, COL_LOC}.issubset(df.columns):
        raise ValueError("DataFrame missing required processed columns for ontology export.")

    # Iterate per victim to keep order logic tidy
    for sid, grp in df.groupby(COL_SID):
        vsid = str(sid)
        v_uri = URIRef(f"{BASE}Victim/{_slug(vsid)}")
        if v_uri not in VICT:
            VICT[v_uri] = True
            victim_count += 1
            g.add((v_uri, RDF.type, C["Victim"]))
            g.add((v_uri, P["hasSerializedId"], Literal(vsid)))
            if _get(df, COL_UID):
                uid = str(grp[COL_UID].iloc[0])
                g.add((v_uri, P["hasUniqueId"], Literal(uid)))
            if _get(df, COL_GENDER):
                gender = str(grp[COL_GENDER].mode(dropna=False).iloc[0])
                if gender and gender != "nan":
                    g.add((v_uri, P["hasGender"], Literal(gender)))
            if _get(df, COL_NATION):
                nat = str(grp[COL_NATION].mode(dropna=False).iloc[0])
                if nat and nat != "nan":
                    g.add((v_uri, P["hasNationality"], Literal(nat)))

        # Visits per hop
        g_sorted = grp.sort_values(COL_ROUTE, kind="stable")
        for _, row in g_sorted.iterrows():
            loc = str(row[COL_LOC]) if pd.notna(row[COL_LOC]) else ""
            if not loc:
                continue
            l_uri = URIRef(f"{BASE}Location/{_slug(loc)}")
            if l_uri not in LOCS:
                LOCS[l_uri] = True
                location_count += 1
                g.add((l_uri, RDF.type, C["Location"]))
                g.add((l_uri, P["label"], Literal(loc)))
                if include_geo and loc in coords_map:
                    (lat, lon) = coords_map[loc]
                    g.add((l_uri, P["lat"], Literal(float(lat), datatype=XSD.float)))
                    g.add((l_uri, P["lon"], Literal(float(lon), datatype=XSD.float)))

            order = int(row[COL_ROUTE]) if pd.notna(row[COL_ROUTE]) else None
            # Create a Visit node with victim, location, order
            visit_uri = URIRef(f"{BASE}Visit/{_slug(vsid)}-{order if order is not None else 'x'}")
            g.add((visit_uri, RDF.type, C["Visit"]))
            g.add((visit_uri, P["atLocation"], l_uri))
            if order is not None:
                g.add((visit_uri, P["visitOrder"], Literal(int(order), datatype=XSD.integer)))
            g.add((v_uri, P["hasVisit"], visit_uri))
            visit_count += 1

            # Perps & Chiefs per row
            if _get(df, COL_PERPS) and isinstance(row.get(COL_PERPS), list):
                for p in row[COL_PERPS]:
                    pname = str(p).strip()
                    if not pname:
                        continue
                    p_uri = URIRef(f"{BASE}Trafficker/{_slug(pname)}")
                    if p_uri not in PERPS:
                        PERPS[p_uri] = True
                        perp_count += 1
                        g.add((p_uri, RDF.type, C["Trafficker"]))
                        g.add((p_uri, P["label"], Literal(pname)))
                    g.add((v_uri, P["hasTrafficker"], p_uri))
            if _get(df, COL_CHIEFS) and isinstance(row.get(COL_CHIEFS), list):
                for c in row[COL_CHIEFS]:
                    cname = str(c).strip()
                    if not cname:
                        continue
                    c_uri = URIRef(f"{BASE}Chief/{_slug(cname)}")
                    if c_uri not in CHIEFS:
                        CHIEFS[c_uri] = True
                        chief_count += 1
                        g.add((c_uri, RDF.type, C["Chief"]))
                        g.add((c_uri, P["label"], Literal(cname)))
                    g.add((v_uri, P["hasChief"], c_uri))

    stats = {
        "victims": victim_count,
        "locations": location_count,
        "visits": visit_count,
        "traffickers": perp_count,
        "chiefs": chief_count,
    }
    return g, stats

def serialize_ttl(g: Graph) -> str:
    return g.serialize(format="turtle")

def union_with_user_ttl(g: Graph, user_ttl_str: str) -> Graph:
    if not user_ttl_str or not user_ttl_str.strip():
        return g
    ug = Graph()
    ug.parse(data=user_ttl_str, format="turtle")
    # Union: add all user triples into g
    for t in ug:
        g.add(t)
    return g

def build_and_export_ttl(
    dataset_ids: List[str],
    base_uri: str,
    mapping_override: Optional[Dict] = None,
    include_geo: bool = True,
    user_ttl_merge: Optional[str] = None,
) -> Tuple[str, Dict[str, int]]:
    """
    Load processed/merged datasets, build ontology graph, optionally merge with user TTL, return TTL string + stats.
    """
    df = concat_processed_frames(dataset_ids)
    g, stats = build_graph_from_processed(df, base_uri=base_uri, mapping_override=mapping_override, include_geo=include_geo)
    if user_ttl_merge:
        g = union_with_user_ttl(g, user_ttl_merge)
    ttl = serialize_ttl(g)
    return ttl, stats

def save_ttl_to_registry(name: str, ttl: str, owner: Optional[str], sources: List[str]) -> str:
    return registry.save_text(
        name=name,
        text=ttl,
        kind="ontology_ttl",
        owner=owner,
        source=",".join(sources),
        ext="ttl",
    )
