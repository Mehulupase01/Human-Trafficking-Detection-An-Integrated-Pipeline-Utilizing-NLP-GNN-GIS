import folium
from geopy.geocoders import Nominatim
import time
import os

def get_coordinates(location):
    geolocator = Nominatim(user_agent="geoapi")
    try:
        loc = geolocator.geocode(location)
        time.sleep(1)
        if loc:
            return (loc.latitude, loc.longitude)
    except:
        return None

def generate_map(data, output_file="gis_map.html"):
    m = folium.Map(location=[0.1, 37.0], zoom_start=2)
    for entry in data:
        victim_id = entry['Victim ID']
        locations = entry['Locations']
        coords_list = []
        for loc in locations:
            coords = get_coordinates(loc)
            if coords:
                coords_list.append(coords)
                folium.Marker(location=coords, popup=f"{loc} ({victim_id})").add_to(m)
        if len(coords_list) > 1:
            folium.PolyLine(coords_list, color="blue", weight=2.5, opacity=0.7).add_to(m)
    m.save(output_file)
    return os.path.abspath(output_file)
