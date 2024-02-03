import googlemaps
import os
from dotenv import load_dotenv
from pathlib import Path
import json


def get_gmaps_client():
    path = Path("Env/.env")
    load_dotenv(dotenv_path=path)
    return googlemaps.Client(key=os.getenv('GMAPS_API_KEY'))


def get_distance_matrix(client, origins, destinations):
    return client.distance_matrix(origins, destinations, mode="driving", departure_time="now")


# def parse_json_dictionary():
#     individual_statements = {}
#     with open("sample.json") as json_file:
#         data = json.load(json_file)
#         i = 0
#         for origin in data['origin_addresses']:
#             j = 0
#             for destination in data['destination_addresses']:
#                 distance = data['rows'][i]['elements'][j]['distance']['text']
#                 duration = data['rows'][i]['elements'][j]['duration']['text']
#                 individual_statements[origin] = {"destination": destination,
#                                                  "params": {"distance": distance, "duration": duration}}
#                 j += 1
#             i += 1
#     return individual_statements

def parse_json_dictionary():
    individual_statements = {}
    addresses = []
    with open("sample.json") as json_file:
        data = json.load(json_file)
        addresses.append(data['origin_addresses'])
        i = 0
        for origin in data['origin_addresses']:
            j = 0
            for destination in data['destination_addresses']:
                distance = data['rows'][i]['elements'][j]['distance']['text']
                duration = data['rows'][i]['elements'][j]['duration']['text']
                individual_statements[f"{origin}_{destination}"] = {
                    "origin": origin,
                    "destination": destination,
                    "params": {"distance": distance, "duration": duration}
                }
                j += 1
            i += 1
    return [individual_statements, addresses]


def export_json(parsed_json_data):
    with open("sample.json", "w") as outfile:
        json.dump(parsed_json_data, outfile)


# def generate_graph():
#     dictionary = parse_json_dictionary()
#     vertices = []
#     keys = list(dictionary.keys())
#
#     for key in keys:
#         BuildingGraph.add_vertex(key)


def run():
    client = get_gmaps_client()
    origins = ["Pittsburgh PA", "State College PA", "Erie PA", "Harrisburg PA", "Philadelphia PA"]
    destinations = ["Pittsburgh PA", "State College PA", "Erie PA", "Harrisburg PA", "Philadelphia PA"]

    response = get_distance_matrix(client, origins, destinations)
    print(response)

    with open("sample.json", "w") as outfile:
        json.dump(response, outfile)


print(parse_json_dictionary())
