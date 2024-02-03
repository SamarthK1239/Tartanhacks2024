import googlemaps
import os
from dotenv import load_dotenv
from pathlib import Path


def get_gmaps_client():
    path = Path("Env/.env")
    load_dotenv(dotenv_path=path)
    return googlemaps.Client(key=os.getenv('GMAPS_API_KEY'))


