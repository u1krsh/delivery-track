import requests

class GoogleMapsAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api"

    def get_map_data(self, location):
        url = f"{self.base_url}/geocode/json?address={location}&key={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception("Error fetching map data")

    def calculate_travel_time(self, origin, destination):
        url = f"{self.base_url}/directions/json?origin={origin}&destination={destination}&key={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            directions = response.json()
            if directions['routes']:
                travel_time = directions['routes'][0]['legs'][0]['duration']['value']
                return travel_time  # travel time in seconds
            else:
                raise Exception("No routes found")
        else:
            raise Exception("Error fetching travel time")