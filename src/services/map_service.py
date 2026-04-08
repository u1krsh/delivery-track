from typing import Dict, List
import json
import os

class MapService:
    def __init__(self, map_data_path: str):
        self.map_data_path = map_data_path
        self.graph = None

    def load_map(self) -> None:
        if not os.path.exists(self.map_data_path):
            raise FileNotFoundError(f"Map data file not found: {self.map_data_path}")
        
        with open(self.map_data_path, 'r') as file:
            map_data = json.load(file)
            self.graph = self.build_graph(map_data)

    def build_graph(self, map_data: Dict) -> 'Graph':
        from models.graph import Graph
        graph = Graph()

        for intersection in map_data['intersections']:
            graph.add_node(intersection['id'], intersection['location'])

        for road in map_data['roads']:
            graph.add_edge(road['start'], road['end'], road['travel_time'])

        return graph

    def get_graph(self) -> 'Graph':
        if self.graph is None:
            raise ValueError("Graph has not been built. Please load the map first.")
        return self.graph

    def save_map(self, map_data: Dict) -> None:
        with open(self.map_data_path, 'w') as file:
            json.dump(map_data, file)