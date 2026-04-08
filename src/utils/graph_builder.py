from models.graph import Graph
from services.map_service import MapService

def build_graph_from_map(map_data):
    graph = Graph()
    for intersection in map_data['intersections']:
        graph.add_node(intersection['id'], intersection['location'])
    
    for road in map_data['roads']:
        start = road['start']
        end = road['end']
        travel_time = road['travel_time']
        graph.add_edge(start, end, travel_time)
    
    return graph

def load_map_data(file_path):
    map_service = MapService()
    return map_service.load_map(file_path)

def create_graph_from_file(file_path):
    map_data = load_map_data(file_path)
    return build_graph_from_map(map_data)