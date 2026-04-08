# filepath: delivery-tracker/delivery-tracker/src/main.py

from services.map_service import MapService
from services.tracking_service import TrackingService

def main():
    map_service = MapService()
    tracking_service = TrackingService()

    # Load map data and create graph
    graph = map_service.load_map("data/maps/sample_map.json")
    
    # Start tracking delivery drivers
    tracking_service.start_tracking(graph)

if __name__ == "__main__":
    main()