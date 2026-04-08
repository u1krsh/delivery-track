import math

class AssignmentService:
    def __init__(self, graph):
        self.graph = graph

    def calculate_distance(self, node1, node2):
        """Calculate Euclidean distance between two nodes"""
        if node1 not in self.graph.nodes or node2 not in self.graph.nodes:
            return float('inf')
            
        x1, y1 = self.graph.nodes[node1]['x'], self.graph.nodes[node1]['y']
        x2, y2 = self.graph.nodes[node2]['x'], self.graph.nodes[node2]['y']
        
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def score_driver(self, driver, pickup_location):
        """
        Calculate a score for a driver based on:
        - Distance to pickup (lower is better)
        - Current workload (lower is better)
        - Driver rating (higher is better)
        - Efficiency score (higher is better)
        """
        if not driver.is_available():
            return -1
            
        distance = self.calculate_distance(driver.current_location, pickup_location)
        
        # Normalize factors
        # Assume max distance on map ~ 1000
        norm_distance = min(distance / 1000, 1.0)
        
        # Workload (number of deliveries)
        workload = driver.get_workload()
        norm_workload = min(workload / 5, 1.0) # Assume max 5 deliveries
        
        # Rating (0-5)
        norm_rating = driver.rating / 5.0
        
        # Efficiency (0-100)
        norm_efficiency = driver.efficiency_score / 100.0
        
        # Scoring weights
        w_dist = 0.4
        w_work = 0.3
        w_rating = 0.2
        w_eff = 0.1
        
        # Score calculation (Higher is better)
        # We invert distance and workload since lower is better
        score = (w_dist * (1 - norm_distance) + 
                 w_work * (1 - norm_workload) + 
                 w_rating * norm_rating + 
                 w_eff * norm_efficiency)
                 
        return score

    def find_best_driver(self, delivery, drivers):
        """Find the best driver for a delivery"""
        pickup_location = delivery.destination # In this simple model, pickup is previous location or we just use dest
        # Actually, for food delivery, pickup is usually a restaurant. 
        # But the current app model seems to just have "current location" and "destination".
        # Let's assume the driver needs to go to the delivery destination (or pickup point?)
        # The current app logic: driver moves from current -> delivery destination.
        # So we calculate distance from driver to delivery destination.
        
        best_driver = None
        best_score = -1
        
        for driver_id, driver in drivers.items():
            score = self.score_driver(driver, delivery.destination)
            
            if score > best_score:
                best_score = score
                best_driver = driver
                
        return best_driver
