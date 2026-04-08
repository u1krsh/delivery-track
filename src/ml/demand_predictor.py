import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class DemandPredictor:
    def __init__(self):
        self.model = KMeans(n_clusters=3, random_state=42)
        self.history = []
        
    def generate_synthetic_history(self, nodes):
        """Generate fake order history for demonstration"""
        if not nodes:
            return
            
        # Create hotspots around random nodes
        node_ids = list(nodes.keys())
        hotspots = np.random.choice(node_ids, size=3, replace=False)
        
        history_data = []
        for _ in range(100):
            # 70% chance to be near a hotspot
            if np.random.random() < 0.7:
                center = nodes[np.random.choice(hotspots)]
            else:
                center = nodes[np.random.choice(node_ids)]
                
            # Add gaussian noise
            x = center['x'] + np.random.normal(0, 50)
            y = center['y'] + np.random.normal(0, 50)
            
            history_data.append([x, y])
            
        self.history = np.array(history_data)
        
    def predict_hotspots(self):
        """Return centers of high demand areas"""
        if len(self.history) < 10:
            return []
            
        self.model.fit(self.history)
        return self.model.cluster_centers_
