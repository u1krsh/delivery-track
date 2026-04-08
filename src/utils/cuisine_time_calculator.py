import pandas as pd
import os
from typing import Dict, List, Tuple, Optional

class CuisineTimeCalculator:
    """
    A module to calculate delivery time based on cuisine type and dish preparation time.
    Uses the cuisine_time_dataset.csv to get dish-specific time multipliers.
    """
    
    def __init__(self, csv_path: str = None):
        if csv_path is None:
            # Default path relative to the src directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(current_dir, '..', '..', 'data', 'cuisine_time_dataset.csv')
        
        self.csv_path = csv_path
        self.cuisine_data = None
        self.load_cuisine_data()
    
    def load_cuisine_data(self) -> bool:
        """Load cuisine data from CSV file"""
        try:
            if os.path.exists(self.csv_path):
                self.cuisine_data = pd.read_csv(self.csv_path)
                return True
            else:
                print(f"Warning: Cuisine data file not found at {self.csv_path}")
                # Create empty DataFrame with expected columns
                self.cuisine_data = pd.DataFrame(columns=['Dish', 'Cuisine', 'BasePrepTime', 'TimeMultiplier', 'Category'])
                return False
        except Exception as e:
            print(f"Error loading cuisine data: {e}")
            self.cuisine_data = pd.DataFrame(columns=['Dish', 'Cuisine', 'BasePrepTime', 'TimeMultiplier', 'Category'])
            return False
    
    def get_all_dishes(self) -> List[str]:
        """Get list of all available dishes"""
        if self.cuisine_data is not None and not self.cuisine_data.empty:
            return sorted(self.cuisine_data['Dish'].tolist())
        return []
    
    def get_cuisines(self) -> List[str]:
        """Get list of all available cuisines"""
        if self.cuisine_data is not None and not self.cuisine_data.empty:
            return sorted(self.cuisine_data['Cuisine'].unique().tolist())
        return []
    
    def get_dishes_by_cuisine(self, cuisine: str) -> List[str]:
        """Get dishes filtered by cuisine type"""
        if self.cuisine_data is not None and not self.cuisine_data.empty:
            filtered_data = self.cuisine_data[self.cuisine_data['Cuisine'] == cuisine]
            return sorted(filtered_data['Dish'].tolist())
        return []
    
    def get_dish_info(self, dish_name: str) -> Optional[Dict]:
        """Get detailed information about a specific dish"""
        if self.cuisine_data is not None and not self.cuisine_data.empty:
            dish_data = self.cuisine_data[self.cuisine_data['Dish'] == dish_name]
            if not dish_data.empty:
                row = dish_data.iloc[0]
                return {
                    'dish': row['Dish'],
                    'cuisine': row['Cuisine'],
                    'base_prep_time': row['BasePrepTime'],
                    'time_multiplier': row['TimeMultiplier'],
                    'category': row['Category']
                }
        return None
    
    def calculate_adjusted_delivery_time(self, base_travel_time: float, dish_name: str) -> Tuple[float, Dict]:
        """
        Calculate total delivery time including dish preparation and travel time.
        
        Args:
            base_travel_time: Time to travel the route (in minutes)
            dish_name: Name of the dish being delivered
            
        Returns:
            Tuple of (total_time, calculation_details)
        """
        dish_info = self.get_dish_info(dish_name)
        
        if dish_info is None:
            # If dish not found, return base travel time with default multiplier
            return base_travel_time, {
                'dish': dish_name,
                'cuisine': 'Unknown',
                'base_prep_time': 0,
                'time_multiplier': 1.0,
                'prep_time': 0,
                'travel_time': base_travel_time,
                'total_time': base_travel_time,
                'found': False
            }
        
        # Calculate preparation time
        prep_time = dish_info['base_prep_time']
        
        # Apply time multiplier to travel time (accounts for delivery complexity, traffic, etc.)
        adjusted_travel_time = base_travel_time * dish_info['time_multiplier']
        
        # Total time = preparation time + adjusted travel time
        total_time = prep_time + adjusted_travel_time
        
        calculation_details = {
            'dish': dish_info['dish'],
            'cuisine': dish_info['cuisine'],
            'category': dish_info['category'],
            'base_prep_time': prep_time,
            'time_multiplier': dish_info['time_multiplier'],
            'prep_time': prep_time,
            'base_travel_time': base_travel_time,
            'adjusted_travel_time': adjusted_travel_time,
            'total_time': total_time,
            'found': True
        }
        
        return total_time, calculation_details
    
    def get_delivery_time_breakdown(self, route_path: List[str], route_weights: List[float], dish_name: str) -> Dict:
        """
        Get detailed breakdown of delivery time for a complete route.
        
        Args:
            route_path: List of nodes in the route
            route_weights: List of travel times between consecutive nodes
            dish_name: Name of the dish being delivered
            
        Returns:
            Dictionary with detailed time breakdown
        """
        if not route_weights:
            return {}
        
        base_travel_time = sum(route_weights)
        total_time, details = self.calculate_adjusted_delivery_time(base_travel_time, dish_name)
        
        # Add route-specific information
        details['route_path'] = route_path
        details['route_segments'] = route_weights
        details['segment_count'] = len(route_weights)
        
        return details
    
    def search_dishes(self, query: str) -> List[Dict]:
        """
        Search for dishes containing the query string.
        
        Args:
            query: Search term
            
        Returns:
            List of matching dish information dictionaries
        """
        if self.cuisine_data is None or self.cuisine_data.empty:
            return []
        
        query_lower = query.lower()
        matching_dishes = []
        
        for _, row in self.cuisine_data.iterrows():
            if (query_lower in row['Dish'].lower() or 
                query_lower in row['Cuisine'].lower() or
                query_lower in row['Category'].lower()):
                
                matching_dishes.append({
                    'dish': row['Dish'],
                    'cuisine': row['Cuisine'],
                    'base_prep_time': row['BasePrepTime'],
                    'time_multiplier': row['TimeMultiplier'],
                    'category': row['Category']
                })
        
        return matching_dishes
    
    def get_cuisine_statistics(self) -> Dict:
        """Get statistics about the cuisine dataset"""
        if self.cuisine_data is None or self.cuisine_data.empty:
            return {}
        
        stats = {
            'total_dishes': len(self.cuisine_data),
            'total_cuisines': len(self.cuisine_data['Cuisine'].unique()),
            'categories': self.cuisine_data['Category'].value_counts().to_dict(),
            'avg_prep_time': self.cuisine_data['BasePrepTime'].mean(),
            'avg_multiplier': self.cuisine_data['TimeMultiplier'].mean(),
            'prep_time_range': {
                'min': self.cuisine_data['BasePrepTime'].min(),
                'max': self.cuisine_data['BasePrepTime'].max()
            },
            'multiplier_range': {
                'min': self.cuisine_data['TimeMultiplier'].min(),
                'max': self.cuisine_data['TimeMultiplier'].max()
            }
        }
        
        return stats

# Convenience function for easy import
def create_cuisine_calculator(csv_path: str = None) -> CuisineTimeCalculator:
    """Create and return a CuisineTimeCalculator instance"""
    return CuisineTimeCalculator(csv_path)