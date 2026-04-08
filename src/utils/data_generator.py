import pandas as pd
import numpy as np
import os

def generate_enhanced_dataset(output_path='data/enhanced_delivery_dataset.csv', n_samples=2000):
    """
    Generates a rich synthetic dataset for delivery time prediction.
    """
    
    # Load base dish data
    base_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cuisine_time_dataset.csv')
    if not os.path.exists(base_data_path):
        print(f"Error: Base data not found at {base_data_path}")
        return

    df_base = pd.read_csv(base_data_path)
    
    data = []
    
    weather_conditions = ['Clear', 'Rain', 'Fog', 'Snow']
    traffic_levels = ['Low', 'Medium', 'High', 'Jam']
    times_of_day = ['Morning', 'Lunch', 'Afternoon', 'Dinner', 'Late']
    
    print(f"Generating {n_samples} synthetic delivery records...")
    
    for _ in range(n_samples):
        # Pick a random dish
        dish_row = df_base.sample(1).iloc[0]
        
        # 1. Basic Features
        cuisine = dish_row['Cuisine']
        prep_time = dish_row['BasePrepTime']
        
        # 2. Environmental Features
        distance = np.random.uniform(0.5, 20.0) # 0.5 to 20 km
        weather = np.random.choice(weather_conditions, p=[0.6, 0.2, 0.1, 0.1])
        traffic = np.random.choice(traffic_levels, p=[0.3, 0.4, 0.2, 0.1])
        time_of_day = np.random.choice(times_of_day, p=[0.1, 0.3, 0.2, 0.3, 0.1])
        driver_rating = np.random.uniform(3.0, 5.0)
        
        # 3. Calculate "Actual" Time (Ground Truth Generation)
        # Base travel speed: 2 mins per km (30km/h)
        base_travel_time = distance * 2.0 
        
        # Modifiers
        weather_factor = 1.0
        if weather == 'Rain': weather_factor = 1.2
        elif weather == 'Fog': weather_factor = 1.3
        elif weather == 'Snow': weather_factor = 1.5
        
        traffic_factor = 1.0
        if traffic == 'Medium': traffic_factor = 1.3
        elif traffic == 'High': traffic_factor = 1.8
        elif traffic == 'Jam': traffic_factor = 2.5
        
        # Driver skill impact (better rating = slightly faster)
        driver_factor = 1.0 - ((driver_rating - 3.0) * 0.05) # Max 10% reduction for 5.0 rating
        
        # Random noise (unexpected delays)
        noise = np.random.normal(0, 3) # +/- 3 mins standard deviation
        
        # Total Calculation
        travel_time = base_travel_time * weather_factor * traffic_factor * driver_factor
        total_time = prep_time + travel_time + noise
        
        # Ensure no negative times
        total_time = max(total_time, prep_time + 5)
        
        data.append({
            'Cuisine': cuisine,
            'BasePrepTime': prep_time,
            'Distance': round(distance, 2),
            'Weather': weather,
            'TrafficLevel': traffic,
            'TimeOfDay': time_of_day,
            'DriverRating': round(driver_rating, 1),
            'DeliveryTime': round(total_time, 1)
        })
        
    # Create DataFrame
    df_enhanced = pd.DataFrame(data)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_enhanced.to_csv(output_path, index=False)
    print(f"Saved enhanced dataset to {output_path}")
    print(df_enhanced.head())

if __name__ == "__main__":
    generate_enhanced_dataset()
