import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

class TimePredictor:
    def __init__(self, data_path=None):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.encoders = {}
        self.is_trained = False
        
        if data_path is None:
            # Default path to enhanced dataset
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_path = os.path.join(current_dir, '..', '..', 'data', 'enhanced_delivery_dataset.csv')
            # Fallback to legacy dataset if enhanced doesn't exist
            if not os.path.exists(self.data_path):
                self.data_path = os.path.join(current_dir, '..', '..', 'data', 'cuisine_time_dataset.csv')
        else:
            self.data_path = data_path

    def train(self):
        """Train the model on the dataset"""
        try:
            if not os.path.exists(self.data_path):
                print(f"Data file not found at {self.data_path}")
                return False

            df = pd.read_csv(self.data_path)
            
            # Check if we are using the enhanced dataset or legacy
            is_enhanced = 'Weather' in df.columns
            
            if not is_enhanced:
                print("Using legacy dataset. Generating synthetic features...")
                # Legacy training logic (synthetic expansion)
                expanded_data = []
                for _, row in df.iterrows():
                    for _ in range(50):
                        distance = np.random.uniform(1, 15)
                        traffic_factor = np.random.uniform(1.0, 2.0)
                        travel_time = distance * 2 * traffic_factor
                        total_time = row['BasePrepTime'] + travel_time
                        
                        expanded_data.append({
                            'Cuisine': row['Cuisine'],
                            'BasePrepTime': row['BasePrepTime'],
                            'Distance': distance,
                            'TrafficFactor': traffic_factor,
                            'TotalTime': total_time
                        })
                df = pd.DataFrame(expanded_data)
                
                # Encode only Cuisine
                le = LabelEncoder()
                df['Cuisine_Encoded'] = le.fit_transform(df['Cuisine'])
                self.encoders['Cuisine'] = le
                
                X = df[['Cuisine_Encoded', 'BasePrepTime', 'Distance', 'TrafficFactor']]
                y = df['TotalTime']
                
            else:
                print("Using enhanced dataset with rich features...")
                # Enhanced training logic
                categorical_cols = ['Cuisine', 'Weather', 'TrafficLevel', 'TimeOfDay']
                
                for col in categorical_cols:
                    le = LabelEncoder()
                    df[f'{col}_Encoded'] = le.fit_transform(df[col])
                    self.encoders[col] = le
                
                # Map TrafficLevel to numeric factor for fallback logic if needed, but model learns it
                # Features: Cuisine, PrepTime, Distance, Weather, Traffic, TimeOfDay, DriverRating
                feature_cols = ['Cuisine_Encoded', 'BasePrepTime', 'Distance', 
                                'Weather_Encoded', 'TrafficLevel_Encoded', 
                                'TimeOfDay_Encoded', 'DriverRating']
                
                X = df[feature_cols]
                y = df['DeliveryTime']

            self.model.fit(X, y)
            self.is_trained = True
            print(f"Time Prediction Model trained successfully on {len(df)} records.")
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False

    def predict(self, cuisine, base_prep_time, distance, 
                traffic_level='Medium', weather='Clear', time_of_day='Dinner', driver_rating=5.0):
        """Predict delivery time with enhanced features"""
        
        # Fallback calculation
        # Map text levels to factors
        traffic_map = {'Low': 1.0, 'Medium': 1.3, 'High': 1.8, 'Jam': 2.5}
        weather_map = {'Clear': 1.0, 'Rain': 1.2, 'Fog': 1.3, 'Snow': 1.5}
        
        traffic_factor = traffic_map.get(traffic_level, 1.3)
        weather_factor = weather_map.get(weather, 1.0)
        
        if not self.is_trained:
            return base_prep_time + (distance * 2 * traffic_factor * weather_factor)
            
        try:
            # Check if we have the right encoders (enhanced model)
            if 'Weather' in self.encoders:
                # Encode inputs
                def safe_encode(col, val):
                    try:
                        return self.encoders[col].transform([val])[0]
                    except:
                        return 0 # Default to first class if unknown
                
                X_pred = pd.DataFrame([[
                    safe_encode('Cuisine', cuisine),
                    base_prep_time,
                    distance,
                    safe_encode('Weather', weather),
                    safe_encode('TrafficLevel', traffic_level),
                    safe_encode('TimeOfDay', time_of_day),
                    driver_rating
                ]], columns=['Cuisine_Encoded', 'BasePrepTime', 'Distance', 
                             'Weather_Encoded', 'TrafficLevel_Encoded', 
                             'TimeOfDay_Encoded', 'DriverRating'])
                
                prediction = self.model.predict(X_pred)[0]
                return max(prediction, base_prep_time)
                
            else:
                # Legacy model prediction
                try:
                    cuisine_encoded = self.encoders['Cuisine'].transform([cuisine])[0]
                except:
                    cuisine_encoded = 0
                    
                # Legacy model expects TrafficFactor (float), not Level (str)
                X_pred = pd.DataFrame([[cuisine_encoded, base_prep_time, distance, traffic_factor]], 
                                    columns=['Cuisine_Encoded', 'BasePrepTime', 'Distance', 'TrafficFactor'])
                
                prediction = self.model.predict(X_pred)[0]
                return max(prediction, base_prep_time)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return base_prep_time + (distance * 2 * traffic_factor * weather_factor)
