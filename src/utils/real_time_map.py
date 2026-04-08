import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import tkintermapview

class RealTimeMap:
    def __init__(self, parent_gui=None):
        self.parent_gui = parent_gui
        self.root = tk.Toplevel() if parent_gui else tk.Tk()
        self.root.title("Real-Time Map (OpenStreetMap)")
        self.root.geometry("1200x800")
        
        self.restaurant_marker = None
        self.customer_marker = None
        self.route_path = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Top Control Panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Search
        ttk.Label(control_frame, text="Search Location:").pack(side=tk.LEFT, padx=5)
        self.search_entry = ttk.Entry(control_frame, width=40)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        self.search_entry.bind("<Return>", self.search_location)
        ttk.Button(control_frame, text="Search", command=self.search_location).pack(side=tk.LEFT, padx=5)
        
        # Action Buttons
        ttk.Button(control_frame, text="Use for Delivery", 
                  command=self.export_data).pack(side=tk.RIGHT, padx=5)
        ttk.Button(control_frame, text="Clear Markers", 
                  command=self.clear_markers).pack(side=tk.RIGHT, padx=5)
        
        # Status Info
        self.info_label = ttk.Label(control_frame, text="Right-click to set locations", font=("Arial", 10, "bold"))
        self.info_label.pack(side=tk.LEFT, padx=20)
        
        # Map Widget
        self.map_widget = tkintermapview.TkinterMapView(self.root, width=1200, height=800, corner_radius=0)
        self.map_widget.pack(fill=tk.BOTH, expand=True)
        
        # Set default location (e.g., New York)
        self.map_widget.set_address("New York, USA")
        self.map_widget.set_zoom(12)
        
        # Right-click menu
        self.map_widget.add_right_click_menu_command(label="Set Restaurant Here", 
                                                    command=self.set_restaurant, pass_coords=True)
        self.map_widget.add_right_click_menu_command(label="Set Customer Here", 
                                                    command=self.set_customer, pass_coords=True)

    def search_location(self, event=None):
        address = self.search_entry.get()
        if address:
            self.map_widget.set_address(address)
            
    def set_restaurant(self, coords):
        if self.restaurant_marker:
            self.restaurant_marker.delete()
            
        self.restaurant_marker = self.map_widget.set_marker(coords[0], coords[1], text="Restaurant", marker_color_circle="blue", marker_color_outside="blue")
        self.calculate_route()
        
    def set_customer(self, coords):
        if self.customer_marker:
            self.customer_marker.delete()
            
        self.customer_marker = self.map_widget.set_marker(coords[0], coords[1], text="Customer", marker_color_circle="green", marker_color_outside="green")
        self.calculate_route()
        
    def calculate_route(self):
        if self.restaurant_marker and self.customer_marker:
            if self.route_path:
                self.route_path.delete()
                
            # Calculate path
            self.route_path = self.map_widget.set_path([self.restaurant_marker.position, self.customer_marker.position])
            
            # Get distance (tkintermapview doesn't give distance directly in set_path return in all versions, 
            # but we can calculate it or use OSRM if needed. For now, let's use geopy or simple haversine if needed, 
            # but set_path usually draws it. Let's assume straight line or try to get info)
            
            # Actually, set_path uses OSRM by default if not specified.
            # We can get the distance if we use the router explicitly or check the path object if supported.
            # For simplicity in this version, let's use the built-in OSRM router to get distance.
            
            # Note: TkinterMapView's set_path returns a path object. 
            # To get distance, we might need to use geopy distance for straight line or trust the visual.
            # Let's use geopy for accurate distance calculation between points for the app logic.
            
            # Get distance
            try:
                # Use OSRM API to get route geometry and distance
                import requests
                import json
                
                # OSRM expects lon,lat
                start_coords = f"{self.restaurant_marker.position[1]},{self.restaurant_marker.position[0]}"
                end_coords = f"{self.customer_marker.position[1]},{self.customer_marker.position[0]}"
                
                url = f"http://router.project-osrm.org/route/v1/driving/{start_coords};{end_coords}?overview=full&geometries=geojson"
                response = requests.get(url)
                data = response.json()
                
                if data['code'] == 'Ok':
                    route = data['routes'][0]
                    dist = route['distance'] / 1000.0 # Convert meters to km
                    duration = route['duration'] / 60.0 # Convert seconds to minutes
                    geometry = route['geometry']['coordinates'] # List of [lon, lat]
                    
                    # Store route geometry (convert to lat, lon for tkintermapview)
                    self.route_geometry = [[lat, lon] for lon, lat in geometry]
                    
                    # Update path on map with actual geometry
                    if self.route_path:
                        self.route_path.delete()
                    self.route_path = self.map_widget.set_path(self.route_geometry)
                    
                else:
                    raise Exception("OSM Route not found")

            except Exception as e:
                print(f"Routing error: {e}")
                # Fallback: Simple Euclidean approximation
                from geopy.distance import geodesic
                dist = geodesic(self.restaurant_marker.position, self.customer_marker.position).km
                duration = (dist / 30) * 60
                self.route_geometry = [self.restaurant_marker.position, self.customer_marker.position]
            
            self.current_distance = dist
            self.current_time = duration
            
            self.info_label.config(text=f"Distance: {dist:.2f} km | Est. Time: {duration:.0f} min")
            
    def clear_markers(self):
        self.map_widget.delete_all_marker()
        self.map_widget.delete_all_path()
        self.restaurant_marker = None
        self.customer_marker = None
        self.route_path = None
        self.route_geometry = None
        self.info_label.config(text="Right-click to set locations")
        
    def export_data(self):
        if not self.restaurant_marker or not self.customer_marker:
            messagebox.showwarning("Missing Points", "Please set both Restaurant and Customer locations.")
            return
            
        if self.parent_gui:
            # Use dedicated import method if available
            if hasattr(self.parent_gui, 'import_osm_route'):
                # Pass geometry if available
                geometry = getattr(self, 'route_geometry', [self.restaurant_marker.position, self.customer_marker.position])
                self.parent_gui.import_osm_route(self.current_distance, self.current_time, geometry)
                messagebox.showinfo("Success", "Route imported into main application!\nCheck the Map and Smart Delivery tabs.")
            else:
                # Fallback for older version
                if hasattr(self.parent_gui, 'distance_var'):
                    try:
                        self.parent_gui.distance_var.set(float(self.current_distance))
                    except:
                        pass
                if hasattr(self.parent_gui, 'log_update'):
                    self.parent_gui.log_update(f"Imported real-time route: {self.current_distance:.2f} km")
                messagebox.showinfo("Success", "Distance data exported.")
            
def create_real_time_map(parent_gui=None):
    app = RealTimeMap(parent_gui)
    if not parent_gui:
        app.root.mainloop()
    return app
