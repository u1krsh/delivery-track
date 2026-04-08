import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import time
from datetime import datetime
import sys
import os

# Add parent directory to path to import our models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Add project root for env/ and inference imports
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.delivery import Delivery
from models.driver import Driver
from models.graph import Graph
from algorithms.routing import Routing
from utils.image_map_creator import create_image_map
from utils.real_time_map import create_real_time_map
from utils.cuisine_time_calculator import CuisineTimeCalculator
from ml.time_predictor import TimePredictor
from ml.demand_predictor import DemandPredictor
from services.assignment_service import AssignmentService

# AI Agent imports
from inference import DeliveryAgent, DEFAULT_API_BASE, DEFAULT_MODEL, DEFAULT_HF_TOKEN
from env.tasks import get_task, TASK_IDS

class DeliveryTrackerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Delivery Tracker System")
        self.root.geometry("1200x800")
        
        # Initialize data structures
        self.graph = Graph()
        self.drivers = {}
        self.deliveries = {}
        self.routing = Routing(self.graph)
        self.cuisine_calculator = CuisineTimeCalculator()
        
        # Initialize ML components
        self.time_predictor = TimePredictor()
        self.time_predictor.train() # Train on startup
        self.demand_predictor = DemandPredictor()
        self.assignment_service = AssignmentService(self.graph)
        
        # Coordinate system variables
        self.show_coordinates = tk.BooleanVar(value=True)
        self.grid_size = 50
        self.canvas_click_enabled = tk.BooleanVar(value=False)
        self.pending_intersection_name = None
        self.distance_var = tk.DoubleVar(value=0.0) # For imported real-time distance
        
        # Create GUI elements
        self.create_widgets()
        self.create_sample_data()
        
    def create_widgets(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_map_tab()
        self.create_drivers_tab()
        self.create_deliveries_tab()
        self.create_routing_tab()
        self.create_tracking_tab()
        self.create_ai_agent_tab()
        
    def create_map_tab(self):
        # Map Management Tab
        self.map_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.map_frame, text="Map Management")
        
        # Map canvas with coordinate system
        canvas_frame = ttk.LabelFrame(self.map_frame, text="Map View")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Canvas with scrollbars
        canvas_container = ttk.Frame(canvas_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.map_canvas = tk.Canvas(canvas_container, bg='white', width=800, height=500,
                                   scrollregion=(0, 0, 1000, 800))
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.map_canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.map_canvas.xview)
        
        self.map_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack canvas and scrollbars
        self.map_canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)
        
        # Bind canvas events
        self.map_canvas.bind("<Button-1>", self.on_canvas_click)
        self.map_canvas.bind("<Motion>", self.on_canvas_motion)
        
        # Coordinate display
        coord_frame = ttk.Frame(canvas_frame)
        coord_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.coord_label = ttk.Label(coord_frame, text="Mouse: (0, 0)")
        self.coord_label.pack(side=tk.LEFT)
        
        ttk.Checkbutton(coord_frame, text="Show Grid", 
                       variable=self.show_coordinates, 
                       command=self.draw_graph).pack(side=tk.RIGHT, padx=5)
        
        # Map controls
        controls_frame = ttk.LabelFrame(self.map_frame, text="Map Controls")
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # First row of controls
        controls_row1 = ttk.Frame(controls_frame)
        controls_row1.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(controls_row1, text="Add Intersection", 
                  command=self.add_intersection).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_row1, text="Click to Add", 
                  command=self.toggle_click_mode).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_row1, text="Add Road", 
                  command=self.add_road).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_row1, text="Clear Map", 
                  command=self.clear_map).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_row1, text="Real Scale View", 
                  command=self.show_real_scale_view).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_row1, text="🔥 Show Hotspots", 
                  command=self.show_hotspots).pack(side=tk.LEFT, padx=5)
        
        # Second row of controls
        controls_row2 = ttk.Frame(controls_frame)
        controls_row2.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(controls_row2, text="Quick Add:").pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_row2, text="Grid 3x3", 
                  command=lambda: self.create_grid(3, 3)).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_row2, text="Grid 4x4", 
                  command=lambda: self.create_grid(4, 4)).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_row2, text="Linear 5", 
                  command=lambda: self.create_linear(5)).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_row2, text="Create from Image", 
                  command=self.open_image_map_creator).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_row2, text="Real-Time Map (OSM)", 
                  command=self.open_real_time_map).pack(side=tk.LEFT, padx=5)
        
        # Coordinate helper
        coord_helper_frame = ttk.LabelFrame(self.map_frame, text="Coordinate Helper")
        coord_helper_frame.pack(fill=tk.X, padx=10, pady=5)
        
        helper_row = ttk.Frame(coord_helper_frame)
        helper_row.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(helper_row, text="Common coordinates:").pack(side=tk.LEFT)
        
        common_coords = [
            ("Top-Left", 100, 100), ("Top-Center", 400, 100), ("Top-Right", 700, 100),
            ("Mid-Left", 100, 300), ("Center", 400, 300), ("Mid-Right", 700, 300),
            ("Bot-Left", 100, 500), ("Bot-Center", 400, 500), ("Bot-Right", 700, 500)
        ]
        
        for name, x, y in common_coords:
            btn = ttk.Button(helper_row, text=f"{name}\n({x},{y})", 
                           command=lambda x=x, y=y: self.quick_add_intersection(x, y))
            btn.pack(side=tk.LEFT, padx=2)
    
    def create_drivers_tab(self):
        # Drivers Tab
        self.drivers_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.drivers_frame, text="Drivers")
        
        # Drivers list
        list_frame = ttk.LabelFrame(self.drivers_frame, text="Active Drivers")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview for drivers
        columns = ("ID", "Name", "Status", "Current Location", "Deliveries")
        self.drivers_tree = ttk.Treeview(list_frame, columns=columns, show="headings")
        
        for col in columns:
            self.drivers_tree.heading(col, text=col)
            self.drivers_tree.column(col, width=150)
            
        self.drivers_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Driver controls
        controls_frame = ttk.Frame(self.drivers_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(controls_frame, text="Add Driver", 
                  command=self.add_driver).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Update Location", 
                  command=self.update_driver_location).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Assign Delivery", 
                  command=self.assign_delivery).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="🤖 Smart Assign", 
                  command=self.smart_assign_delivery).pack(side=tk.LEFT, padx=5)
        
    def create_deliveries_tab(self):
        # Deliveries Tab
        self.deliveries_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.deliveries_frame, text="Deliveries")
        
        # Deliveries list
        list_frame = ttk.LabelFrame(self.deliveries_frame, text="Delivery Orders")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview for deliveries
        columns = ("ID", "Destination", "Status", "Progress", "Assigned Driver", "Created")
        self.deliveries_tree = ttk.Treeview(list_frame, columns=columns, show="headings")
        
        for col in columns:
            self.deliveries_tree.heading(col, text=col)
            self.deliveries_tree.column(col, width=120)
            
        self.deliveries_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Delivery controls
        controls_frame = ttk.Frame(self.deliveries_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(controls_frame, text="New Delivery", 
                  command=self.create_delivery).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Update Status", 
                  command=self.update_delivery_status).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Update Progress", 
                  command=self.update_delivery_progress).pack(side=tk.LEFT, padx=5)
        
    def create_routing_tab(self):
        # Routing Tab with integrated cuisine features
        self.routing_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.routing_frame, text="Smart Route Planning")
        
        # Create main container with two columns
        main_container = ttk.Frame(self.routing_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left column for route and cuisine settings
        left_frame = ttk.Frame(main_container)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Right column for results and dish info
        right_frame = ttk.Frame(main_container)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # === LEFT COLUMN: ROUTE & CUISINE SETTINGS ===
        
        # Route Configuration Section
        route_config_frame = ttk.LabelFrame(left_frame, text="🗺️ Route Configuration")
        route_config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Location selection with enhanced layout
        locations_frame = ttk.Frame(route_config_frame)
        locations_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(locations_frame, text="From:", font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.start_var = tk.StringVar()
        self.start_combo = ttk.Combobox(locations_frame, textvariable=self.start_var, width=25)
        self.start_combo.grid(row=0, column=1, padx=5, pady=2, sticky='ew')
        
        ttk.Label(locations_frame, text="To:", font=('Arial', 9, 'bold')).grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.end_var = tk.StringVar()
        self.end_combo = ttk.Combobox(locations_frame, textvariable=self.end_var, width=25)
        self.end_combo.grid(row=1, column=1, padx=5, pady=2, sticky='ew')
        
        locations_frame.columnconfigure(1, weight=1)
        
        # Algorithm selection
        algorithm_frame = ttk.Frame(route_config_frame)
        algorithm_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Label(algorithm_frame, text="Algorithm:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        self.algorithm_var = tk.StringVar(value="A* Search")
        algorithm_combo = ttk.Combobox(algorithm_frame, textvariable=self.algorithm_var, 
                                     values=["BFS", "DFS", "Dijkstra", "A* Search"], width=15, state="readonly")
        algorithm_combo.pack(side=tk.LEFT, padx=5)
        
        # Basic route buttons
        basic_buttons_frame = ttk.Frame(route_config_frame)
        basic_buttons_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(basic_buttons_frame, text="🔍 Find Basic Route", 
                  command=self.find_route).pack(side=tk.LEFT, padx=5)
        ttk.Button(basic_buttons_frame, text="⚡ Optimize All", 
                  command=self.optimize_routes).pack(side=tk.LEFT, padx=5)
        ttk.Button(basic_buttons_frame, text="🏁 Compare Algorithms", 
                  command=self.compare_algorithms).pack(side=tk.LEFT, padx=5)
        
        # === CUISINE INTEGRATION SECTION ===
        cuisine_frame = ttk.LabelFrame(left_frame, text="🍽️ Smart Delivery Planning")
        cuisine_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Cuisine search with improved layout
        search_frame = ttk.Frame(cuisine_frame)
        search_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(search_frame, text="Find Dish:", font=('Arial', 9, 'bold')).pack(anchor='w')
        
        search_entry_frame = ttk.Frame(search_frame)
        search_entry_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.dish_search_var = tk.StringVar()
        self.dish_search_entry = ttk.Entry(search_entry_frame, textvariable=self.dish_search_var, 
                                          font=('Arial', 9), width=30)
        self.dish_search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.dish_search_entry.bind('<KeyRelease>', lambda event: self.search_dishes_simple())
        
        ttk.Button(search_entry_frame, text="🔎", width=3,
                  command=lambda: self.search_dishes_simple()).pack(side=tk.RIGHT)
        
        # Quick cuisine filters
        cuisine_filter_frame = ttk.Frame(search_frame)
        cuisine_filter_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(cuisine_filter_frame, text="Quick Filters:", font=('Arial', 8)).pack(anchor='w')
        filter_buttons_frame = ttk.Frame(cuisine_filter_frame)
        filter_buttons_frame.pack(fill=tk.X, pady=(2, 0))
        
        cuisines = ["Italian", "Asian", "American", "Mexican", "Dessert"]
        for cuisine in cuisines:
            ttk.Button(filter_buttons_frame, text=cuisine, width=8,
                      command=lambda c=cuisine: self.filter_by_cuisine(c)).pack(side=tk.LEFT, padx=2)
        
        # Dish selection
        selection_frame = ttk.Frame(cuisine_frame)
        selection_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Label(selection_frame, text="Selected Dish:", font=('Arial', 9, 'bold')).pack(anchor='w')
        
        dish_combo_frame = ttk.Frame(selection_frame)
        dish_combo_frame.pack(fill=tk.X, pady=(5, 10))
        
        self.selected_dish_var = tk.StringVar()
        self.dish_combo = ttk.Combobox(dish_combo_frame, textvariable=self.selected_dish_var, 
                                      font=('Arial', 9), width=30, state="readonly")
        self.dish_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.dish_combo.bind('<<ComboboxSelected>>', self.on_dish_selected_simple)
        
        # Populate dish combo with all dishes initially
        try:
            all_dishes = self.cuisine_calculator.get_all_dishes()
            self.dish_combo['values'] = all_dishes
        except Exception as e:
            print(f"Error loading dishes: {e}")
            self.dish_combo['values'] = ["Loading dishes..."]
        
        # Smart calculation button
        smart_calc_frame = ttk.Frame(cuisine_frame)
        smart_calc_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Distance display
        self.distance_display_var = tk.StringVar(value="Real-World Distance: Not Set")
        ttk.Label(smart_calc_frame, textvariable=self.distance_display_var, 
                 foreground="blue").pack(fill=tk.X, pady=(0, 5))
        
        # Trace distance_var to update display
        self.distance_var.trace_add("write", self.update_distance_display)
        
        self.smart_calc_btn = ttk.Button(smart_calc_frame, text="🧠 Calculate AI Delivery Time", 
                                        command=lambda: self.calculate_delivery_time_ai())
        self.smart_calc_btn.pack(fill=tk.X)
        
        # === RIGHT COLUMN: RESULTS & INFO ===
        
        # Dish Information Panel
        dish_info_frame = ttk.LabelFrame(right_frame, text="📋 Dish Information")
        dish_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.dish_info_text = tk.Text(dish_info_frame, height=8, wrap=tk.WORD, font=('Arial', 9))
        dish_info_scrollbar = ttk.Scrollbar(dish_info_frame, orient=tk.VERTICAL, command=self.dish_info_text.yview)
        self.dish_info_text.configure(yscrollcommand=dish_info_scrollbar.set)
        
        self.dish_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        dish_info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Initial dish info message
        self.dish_info_text.insert(tk.END, "🍽️ SMART DELIVERY PLANNING\n\n" +
                                           "• Search for dishes by name or cuisine\n" +
                                           "• Select a dish to see preparation details\n" +
                                           "• Get delivery time estimates that include:\n" +
                                           "  - Food preparation time\n" +
                                           "  - Route optimization\n" +
                                           "  - Cuisine-specific delivery factors\n\n" +
                                           "💡 Tip: Use quick filters or type to search!")
        self.dish_info_text.config(state=tk.DISABLED)
        dish_info_scrollbar = ttk.Scrollbar(dish_info_frame, orient=tk.VERTICAL, command=self.dish_info_text.yview)
        self.dish_info_text.configure(yscrollcommand=dish_info_scrollbar.set)
        
        self.dish_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        dish_info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Route Results and Analysis
        results_frame = ttk.LabelFrame(right_frame, text="📊 Route Analysis & Results")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.route_text = tk.Text(results_frame, height=15, wrap=tk.WORD, font=('Consolas', 9))
        route_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.route_text.yview)
        self.route_text.configure(yscrollcommand=route_scrollbar.set)
        
        self.route_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        route_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Initial welcome message in route results
        welcome_msg = """🚀 SMART ROUTE PLANNING SYSTEM
        
Ready to calculate optimized delivery routes!

FEATURES:
✓ Multiple pathfinding algorithms (BFS, DFS, Dijkstra, A*)
✓ Cuisine-aware delivery time estimation
✓ Real-time dish search and filtering
✓ Detailed route analysis with preparation times

INSTRUCTIONS:
1. Select start and end locations
2. Choose a dish for delivery
3. Click 'Calculate Smart Delivery Time' for full analysis
   OR use 'Find Basic Route' for simple pathfinding

Results will appear here with detailed breakdowns!
"""
        self.route_text.insert(tk.END, welcome_msg)
        self.route_text.config(state=tk.DISABLED)
        
    def create_tracking_tab(self):
        # Real-time Tracking Tab
        self.tracking_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tracking_frame, text="Live Tracking")
        
        # Status dashboard
        dashboard_frame = ttk.LabelFrame(self.tracking_frame, text="System Dashboard")
        dashboard_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Statistics
        self.stats_frame = ttk.Frame(dashboard_frame)
        self.stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.total_drivers_label = ttk.Label(self.stats_frame, text="Total Drivers: 0")
        self.total_drivers_label.grid(row=0, column=0, padx=20)
        
        self.active_deliveries_label = ttk.Label(self.stats_frame, text="Active Deliveries: 0")
        self.active_deliveries_label.grid(row=0, column=1, padx=20)
        
        self.completed_deliveries_label = ttk.Label(self.stats_frame, text="Completed: 0")
        self.completed_deliveries_label.grid(row=0, column=2, padx=20)
        
        # Live updates
        updates_frame = ttk.LabelFrame(self.tracking_frame, text="Live Updates")
        updates_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.updates_text = tk.Text(updates_frame, height=25, wrap=tk.WORD)
        updates_scrollbar = ttk.Scrollbar(updates_frame, orient=tk.VERTICAL, command=self.updates_text.yview)
        self.updates_text.configure(yscrollcommand=updates_scrollbar.set)
        
        self.updates_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        updates_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Control buttons
        control_frame = ttk.Frame(self.tracking_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.tracking_active = tk.BooleanVar()
        ttk.Checkbutton(control_frame, text="Real-time Tracking", 
                       variable=self.tracking_active, 
                       command=self.toggle_tracking).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Refresh", 
                  command=self.refresh_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Clear Log", 
                  command=self.clear_log).pack(side=tk.LEFT, padx=5)    
    def on_canvas_click(self, event):
        """Handle canvas click events"""
        if self.canvas_click_enabled.get():
            x = self.map_canvas.canvasx(event.x)
            y = self.map_canvas.canvasy(event.y)
            
            # Snap to grid
            x = round(x / self.grid_size) * self.grid_size
            y = round(y / self.grid_size) * self.grid_size
            
            self.add_intersection_at_coordinates(x, y)
    
    def on_canvas_motion(self, event):
        """Update coordinate display on mouse movement"""
        x = self.map_canvas.canvasx(event.x)
        y = self.map_canvas.canvasy(event.y)
        self.coord_label.config(text=f"Mouse: ({int(x)}, {int(y)})")
    
    def toggle_click_mode(self):
        """Toggle click-to-add mode"""
        self.canvas_click_enabled.set(not self.canvas_click_enabled.get())
        mode = "ON" if self.canvas_click_enabled.get() else "OFF"
        self.log_update(f"Click-to-add mode: {mode}")
        
        # Change cursor to indicate mode
        if self.canvas_click_enabled.get():
            self.map_canvas.config(cursor="crosshair")
        else:
            self.map_canvas.config(cursor="")
    
    def add_intersection_at_coordinates(self, x, y):
        """Add intersection at specific coordinates"""
        node_id = simpledialog.askstring("Add Intersection", 
                                       f"Enter intersection ID for position ({int(x)}, {int(y)}):")
        if node_id and node_id not in self.graph.nodes:
            self.graph.add_node(node_id, {"x": x, "y": y})
            self.routing = Routing(self.graph)
            self.update_location_combos()
            self.draw_graph()
            self.log_update(f"Added intersection {node_id} at ({int(x)}, {int(y)})")
    
    def quick_add_intersection(self, x, y):
        """Quick add intersection with preset coordinates"""
        node_id = simpledialog.askstring("Add Intersection", 
                                       f"Enter intersection ID for position ({x}, {y}):")
        if node_id and node_id not in self.graph.nodes:
            self.graph.add_node(node_id, {"x": x, "y": y})
            self.routing = Routing(self.graph)
            self.update_location_combos()
            self.draw_graph()
            self.log_update(f"Added intersection {node_id} at ({x}, {y})")
    
    def create_grid(self, rows, cols):
        """Create a grid of intersections"""
        start_x, start_y = 100, 100
        spacing_x, spacing_y = 150, 150
        
        nodes_added = []
        
        for row in range(rows):
            for col in range(cols):
                node_id = f"G{row+1}{col+1}"
                x = start_x + col * spacing_x
                y = start_y + row * spacing_y
                
                if node_id not in self.graph.nodes:
                    self.graph.add_node(node_id, {"x": x, "y": y})
                    nodes_added.append(node_id)
        
        # Add horizontal connections
        for row in range(rows):
            for col in range(cols - 1):
                node1 = f"G{row+1}{col+1}"
                node2 = f"G{row+1}{col+2}"
                if not self.graph.has_edge(node1, node2):
                    self.graph.add_edge(node1, node2, 5)  # Default weight
        
        # Add vertical connections
        for row in range(rows - 1):
            for col in range(cols):
                node1 = f"G{row+1}{col+1}"
                node2 = f"G{row+2}{col+1}"
                if not self.graph.has_edge(node1, node2):
                    self.graph.add_edge(node1, node2, 5)  # Default weight
        
        self.routing = Routing(self.graph)
        self.update_location_combos()
        self.draw_graph()
        self.log_update(f"Created {rows}x{cols} grid with {len(nodes_added)} new intersections")
    
    def create_linear(self, count):
        """Create a linear chain of intersections"""
        start_x, start_y = 100, 300
        spacing = 120
        
        nodes_added = []
        
        for i in range(count):
            node_id = f"L{i+1}"
            x = start_x + i * spacing
            y = start_y
            
            if node_id not in self.graph.nodes:
                self.graph.add_node(node_id, {"x": x, "y": y})
                nodes_added.append(node_id)
        
        # Connect adjacent nodes
        for i in range(count - 1):
            node1 = f"L{i+1}"
            node2 = f"L{i+2}"
            if not self.graph.has_edge(node1, node2):
                self.graph.add_edge(node1, node2, 4)  # Default weight
        
        self.routing = Routing(self.graph)
        self.update_location_combos()
        self.draw_graph()
        self.log_update(f"Created linear chain with {len(nodes_added)} new intersections")
    
    def create_sample_data(self):
        # Create sample intersections
        intersections = [
            ("A", 100, 100), ("B", 300, 100), ("C", 500, 100),
            ("D", 100, 300), ("E", 300, 300), ("F", 500, 300),
            ("G", 100, 500), ("H", 300, 500), ("I", 500, 500)
        ]
        
        for node_id, x, y in intersections:
            self.graph.add_node(node_id, {"x": x, "y": y})
        
        # Create sample roads with time weights
        roads = [
            ("A", "B", 5), ("B", "C", 7), ("A", "D", 6),
            ("B", "E", 4), ("C", "F", 3), ("D", "E", 8),
            ("E", "F", 5), ("D", "G", 9), ("E", "H", 6),
            ("F", "I", 4), ("G", "H", 7), ("H", "I", 5)
        ]
        
        for start, end, weight in roads:
            self.graph.add_edge(start, end, weight)
        
        # Recreate routing with updated graph
        self.routing = Routing(self.graph)
        
        # Update combo boxes
        self.update_location_combos()
        self.draw_graph()
        
        # Add sample drivers
        self.drivers["D001"] = Driver("D001", "John Doe", "A")
        self.drivers["D002"] = Driver("D002", "Jane Smith", "E")
        
        # Add sample deliveries
        self.deliveries["DEL001"] = Delivery("DEL001", "C")
        self.deliveries["DEL002"] = Delivery("DEL002", "I")
        
        self.refresh_all_displays()
    
    def add_intersection(self):
        # Simple dialog for now
        self.add_intersection_at_coordinates(400, 300)
    
    def add_road(self):
        nodes = list(self.graph.nodes.keys())
        if len(nodes) < 2:
            messagebox.showwarning("Warning", "Need at least 2 intersections to add a road")
            return
        
        start = simpledialog.askstring("Start Node", f"Start ({', '.join(nodes)}):")
        end = simpledialog.askstring("End Node", f"End ({', '.join(nodes)}):")
        weight = simpledialog.askinteger("Weight", "Time (minutes):", minvalue=1)
        
        if start in nodes and end in nodes and weight:
            self.graph.add_edge(start, end, weight)
            self.routing = Routing(self.graph)
            self.draw_graph()
            self.log_update(f"Added road from {start} to {end} (time: {weight} min)")
    
    def clear_map(self):
        self.graph = Graph()
        self.routing = Routing(self.graph)
        self.map_canvas.delete("all")
        self.update_location_combos()
        self.draw_graph()
        self.log_update("Map cleared")
    
    def draw_graph(self):
        self.map_canvas.delete("all")
        
        # Draw coordinate grid if enabled
        if self.show_coordinates.get():
            self.draw_coordinate_grid()
        
        # Draw edges first
        for node1 in self.graph.edges:
            for node2, weight in self.graph.edges[node1].items():
                x1, y1 = self.graph.nodes[node1]["x"], self.graph.nodes[node1]["y"]
                x2, y2 = self.graph.nodes[node2]["x"], self.graph.nodes[node2]["y"]
                
                # Draw line
                self.map_canvas.create_line(x1, y1, x2, y2, width=2, fill="blue")
                
                # Draw weight label
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                self.map_canvas.create_text(mid_x, mid_y, text=str(weight), 
                                          fill="red", font=("Arial", 8), 
                                          tags="weight")
        
        # Draw nodes
        for node_id, data in self.graph.nodes.items():
            x, y = data["x"], data["y"]
            # Draw circle for intersection
            self.map_canvas.create_oval(x-15, y-15, x+15, y+15, 
                                      fill="yellow", outline="black", width=2,
                                      tags="node")
            # Draw label
            self.map_canvas.create_text(x, y, text=node_id, font=("Arial", 10, "bold"),
                                      tags="node_label")
            # Draw coordinates
            self.map_canvas.create_text(x, y-25, text=f"({int(x)},{int(y)})", 
                                      font=("Arial", 7), fill="gray",
                                      tags="coordinates")
    
    def draw_coordinate_grid(self):
        """Draw coordinate grid on canvas"""
        # Get canvas dimensions
        canvas_width = 1000
        canvas_height = 800
        
        # Draw vertical lines
        for x in range(0, canvas_width, self.grid_size):
            self.map_canvas.create_line(x, 0, x, canvas_height, 
                                      fill="lightgray", width=1, tags="grid")
            if x % (self.grid_size * 2) == 0:  # Label every other line
                self.map_canvas.create_text(x, 10, text=str(x), 
                                          fill="gray", font=("Arial", 8), tags="grid_label")
        
        # Draw horizontal lines
        for y in range(0, canvas_height, self.grid_size):
            self.map_canvas.create_line(0, y, canvas_width, y, 
                                      fill="lightgray", width=1, tags="grid")
            if y % (self.grid_size * 2) == 0:  # Label every other line
                self.map_canvas.create_text(10, y, text=str(y), 
                                          fill="gray", font=("Arial", 8), tags="grid_label")
    
    def add_driver(self):
        driver_id = simpledialog.askstring("Add Driver", "Enter driver ID:")
        if driver_id and driver_id not in self.drivers:
            name = simpledialog.askstring("Driver Name", "Enter driver name:")
            nodes = list(self.graph.nodes.keys())
            if nodes:
                location = simpledialog.askstring("Location", f"Enter current location ({', '.join(nodes)}):")
                if location in nodes and name:
                    self.drivers[driver_id] = Driver(driver_id, name, location)
                    self.refresh_drivers_display()
                    self.log_update(f"Added driver {name} (ID: {driver_id}) at {location}")
    
    def update_driver_location(self):
        selection = self.drivers_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a driver")
            return
            
        item = self.drivers_tree.item(selection[0])
        driver_id = item['values'][0]
        
        nodes = list(self.graph.nodes.keys())
        location = simpledialog.askstring("Update Location", 
                                        f"New location ({', '.join(nodes)}):")
        
        if location in nodes:
            self.drivers[driver_id].update_location(location)
            self.refresh_drivers_display()
            self.log_update(f"Driver {driver_id} moved to {location}")
    
    def create_delivery(self):
        delivery_id = simpledialog.askstring("New Delivery", "Enter delivery ID:")
        if delivery_id and delivery_id not in self.deliveries:
            nodes = list(self.graph.nodes.keys())
            destination = simpledialog.askstring("Destination", 
                                               f"Enter destination ({', '.join(nodes)}):")
            if destination in nodes:
                self.deliveries[delivery_id] = Delivery(delivery_id, destination)
                self.refresh_deliveries_display()
                self.log_update(f"Created delivery {delivery_id} to {destination}")
    
    def assign_delivery(self):
        driver_selection = self.drivers_tree.selection()
        if not driver_selection:
            messagebox.showwarning("Warning", "Please select a driver")
            return
            
        driver_item = self.drivers_tree.item(driver_selection[0])
        driver_id = driver_item['values'][0]
        
        # Show available deliveries
        available_deliveries = [d_id for d_id, delivery in self.deliveries.items() 
                              if delivery.status == "Pending"]
        
        if not available_deliveries:
            messagebox.showinfo("Info", "No pending deliveries available")
            return
            
        delivery_id = simpledialog.askstring("Assign Delivery", 
                                           f"Select delivery ID ({', '.join(available_deliveries)}):")
        
        if delivery_id in available_deliveries:
            self.drivers[driver_id].assign_delivery(delivery_id)
            self.deliveries[delivery_id].update_status("Assigned")
            self.refresh_all_displays()
            self.log_update(f"Assigned delivery {delivery_id} to driver {driver_id}")
    
    def update_delivery_status(self):
        selection = self.deliveries_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a delivery")
            return
            
        item = self.deliveries_tree.item(selection[0])
        delivery_id = item['values'][0]
        
        statuses = ["Pending", "Assigned", "In Transit", "Delivered", "Completed"]
        status = simpledialog.askstring("Update Status", 
                                      f"New status ({', '.join(statuses)}):")
        
        if status in statuses:
            self.deliveries[delivery_id].update_status(status)
            self.refresh_deliveries_display()
            self.log_update(f"Delivery {delivery_id} status updated to {status}")
    
    def update_delivery_progress(self):
        selection = self.deliveries_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a delivery")
            return
            
        item = self.deliveries_tree.item(selection[0])
        delivery_id = item['values'][0]
        
        progress = simpledialog.askinteger("Update Progress", 
                                         "Progress increment (0-100):", 
                                         minvalue=0, maxvalue=100)
        
        if progress is not None:
            self.deliveries[delivery_id].update_progress(progress)
            self.refresh_deliveries_display()
            self.log_update(f"Delivery {delivery_id} progress updated by {progress}%")

    def find_route(self):
        start = self.start_var.get()
        end = self.end_var.get()
        algorithm = self.algorithm_var.get()
        
        if not start or not end:
            messagebox.showwarning("Warning", "Please select start and end locations")
            return
            
        if start not in self.graph.nodes or end not in self.graph.nodes:
            messagebox.showerror("Error", "Invalid start or end location")
            return
        
        try:
            if algorithm == "BFS":
                path = self.routing.find_shortest_path_bfs(start, end)
            elif algorithm == "DFS":
                path = self.routing.find_shortest_path_dfs(start, end)
            elif algorithm == "A* Search":
                path = self.routing.a_star_search(start, end)
            else:  # Dijkstra
                path = self.routing.shortest_path(start, end)
            
            if path:
                total_time = self.routing.calculate_route_time(path)
                result = f"\n🗺️ BASIC {algorithm.upper()} ROUTE CALCULATION\n"
                result += f"{'='*60}\n\n"
                result += f"Route: {start} → {end}\n"
                result += f"Algorithm: {algorithm}\n"
                result += f"Path: {' → '.join(path)}\n"
                result += f"Segments: {len(path)-1}\n"
                result += f"Travel Time: {total_time} minutes\n\n"
                result += f"💡 For cuisine-aware delivery estimates, select a dish\n"
                result += f"   and use 'Calculate Smart Delivery Time'\n"
                result += f"{'='*60}\n"
                
                self.route_text.config(state=tk.NORMAL)
                self.route_text.delete(1.0, tk.END)
                self.route_text.insert(tk.END, result)
                self.route_text.see(tk.END)
                self.route_text.config(state=tk.DISABLED)
                self.log_update(f"Found {algorithm} route from {start} to {end}")
            else:
                messagebox.showwarning("Warning", f"No route found from {start} to {end}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Route calculation failed: {str(e)}")

    def compare_algorithms(self):
        start = self.start_var.get()
        end = self.end_var.get()
        if not start or not end: return
        
        results = self.routing.compare_algorithms(start, end)
        self.route_text.config(state=tk.NORMAL)
        self.route_text.delete(1.0, tk.END)
        self.route_text.insert(tk.END, str(results))
        self.route_text.config(state=tk.DISABLED)

    def optimize_routes(self):
        self.log_update("Optimizing routes...")

    def toggle_tracking(self):
        if self.tracking_active.get():
            self.start_live_tracking()
        else:
            self.stop_live_tracking()
        self.refresh_drivers_display()
        self.refresh_deliveries_display()
        self.update_location_combos()
        
    def update_location_combos(self):
        nodes = sorted(list(self.graph.nodes.keys()))
        self.start_combo['values'] = nodes
        self.end_combo['values'] = nodes
        
    def refresh_drivers_display(self):
        for item in self.drivers_tree.get_children():
            self.drivers_tree.delete(item)
        for driver in self.drivers.values():
            self.drivers_tree.insert("", tk.END, values=(driver.driver_id, driver.name, driver.status, driver.current_location, len(driver.assigned_deliveries)))
            
    def refresh_deliveries_display(self):
        for item in self.deliveries_tree.get_children():
            self.deliveries_tree.delete(item)
        for delivery in self.deliveries.values():
            driver_name = "None"
            for d in self.drivers.values():
                if delivery.delivery_id in d.assigned_deliveries:
                    driver_name = d.name
                    break
            self.deliveries_tree.insert("", tk.END, values=(delivery.delivery_id, delivery.destination, delivery.status, f"{delivery.progress}%", driver_name, "Now"))

    def log_update(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.updates_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.updates_text.see(tk.END)

    def search_dishes_simple(self):
        query = self.dish_search_var.get()
        if not query:
            # Show all dishes if empty
            dishes = self.cuisine_calculator.get_all_dishes()
            self.dish_combo['values'] = dishes
            return

        # Smart Search with Fuzzy Matching
        import difflib
        all_dishes = self.cuisine_calculator.get_all_dishes()
        
        # 1. Exact/Substring match (High priority)
        matches = [d for d in all_dishes if query.lower() in d.lower()]
        
        # 2. Fuzzy match (typos)
        fuzzy_matches = difflib.get_close_matches(query, all_dishes, n=5, cutoff=0.6)
        
        # Combine and deduplicate, preserving order
        results = []
        seen = set()
        
        for d in matches + fuzzy_matches:
            if d not in seen:
                results.append(d)
                seen.add(d)
        
        if not results:
            self.dish_combo['values'] = ["No matches found"]
        else:
            # Add rich info to dropdown values (optional, or just keep names)
            # For now, just names to keep selection simple
            self.dish_combo['values'] = results
            
            # Auto-select first match if exact
            if len(results) > 0 and results[0].lower() == query.lower():
                self.dish_combo.set(results[0])
                self.on_dish_selected_simple(None)

    def on_dish_selected_simple(self, event):
        dish = self.selected_dish_var.get()
        info = self.cuisine_calculator.get_dish_info(dish)
        if info:
            self.dish_info_text.config(state=tk.NORMAL)
            self.dish_info_text.delete(1.0, tk.END)
            
            # Rich Display
            display_text = f"""DISH DETAILS
--------------------------------
Name:      {info['dish']}
Cuisine:   {info['cuisine']}
Category:  {info['category']}
Prep Time: {info['base_prep_time']} min
"""
            self.dish_info_text.insert(tk.END, display_text)
            self.dish_info_text.config(state=tk.DISABLED)

    def filter_by_cuisine(self, cuisine):
        dishes = self.cuisine_calculator.get_dishes_by_cuisine(cuisine)
        self.dish_combo['values'] = dishes

        start = self.start_var.get()
        end = self.end_var.get()
        dish = self.selected_dish_var.get()
        
        if not start or not end or not dish:
            messagebox.showwarning("Missing Information", "Please select start, end, and a dish.")
            return

        # Get route
        path = self.routing.a_star_search(start, end)
        if not path:
            self.route_text.config(state=tk.NORMAL)
            self.route_text.delete(1.0, tk.END)
            self.route_text.insert(tk.END, "No path found between selected locations.")
            self.route_text.config(state=tk.DISABLED)
            return

        # Calculate base metrics
        base_travel_time = self.routing.calculate_route_time(path)
        distance = base_travel_time * 0.5 # Approximation for demo
        
        dish_info = self.cuisine_calculator.get_dish_info(dish)
        if not dish_info:
            dish_info = {'base_prep_time': 15, 'cuisine': 'Unknown'} # Default
            
        # AI Prediction
        predicted_time = self.time_predictor.predict(
            dish_info['cuisine'], 
            dish_info['base_prep_time'], 
            distance
        )
        
        # Display results
        self.route_text.config(state=tk.NORMAL)
        self.route_text.delete(1.0, tk.END)
        
        result = f"""AI DELIVERY PREDICTION
        
Dish: {dish} ({dish_info['cuisine']})
Route: {' -> '.join(path)}

Time Breakdown:
------------------
Base Prep Time:    {dish_info['base_prep_time']} min
Travel Distance:   {distance:.1f} km
Traffic Factor:    1.2x (Simulated)

AI Predicted Total Time: {predicted_time:.1f} min
(vs Standard Estimate: {dish_info['base_prep_time'] + base_travel_time:.1f} min)

Confidence: High (Random Forest Model)
"""
        self.route_text.insert(tk.END, result)
        self.route_text.config(state=tk.DISABLED)

    def update_distance_display(self, *args):
        dist = self.distance_var.get()
        if dist > 0:
            self.distance_display_var.set(f"Real-World Distance: {dist:.2f} km")
        else:
            self.distance_display_var.set("Real-World Distance: Not Set")

    def import_osm_route(self, distance, time_min, geometry=None):
        """Import route data from OSM and visualize it on the graph"""
        # Update data variables
        self.distance_var.set(float(distance))
        
        # Clear existing graph to show just this route
        self.graph = Graph()
        self.drivers = {}
        self.deliveries = {}
        
        if not geometry or len(geometry) < 2:
            # Fallback to simple start/end
            start_node = "OSM_Start"
            end_node = "OSM_End"
            self.graph.add_node(start_node, {'x': 200, 'y': 400})
            self.graph.add_node(end_node, {'x': 800, 'y': 400})
            self.graph.add_edge(start_node, end_node, weight=round(distance, 2))
        else:
            # Visualize the full path
            # Normalize coordinates to fit canvas (approx 1000x800)
            lats = [p[0] for p in geometry]
            lons = [p[1] for p in geometry]
            
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            
            lat_range = max_lat - min_lat if max_lat != min_lat else 1
            lon_range = max_lon - min_lon if max_lon != min_lon else 1
            
            # Scale to canvas with padding
            padding = 50
            width = 1000 - 2 * padding
            height = 700 - 2 * padding
            
            # Downsample to reduce node count (User requested optimization)
            # Target approx 15 nodes for a cleaner look
            target_nodes = 15
            step = max(1, len(geometry) // target_nodes) 
            points = geometry[::step]
            if points[-1] != geometry[-1]:
                points.append(geometry[-1])
                
            prev_node = None
            
            for i, (lat, lon) in enumerate(points):
                node_id = f"R{i}"
                if i == 0: node_id = "Start"
                if i == len(points)-1: node_id = "End"
                
                # Simple projection
                x = padding + ((lon - min_lon) / lon_range) * width
                y = height + padding - ((lat - min_lat) / lat_range) * height # Invert Y for screen coords
                
                self.graph.add_node(node_id, {'x': int(x), 'y': int(y)})
                
                if prev_node:
                    # Calculate segment distance (approx)
                    self.graph.add_edge(prev_node, node_id, weight=distance/len(points))
                
                prev_node = node_id
        
        # Refresh display
        self.refresh_all_displays()
        self.log_update(f"Imported OSM Route: {distance:.2f} km, ~{time_min:.0f} min")
        
        # Switch to Map tab to show the result
        self.notebook.select(0) 

    def calculate_delivery_time_ai(self):
        """Calculate delivery time using AI model"""
        dish = self.selected_dish_var.get()
        
        if not dish:
            messagebox.showwarning("Missing Information", "Please select a dish.")
            return

        # Check for real-world distance first
        real_dist = self.distance_var.get()
        path = []
        
        if real_dist > 0:
            distance = real_dist
            # Estimate travel time (assuming 30km/h city speed)
            base_travel_time = (distance / 30) * 60 
            route_desc = "Real-World Route (OSM)"
        else:
            # Fallback to graph routing
            start = self.start_var.get()
            end = self.end_var.get()
            
            if not start or not end:
                messagebox.showwarning("Missing Information", "Please select start and end locations (or use Real-Time Map).")
                return

            path = self.routing.a_star_search(start, end)
            if not path:
                self.route_text.config(state=tk.NORMAL)
                self.route_text.delete(1.0, tk.END)
                self.route_text.insert(tk.END, "No path found between selected locations.")
                self.route_text.config(state=tk.DISABLED)
                return

            # Calculate base metrics
            base_travel_time = self.routing.calculate_route_time(path)
            distance = base_travel_time * 0.5 # Approximation
            route_desc = ' -> '.join(path)
        
        dish_info = self.cuisine_calculator.get_dish_info(dish)
        if not dish_info:
            dish_info = {'base_prep_time': 15, 'cuisine': 'Unknown'} # Default
            
        # AI Prediction
        predicted_time = self.time_predictor.predict(
            dish_info['cuisine'], 
            dish_info['base_prep_time'], 
            distance
        )
        
        # Display results
        self.route_text.config(state=tk.NORMAL)
        self.route_text.delete(1.0, tk.END)
        
        result = f"""AI DELIVERY PREDICTION
        
Dish: {dish} ({dish_info['cuisine']})
Route: {route_desc}

Time Breakdown:
------------------
Base Prep Time:    {dish_info['base_prep_time']} min
Travel Distance:   {distance:.1f} km
Traffic Factor:    1.2x (Simulated)

AI Predicted Total Time: {predicted_time:.1f} min
(vs Standard Estimate: {dish_info['base_prep_time'] + base_travel_time:.1f} min)

Confidence: High (Random Forest Model)
"""
        self.route_text.insert(tk.END, result)
        self.route_text.config(state=tk.DISABLED)

    def smart_assign_delivery(self):
        """Assign delivery using Smart Assignment Service"""
        # Get pending deliveries
        pending = [d for d in self.deliveries.values() if d.status == "Pending"]
        if not pending:
            messagebox.showinfo("Info", "No pending deliveries.")
            return
            
        # For demo, just pick first pending
        delivery = pending[0]
        
        # Find best driver
        best_driver = self.assignment_service.find_best_driver(delivery, self.drivers)
        
        if best_driver:
            best_driver.assign_delivery(delivery.delivery_id)
            delivery.update_status("Assigned")
            self.refresh_all_displays()
            
            msg = f"🤖 Smart Assigned {delivery.delivery_id} to {best_driver.name}\n"
            msg += f"Score: High | Rating: {best_driver.rating}⭐ | Load: {len(best_driver.assigned_deliveries)}"
            messagebox.showinfo("Smart Assignment", msg)
            self.log_update(f"Smart assigned {delivery.delivery_id} to {best_driver.name}")
        else:
            messagebox.showwarning("Warning", "No suitable drivers found.")

    def show_hotspots(self):
        """Visualize demand hotspots"""
        self.demand_predictor.generate_synthetic_history(self.graph.nodes)
        hotspots = self.demand_predictor.predict_hotspots()
        
        if len(hotspots) == 0:
            return
            
        # Draw hotspots
        for x, y in hotspots:
            # Draw semi-transparent red circle (simulated with stipple)
            r = 40
            self.map_canvas.create_oval(x-r, y-r, x+r, y+r, 
                                      fill="red", outline="", stipple="gray50", tags="hotspot")
            
        self.log_update(f"Visualized {len(hotspots)} demand hotspots")
        
        # Auto-clear after 5 seconds
        self.root.after(5000, lambda: self.map_canvas.delete("hotspot"))

    def refresh_all_displays(self):
        self.draw_graph()
        self.refresh_drivers_display()
        self.refresh_deliveries_display()
        self.update_location_combos()

    def draw_graph(self):
        self.map_canvas.delete("all")
        self.draw_coordinate_grid()
        
        # Draw edges
        for node1, edges in self.graph.edges.items():
            for node2, weight in edges.items():
                if node1 < node2:
                    x1, y1 = self.graph.nodes[node1]['x'], self.graph.nodes[node1]['y']
                    x2, y2 = self.graph.nodes[node2]['x'], self.graph.nodes[node2]['y']
                    self.map_canvas.create_line(x1, y1, x2, y2, fill="gray", width=2, tags="edge")
                    
                    # Draw weight
                    mx, my = (x1+x2)/2, (y1+y2)/2
                    self.map_canvas.create_text(mx, my, text=str(weight), fill="blue", font=("Arial", 8))
        
        # Draw nodes
        for node_id, data in self.graph.nodes.items():
            x, y = data['x'], data['y']
            self.map_canvas.create_oval(x-15, y-15, x+15, y+15, fill="white", outline="black", width=2, tags="node")
            self.map_canvas.create_text(x, y, text=node_id, font=("Arial", 8, "bold"), tags="node_label")
            self.map_canvas.create_text(x, y-25, text=f"({int(x)},{int(y)})", font=("Arial", 7), fill="gray", tags="coordinates")

    def refresh_data(self):
        self.refresh_all_displays()
        self.log_update("Data refreshed")

    def clear_log(self):
        self.updates_text.delete(1.0, tk.END)

    def toggle_tracking(self):
        if self.tracking_active.get():
            self.start_live_tracking()
        else:
            self.stop_live_tracking()

    def start_live_tracking(self):
        self.log_update("Live tracking started")
        
    def stop_live_tracking(self):
        self.log_update("Live tracking stopped")

    def open_image_map_creator(self):
        """Open the image map creator tool"""
        create_image_map(self)

    def open_real_time_map(self):
        """Open the real-time map tool"""
        create_real_time_map(self)

    def show_real_scale_view(self):
        messagebox.showinfo("Info", "Real scale view not implemented")

    # ══════════════════════════════════════════════════════════════════
    # AI Agent Tab
    # ══════════════════════════════════════════════════════════════════

    def create_ai_agent_tab(self):
        """Create the AI Agent tab for running HuggingFace-powered dispatch agent"""
        self.agent_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.agent_frame, text="AI Agent")

        # AI Agent state
        self.ai_agent = None
        self.ai_agent_thread = None
        self.ai_running = False

        # ── Top: Controls row ────────────────────────────────────────
        controls_frame = ttk.LabelFrame(self.agent_frame, text="Agent Controls")
        controls_frame.pack(fill=tk.X, padx=10, pady=5)

        controls_row = ttk.Frame(controls_frame)
        controls_row.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(controls_row, text="Task:").pack(side=tk.LEFT, padx=5)
        self.ai_task_var = tk.StringVar(value="easy")
        self._ai_task_values = list(TASK_IDS)
        self.ai_task_combo = ttk.Combobox(controls_row, textvariable=self.ai_task_var,
                                  values=self._ai_task_values, state="readonly", width=12)
        self.ai_task_combo.pack(side=tk.LEFT, padx=5)
        self.ai_task_combo.bind("<<ComboboxSelected>>", self._ai_on_task_selected)

        ttk.Button(controls_row, text="Use Map Editor Graph",
                   command=self._ai_load_from_map_editor).pack(side=tk.LEFT, padx=5)

        ttk.Label(controls_row, text="Model:").pack(side=tk.LEFT, padx=(15, 5))
        self.ai_model_var = tk.StringVar(value=DEFAULT_MODEL)
        model_entry = ttk.Entry(controls_row, textvariable=self.ai_model_var, width=35)
        model_entry.pack(side=tk.LEFT, padx=5)

        self.ai_run_btn = ttk.Button(controls_row, text="Run Agent",
                                     command=self._ai_start_agent)
        self.ai_run_btn.pack(side=tk.LEFT, padx=10)

        self.ai_stop_btn = ttk.Button(controls_row, text="Stop",
                                      command=self._ai_stop_agent, state=tk.DISABLED)
        self.ai_stop_btn.pack(side=tk.LEFT, padx=5)

        self.ai_run_all_btn = ttk.Button(controls_row, text="Run All Tasks",
                                         command=self._ai_start_all)
        self.ai_run_all_btn.pack(side=tk.LEFT, padx=5)

        # ── Middle: Split into Map + Log ──────────────────────────────
        middle_frame = ttk.Frame(self.agent_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left: Task info + Map canvas
        left_frame = ttk.Frame(middle_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Task info
        info_frame = ttk.LabelFrame(left_frame, text="Task Info")
        info_frame.pack(fill=tk.X, pady=(0, 5))

        self.ai_task_info = tk.Text(info_frame, height=4, wrap=tk.WORD,
                                    font=('Arial', 9), state=tk.DISABLED)
        self.ai_task_info.pack(fill=tk.X, padx=5, pady=5)

        # Map canvas
        map_frame = ttk.LabelFrame(left_frame, text="Environment Map")
        map_frame.pack(fill=tk.BOTH, expand=True)

        self.ai_canvas = tk.Canvas(map_frame, bg='white', width=500, height=400)
        self.ai_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right: Action log + Scorecard
        right_frame = ttk.Frame(middle_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Scorecard
        score_frame = ttk.LabelFrame(right_frame, text="Scorecard")
        score_frame.pack(fill=tk.X, pady=(0, 5))

        score_inner = ttk.Frame(score_frame)
        score_inner.pack(fill=tk.X, padx=5, pady=5)

        self.ai_score_labels = {}
        score_items = [("Score", 0), ("Completion", 1), ("Efficiency", 2),
                       ("Speed", 3), ("Validity", 4)]
        for name, col in score_items:
            ttk.Label(score_inner, text=name+":", font=('Arial', 9, 'bold')).grid(
                row=0, column=col*2, padx=(10,2), sticky='e')
            lbl = ttk.Label(score_inner, text="—", font=('Consolas', 10))
            lbl.grid(row=0, column=col*2+1, padx=(0,10), sticky='w')
            self.ai_score_labels[name.lower()] = lbl

        # Step counter
        self.ai_step_label = ttk.Label(score_frame, text="Step: 0 / 0")
        self.ai_step_label.pack(padx=5, pady=(0, 5))

        # Action log
        log_frame = ttk.LabelFrame(right_frame, text="Agent Actions")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.ai_log = tk.Text(log_frame, wrap=tk.WORD, font=('Consolas', 9),
                               state=tk.DISABLED)
        ai_log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL,
                                       command=self.ai_log.yview)
        self.ai_log.configure(yscrollcommand=ai_log_scroll.set)
        self.ai_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        ai_log_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        # Configure log tags
        self.ai_log.tag_configure("valid", foreground="green")
        self.ai_log.tag_configure("invalid", foreground="red")
        self.ai_log.tag_configure("info", foreground="blue")
        self.ai_log.tag_configure("header", foreground="#333", font=('Consolas', 10, 'bold'))

        ttk.Button(right_frame, text="Clear Log",
                   command=self._ai_clear_log).pack(pady=(5, 0))

        # Custom map-editor task config holder
        self._ai_custom_config = None

        # Load initial task
        self._ai_on_task_selected(None)

    # ── AI Agent: Load from Map Editor ────────────────────────────────

    def _ai_load_from_map_editor(self):
        """Import the current Map Editor graph, drivers, and deliveries as a custom AI agent task."""
        # Validate that the map has enough data
        if not self.graph.nodes:
            messagebox.showwarning("Empty Map", "The Map Editor has no intersections. Add nodes first.")
            return
        if not self.graph.edges:
            messagebox.showwarning("No Roads", "The Map Editor has no roads. Add edges first.")
            return
        if not self.drivers:
            messagebox.showwarning("No Drivers", "Add at least one driver in the Drivers tab before using the map for the agent.")
            return
        if not self.deliveries:
            messagebox.showwarning("No Deliveries", "Add at least one delivery in the Deliveries tab before using the map for the agent.")
            return

        # Build nodes list
        nodes = []
        for nid, data in self.graph.nodes.items():
            nodes.append({"id": nid, "x": float(data.get("x", 0)), "y": float(data.get("y", 0))})

        # Build edges list (deduplicate undirected edges)
        seen_edges = set()
        edges = []
        for n1, neighbors in self.graph.edges.items():
            for n2, weight in neighbors.items():
                edge_key = tuple(sorted([n1, n2]))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append({"start": n1, "end": n2, "weight": float(weight)})

        # Build drivers list
        drivers_list = []
        for drv in self.drivers.values():
            drivers_list.append({
                "id": drv.driver_id,
                "name": drv.name,
                "location": drv.current_location,
                "capacity": getattr(drv, 'capacity', 2),
            })

        # Build deliveries list
        deliveries_list = []
        for dlv in self.deliveries.values():
            entry = {"id": dlv.delivery_id, "destination": dlv.destination}
            if hasattr(dlv, 'pickup') and dlv.pickup:
                entry["pickup"] = dlv.pickup
            deliveries_list.append(entry)

        # Build the description for the LLM
        node_ids = [n["id"] for n in nodes]
        edge_strs = [f"{e['start']}--{e['weight']}--{e['end']}" for e in edges]
        driver_strs = [f"{d['id']} ({d['name']}) at {d['location']}" for d in drivers_list]
        delivery_strs = []
        for d in deliveries_list:
            s = f"{d['id']}: deliver to {d['destination']}"
            if d.get('pickup'):
                s += f" (pickup at {d['pickup']})"
            delivery_strs.append(s)

        description = (
            "You are a delivery dispatcher managing a custom map.\n\n"
            f"GRAPH: {len(nodes)} nodes: {', '.join(node_ids)}\n"
            f"EDGES: {', '.join(edge_strs)}\n\n"
            f"DRIVERS:\n" + "\n".join(f"  - {s}" for s in driver_strs) + "\n\n"
            f"DELIVERIES:\n" + "\n".join(f"  - {s}" for s in delivery_strs) + "\n\n"
            "OBJECTIVE: Assign each delivery to a driver, route them "
            "efficiently, and complete all deliveries.\n\n"
            "ACTIONS AVAILABLE:\n"
            "  - assign_driver(driver_id, delivery_id)\n"
            "  - move_driver(driver_id, target_node)  [adjacent nodes only]\n"
            "  - pickup_delivery(driver_id, delivery_id)  [at pickup node, if applicable]\n"
            "  - complete_delivery(driver_id, delivery_id)  [at destination]\n"
        )

        # Determine max steps heuristic
        max_steps = max(10, len(deliveries_list) * 8 + len(nodes) * 2)

        from env.models import TaskConfig
        self._ai_custom_config = TaskConfig(
            task_id="custom",
            task_name="Map Editor Task",
            difficulty="custom",
            description=description,
            max_steps=max_steps,
            seed=42,
            nodes=nodes,
            edges=edges,
            drivers=drivers_list,
            deliveries=deliveries_list,
        )

        # Add "custom" to the task dropdown if not present, then select it
        if "custom" not in self._ai_task_values:
            self._ai_task_values.append("custom")
            self.ai_task_combo['values'] = self._ai_task_values

        self.ai_task_var.set("custom")
        self._ai_on_task_selected(None)

        self._ai_log_msg("Loaded map from Map Editor as custom task.", "info")
        self._ai_log_msg(
            f"  Nodes: {len(nodes)}  |  Edges: {len(edges)}  |  "
            f"Drivers: {len(drivers_list)}  |  Deliveries: {len(deliveries_list)}", "info")

    # ── AI Agent: Event handlers ─────────────────────────────────────

    def _ai_on_task_selected(self, event):
        """Load task info and draw the map"""
        task_id = self.ai_task_var.get()
        try:
            if task_id == "custom" and self._ai_custom_config:
                cfg = self._ai_custom_config
            else:
                cfg = get_task(task_id)
            self._ai_current_config = cfg

            self.ai_task_info.config(state=tk.NORMAL)
            self.ai_task_info.delete("1.0", tk.END)
            self.ai_task_info.insert(tk.END,
                f"Task: {cfg.task_name}  |  Difficulty: {cfg.difficulty.upper()}  |  "
                f"Max Steps: {cfg.max_steps}  |  Drivers: {len(cfg.drivers)}  |  "
                f"Deliveries: {len(cfg.deliveries)}  |  Nodes: {len(cfg.nodes)}")
            self.ai_task_info.config(state=tk.DISABLED)

            self._ai_draw_map(cfg)
            self._ai_reset_scores()
            self.ai_step_label.config(text=f"Step: 0 / {cfg.max_steps}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load task: {e}")

    def _ai_start_agent(self):
        """Start the AI agent on the selected task"""
        if self.ai_running:
            return
        task_id = self.ai_task_var.get()
        self.ai_running = True
        self.ai_run_btn.config(state=tk.DISABLED)
        self.ai_run_all_btn.config(state=tk.DISABLED)
        self.ai_stop_btn.config(state=tk.NORMAL)
        self._ai_clear_log()
        self._ai_reset_scores()

        self._ai_log_msg(f"{'='*45}", "header")
        self._ai_log_msg(f"  Starting AI Agent — Task: {task_id.upper()}", "header")
        self._ai_log_msg(f"  Model: {self.ai_model_var.get()}", "info")
        self._ai_log_msg(f"{'='*45}\n", "header")

        self.ai_agent = DeliveryAgent(
            api_base=DEFAULT_API_BASE,
            model=self.ai_model_var.get(),
            api_key=DEFAULT_HF_TOKEN,
        )

        self.ai_agent_thread = threading.Thread(
            target=self._ai_run_thread, args=(task_id,), daemon=True)
        self.ai_agent_thread.start()

    def _ai_start_all(self):
        """Run all tasks sequentially"""
        if self.ai_running:
            return
        self.ai_running = True
        self.ai_run_btn.config(state=tk.DISABLED)
        self.ai_run_all_btn.config(state=tk.DISABLED)
        self.ai_stop_btn.config(state=tk.NORMAL)
        self._ai_clear_log()
        self._ai_reset_scores()

        self._ai_log_msg(f"{'='*45}", "header")
        self._ai_log_msg("  Running ALL Tasks (easy → medium → hard)", "header")
        self._ai_log_msg(f"{'='*45}\n", "header")

        self.ai_agent = DeliveryAgent(
            api_base=DEFAULT_API_BASE,
            model=self.ai_model_var.get(),
            api_key=DEFAULT_HF_TOKEN,
        )

        self.ai_agent_thread = threading.Thread(
            target=self._ai_run_all_thread, daemon=True)
        self.ai_agent_thread.start()

    def _ai_stop_agent(self):
        """Stop the running agent"""
        if self.ai_agent:
            self.ai_agent.stop()
        self._ai_log_msg("\n⏹ Agent stopped by user.", "invalid")

    def _ai_run_thread(self, task_id):
        """Background thread: run single task"""
        try:
            custom_cfg = self._ai_custom_config if task_id == "custom" else None
            report = self.ai_agent.run_task(
                task_id,
                on_step=self._ai_on_step,
                on_error=lambda msg: self.root.after(0, self._ai_log_msg, f"ERROR: {msg}", "invalid"),
                task_config=custom_cfg,
            )
            if report:
                self.root.after(0, self._ai_show_report, report)
            self.root.after(0, self._ai_finished)
        except Exception as e:
            self.root.after(0, self._ai_log_msg, f"ERROR: {e}", "invalid")
            self.root.after(0, self._ai_finished)

    def _ai_run_all_thread(self):
        """Background thread: run all tasks"""
        try:
            all_reports = {}
            for task_id in TASK_IDS:
                if not self.ai_running:
                    break
                self.root.after(0, self._ai_switch_task, task_id)
                self.root.after(0, self._ai_log_msg,
                    f"\n{'─'*35}\n  Task: {task_id.upper()}\n{'─'*35}", "header")

                report = self.ai_agent.run_task(
                    task_id,
                    on_step=self._ai_on_step,
                    on_error=lambda msg: self.root.after(0, self._ai_log_msg, f"ERROR: {msg}", "invalid"),
                )
                if report:
                    all_reports[task_id] = report
                    self.root.after(0, self._ai_log_msg,
                        f"  ✓ Score: {report.score:.4f}", "valid")

            if all_reports:
                avg = sum(r.score for r in all_reports.values()) / len(all_reports)
                self.root.after(0, self._ai_log_msg,
                    f"\n{'='*35}\n  AGGREGATE: {avg:.4f}\n{'='*35}", "header")
                last = list(all_reports.values())[-1]
                self.root.after(0, self._ai_show_report, last)

            self.root.after(0, self._ai_finished)
        except Exception as e:
            self.root.after(0, self._ai_log_msg, f"ERROR: {e}", "invalid")
            self.root.after(0, self._ai_finished)

    def _ai_switch_task(self, task_id):
        """Switch task display (main thread)"""
        self.ai_task_var.set(task_id)
        self._ai_on_task_selected(None)

    # ── AI Agent: Callbacks ───────────────────────────────────────────

    def _ai_on_step(self, step_num, action_dict, reward, valid, obs_summary, result):
        """Called after each agent step (from background thread)"""
        self.root.after(0, self._ai_update_step,
                        step_num, action_dict, reward, valid, obs_summary, result)

    def _ai_update_step(self, step_num, action_dict, reward, valid, obs_summary, result):
        """Update UI after a step (main thread)"""
        max_steps = self._ai_current_config.max_steps if hasattr(self, '_ai_current_config') else '?'
        self.ai_step_label.config(text=f"Step: {step_num} / {max_steps}")

        action_type = action_dict.get('action_type', '?')
        driver = action_dict.get('driver_id', '')
        delivery = action_dict.get('delivery_id', '')
        target = action_dict.get('target_node', '')

        desc = f"{action_type}"
        if driver: desc += f" {driver}"
        if delivery: desc += f" → {delivery}"
        if target: desc += f" → {target}"

        tag = "valid" if valid else "invalid"
        self._ai_log_msg(f"[{step_num:>2}] {desc}  (reward: {reward:+.3f} {'✓' if valid else '✗'})", tag)

        # Update map with driver positions
        if result and result.observation:
            self._ai_update_map(result.observation)

    def _ai_finished(self):
        """Agent finished (main thread)"""
        self.ai_running = False
        self.ai_run_btn.config(state=tk.NORMAL)
        self.ai_run_all_btn.config(state=tk.NORMAL)
        self.ai_stop_btn.config(state=tk.DISABLED)
        if not self.ai_agent or not self.ai_agent._stop_requested:
            self._ai_log_msg("\n✅ Agent finished.", "valid")

    def _ai_show_report(self, report):
        """Display grade report in scorecard"""
        self.ai_score_labels["score"].config(text=f"{report.score:.4f}")
        self.ai_score_labels["completion"].config(text=f"{report.completion_score:.4f}")
        self.ai_score_labels["efficiency"].config(text=f"{report.efficiency_score:.4f}")
        self.ai_score_labels["speed"].config(text=f"{report.speed_score:.4f}")
        self.ai_score_labels["validity"].config(text=f"{report.validity_score:.4f}")

    # ── AI Agent: Map drawing ─────────────────────────────────────────

    def _ai_draw_map(self, cfg):
        """Draw the task graph on the AI agent canvas"""
        self.ai_canvas.delete("all")
        self.ai_canvas.update_idletasks()

        cw = self.ai_canvas.winfo_width() or 500
        ch = self.ai_canvas.winfo_height() or 400

        if not cfg.nodes:
            return

        xs = [n["x"] for n in cfg.nodes]
        ys = [n["y"] for n in cfg.nodes]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        range_x = max_x - min_x or 1
        range_y = max_y - min_y or 1

        pad = 50
        scale = min((cw - 2*pad) / range_x, (ch - 2*pad) / range_y)
        off_x = (cw - range_x * scale) / 2
        off_y = (ch - range_y * scale) / 2

        def tx(x): return off_x + (x - min_x) * scale
        def ty(y): return off_y + (y - min_y) * scale

        node_pos = {}
        for n in cfg.nodes:
            node_pos[n["id"]] = (tx(n["x"]), ty(n["y"]))

        # Edges
        for edge in cfg.edges:
            x1, y1 = node_pos[edge["start"]]
            x2, y2 = node_pos[edge["end"]]
            self.ai_canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)
            mx, my = (x1+x2)/2, (y1+y2)/2
            self.ai_canvas.create_text(mx, my-8, text=str(edge["weight"]),
                                       fill="red", font=("Arial", 8))

        # Identify special nodes
        driver_nodes = {d["location"] for d in cfg.drivers}
        pickup_nodes = {d["pickup"] for d in cfg.deliveries if d.get("pickup")}
        dest_nodes = {d["destination"] for d in cfg.deliveries}

        # Nodes
        r = 15
        for nid, (x, y) in node_pos.items():
            if nid in driver_nodes:
                fill = "#90EE90"  # light green
            elif nid in pickup_nodes:
                fill = "#FFD700"  # gold
            elif nid in dest_nodes:
                fill = "#FF6B6B"  # light red
            else:
                fill = "yellow"

            self.ai_canvas.create_oval(x-r, y-r, x+r, y+r,
                                       fill=fill, outline="black", width=2,
                                       tags=f"node_{nid}")
            self.ai_canvas.create_text(x, y, text=nid,
                                       font=("Arial", 9, "bold"),
                                       tags=f"lbl_{nid}")

        # Driver labels
        for drv in cfg.drivers:
            if drv["location"] in node_pos:
                x, y = node_pos[drv["location"]]
                self.ai_canvas.create_text(x, y-r-12,
                    text=f"🚗{drv['id']}", fill="green",
                    font=("Arial", 8, "bold"), tags=f"drv_{drv['id']}")

        # Delivery labels
        for dlv in cfg.deliveries:
            if dlv["destination"] in node_pos:
                x, y = node_pos[dlv["destination"]]
                self.ai_canvas.create_text(x, y+r+12,
                    text=f"📦{dlv['id']}", fill="red",
                    font=("Arial", 8), tags=f"del_{dlv['id']}")
            if dlv.get("pickup") and dlv["pickup"] in node_pos:
                x, y = node_pos[dlv["pickup"]]
                self.ai_canvas.create_text(x, y+r+12,
                    text=f"🍔{dlv['id']}", fill="#B8860B",
                    font=("Arial", 8), tags=f"pkp_{dlv['id']}")

        self._ai_node_positions = node_pos

    def _ai_update_map(self, obs):
        """Update driver positions on the AI map (main thread)"""
        if not hasattr(self, '_ai_node_positions'):
            return
        node_pos = self._ai_node_positions
        r = 15

        for drv in obs.drivers:
            self.ai_canvas.delete(f"drv_{drv.driver_id}")
            if drv.current_location in node_pos:
                x, y = node_pos[drv.current_location]
                self.ai_canvas.create_text(x, y-r-12,
                    text=f"🚗{drv.driver_id}", fill="green",
                    font=("Arial", 8, "bold"), tags=f"drv_{drv.driver_id}")

        for dlv in obs.deliveries:
            if dlv.status == "Completed":
                self.ai_canvas.delete(f"del_{dlv.delivery_id}")
                if dlv.destination in node_pos:
                    x, y = node_pos[dlv.destination]
                    self.ai_canvas.create_text(x, y+r+12,
                        text=f"✅{dlv.delivery_id}", fill="green",
                        font=("Arial", 8), tags=f"del_{dlv.delivery_id}")

    # ── AI Agent: Helpers ─────────────────────────────────────────────

    def _ai_log_msg(self, message, tag=None):
        """Append a message to the AI agent log"""
        self.ai_log.config(state=tk.NORMAL)
        if tag:
            self.ai_log.insert(tk.END, message + "\n", tag)
        else:
            self.ai_log.insert(tk.END, message + "\n")
        self.ai_log.see(tk.END)
        self.ai_log.config(state=tk.DISABLED)

    def _ai_clear_log(self):
        """Clear the AI agent log"""
        self.ai_log.config(state=tk.NORMAL)
        self.ai_log.delete("1.0", tk.END)
        self.ai_log.config(state=tk.DISABLED)

    def _ai_reset_scores(self):
        """Reset scorecard values"""
        for lbl in self.ai_score_labels.values():
            lbl.config(text="—")

def main():
    root = tk.Tk()
    app = DeliveryTrackerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
