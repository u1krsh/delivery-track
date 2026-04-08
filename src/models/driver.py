class Driver:
    def __init__(self, driver_id, name, current_location):
        self.driver_id = driver_id
        self.name = name
        self.current_location = current_location
        self.status = "Available"
        self.assigned_deliveries = []
        self.route_history = []
        self.rating = 5.0 # Default 5 stars
        self.efficiency_score = 100.0 # Default 100% efficiency

    def update_location(self, new_location):
        """Update driver's current location"""
        self.route_history.append({
            "from": self.current_location,
            "to": new_location,
            "timestamp": None  # You can add datetime if needed
        })
        self.current_location = new_location

    def assign_delivery(self, delivery_id):
        """Assign a delivery to this driver"""
        if delivery_id not in self.assigned_deliveries:
            self.assigned_deliveries.append(delivery_id)
            self.status = "Busy" if self.assigned_deliveries else "Available"

    def complete_delivery(self, delivery_id):
        """Mark a delivery as completed"""
        if delivery_id in self.assigned_deliveries:
            self.assigned_deliveries.remove(delivery_id)
            self.status = "Busy" if self.assigned_deliveries else "Available"

    def get_driver_info(self):
        """Get driver information"""
        return {
            "driver_id": self.driver_id,
            "name": self.name,
            "current_location": self.current_location,
            "status": self.status,
            "assigned_deliveries": self.assigned_deliveries,
            "total_deliveries": len(self.route_history)
        }

    def is_available(self):
        """Check if driver is available for new deliveries"""
        return self.status == "Available"

    def get_workload(self):
        """Get current workload (number of assigned deliveries)"""
        return len(self.assigned_deliveries)