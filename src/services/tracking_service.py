from datetime import datetime

class TrackingService:
    def __init__(self):
        self.drivers = {}

    def add_driver(self, driver_id, initial_location):
        self.drivers[driver_id] = {
            'location': initial_location,
            'status': 'active',
            'last_update': datetime.now()
        }

    def update_driver_location(self, driver_id, new_location):
        if driver_id in self.drivers:
            self.drivers[driver_id]['location'] = new_location
            self.drivers[driver_id]['last_update'] = datetime.now()
        else:
            raise ValueError("Driver ID not found.")

    def update_driver_status(self, driver_id, status):
        if driver_id in self.drivers:
            self.drivers[driver_id]['status'] = status
            self.drivers[driver_id]['last_update'] = datetime.now()
        else:
            raise ValueError("Driver ID not found.")

    def get_driver_info(self, driver_id):
        if driver_id in self.drivers:
            return self.drivers[driver_id]
        else:
            raise ValueError("Driver ID not found.")

    def get_all_drivers(self):
        return self.drivers