class Delivery:
    def __init__(self, delivery_id, destination):
        self.delivery_id = delivery_id
        self.destination = destination
        self.status = "Pending"
        self.progress = 0

    def update_status(self, new_status):
        self.status = new_status

    def update_progress(self, progress_increment):
        self.progress += progress_increment
        if self.progress >= 100:
            self.status = "Completed"

    def get_delivery_info(self):
        return {
            "delivery_id": self.delivery_id,
            "destination": self.destination,
            "status": self.status,
            "progress": self.progress
        }