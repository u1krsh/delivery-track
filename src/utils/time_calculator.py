def calculate_travel_time(distance, speed):
    if speed <= 0:
        raise ValueError("Speed must be greater than zero.")
    return distance / speed

def calculate_distance(coord1, coord2):
    from geopy.distance import great_circle
    return great_circle(coord1, coord2).meters

def estimate_time_between_points(coord1, coord2, speed):
    distance = calculate_distance(coord1, coord2)
    return calculate_travel_time(distance, speed)