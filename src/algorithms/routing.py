from collections import deque
import heapq
import math

class Routing:
    def __init__(self, graph):
        self.graph = graph

    def find_shortest_path_bfs(self, start_node, end_node):
        """BFS implementation for finding shortest path"""
        if start_node not in self.graph.nodes or end_node not in self.graph.nodes:
            return None
        
        queue = deque([(start_node, [start_node])])
        visited = {start_node}
        
        while queue:
            current, path = queue.popleft()
            
            if current == end_node:
                return path
            
            for neighbor in self.graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None

    def find_shortest_path_dfs(self, start_node, end_node):
        """DFS implementation for finding path"""
        if start_node not in self.graph.nodes or end_node not in self.graph.nodes:
            return None
        
        def dfs_recursive(current, path, visited):
            if current == end_node:
                return path
            
            visited.add(current)
            
            for neighbor in self.graph.get_neighbors(current):
                if neighbor not in visited:
                    result = dfs_recursive(neighbor, path + [neighbor], visited.copy())
                    if result:
                        return result
            
            return None
        
        return dfs_recursive(start_node, [start_node], set())

    def shortest_path(self, start, end):
        """Dijkstra's algorithm for shortest path with weights"""
        if start not in self.graph.nodes or end not in self.graph.nodes:
            return None
        
        # Dijkstra's algorithm
        distances = {node: float('inf') for node in self.graph.nodes}
        distances[start] = 0
        previous = {}
        pq = [(0, start)]
        
        while pq:
            current_distance, current = heapq.heappop(pq)
            
            if current == end:
                # Reconstruct path
                path = []
                while current in previous:
                    path.append(current)
                    current = previous[current]
                path.append(start)
                return path[::-1]
            
            if current_distance > distances[current]:
                continue
            
            for neighbor in self.graph.get_neighbors(current):
                weight = self.graph.get_edge_weight(current, neighbor)
                distance = current_distance + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        return None

    def a_star_search(self, start, end):
        """A* algorithm for optimal pathfinding with heuristic"""
        if start not in self.graph.nodes or end not in self.graph.nodes:
            return None
        
        # Get coordinates for heuristic calculation
        def get_coordinates(node):
            attrs = self.graph.nodes.get(node, {})
            return attrs.get('x', 0), attrs.get('y', 0)
        
        def heuristic(node1, node2):
            """Euclidean distance heuristic"""
            x1, y1 = get_coordinates(node1)
            x2, y2 = get_coordinates(node2)
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # A* algorithm implementation
        open_set = [(0, start)]  # (f_score, node)
        came_from = {}
        g_score = {node: float('inf') for node in self.graph.nodes}
        g_score[start] = 0
        f_score = {node: float('inf') for node in self.graph.nodes}
        f_score[start] = heuristic(start, end)
        
        open_set_hash = {start}  # For O(1) membership testing
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_hash.discard(current)
            
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in self.graph.get_neighbors(current):
                # Calculate tentative g_score
                edge_weight = self.graph.get_edge_weight(current, neighbor)
                tentative_g_score = g_score[current] + edge_weight
                
                if tentative_g_score < g_score[neighbor]:
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        return None

    def calculate_route_time(self, path):
        """Calculate total time for a given path"""
        if not path or len(path) < 2:
            return 0
        
        total_time = 0
        for i in range(len(path) - 1):
            total_time += self.graph.get_edge_weight(path[i], path[i + 1])
        return total_time
    
    def compare_algorithms(self, start, end):
        """Compare all algorithms and return performance metrics"""
        import time
        
        algorithms = {
            'BFS': self.find_shortest_path_bfs,
            'DFS': self.find_shortest_path_dfs,
            'Dijkstra': self.shortest_path,
            'A* Search': self.a_star_search
        }
        
        results = {}
        
        for name, algorithm in algorithms.items():
            start_time = time.time()
            path = algorithm(start, end)
            end_time = time.time()
            
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            if path:
                route_time = self.calculate_route_time(path)
                path_length = len(path)
            else:
                route_time = float('inf')
                path_length = 0
            
            results[name] = {
                'path': path,
                'execution_time_ms': round(execution_time, 3),
                'route_time': route_time,
                'path_length': path_length,
                'found_path': path is not None
            }
        
        return results