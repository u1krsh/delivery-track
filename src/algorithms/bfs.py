from collections import deque

def bfs(graph, start_node, end_node):
    """
    Breadth-First Search algorithm to find shortest path between two nodes
    """
    if start_node not in graph.nodes or end_node not in graph.nodes:
        return None
    
    queue = deque([(start_node, [start_node])])
    visited = {start_node}
    
    while queue:
        current, path = queue.popleft()
        
        if current == end_node:
            return path
        
        for neighbor in graph.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None

def bfs_all_paths(graph, start_node):
    """
    BFS to find shortest paths from start_node to all reachable nodes
    """
    if start_node not in graph.nodes:
        return {}
    
    distances = {start_node: 0}
    paths = {start_node: [start_node]}
    queue = deque([start_node])
    visited = {start_node}
    
    while queue:
        current = queue.popleft()
        
        for neighbor in graph.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = distances[current] + 1
                paths[neighbor] = paths[current] + [neighbor]
                queue.append(neighbor)
    
    return paths
