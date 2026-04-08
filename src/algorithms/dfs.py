def dfs(graph, start_node, end_node):
    """
    Depth-First Search algorithm to find a path between two nodes
    """
    if start_node not in graph.nodes or end_node not in graph.nodes:
        return None
    
    def dfs_recursive(current, path, visited):
        if current == end_node:
            return path
        
        visited.add(current)
        
        for neighbor in graph.get_neighbors(current):
            if neighbor not in visited:
                result = dfs_recursive(neighbor, path + [neighbor], visited.copy())
                if result:
                    return result
        
        return None
    
    return dfs_recursive(start_node, [start_node], set())

def dfs_iterative(graph, start_node, end_node):
    """
    Iterative DFS implementation using stack
    """
    if start_node not in graph.nodes or end_node not in graph.nodes:
        return None
    
    stack = [(start_node, [start_node])]
    visited = set()
    
    while stack:
        current, path = stack.pop()
        
        if current == end_node:
            return path
        
        if current not in visited:
            visited.add(current)
            
            for neighbor in graph.get_neighbors(current):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
    
    return None

def dfs_all_paths(graph, start_node, end_node):
    """
    Find all possible paths between start_node and end_node using DFS
    """
    if start_node not in graph.nodes or end_node not in graph.nodes:
        return []
    
    all_paths = []
    
    def dfs_all_recursive(current, path, visited):
        if current == end_node:
            all_paths.append(path[:])
            return
        
        visited.add(current)
        
        for neighbor in graph.get_neighbors(current):
            if neighbor not in visited:
                path.append(neighbor)
                dfs_all_recursive(neighbor, path, visited.copy())
                path.pop()
    
    dfs_all_recursive(start_node, [start_node], set())
    return all_paths