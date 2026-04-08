class Graph:
    def __init__(self):
        self.nodes = {}  # node_id: {attributes}
        self.edges = {}  # node_id: {neighbor_id: weight}
    
    def add_node(self, node_id, attributes=None):
        """Add a node to the graph"""
        self.nodes[node_id] = attributes or {}
        if node_id not in self.edges:
            self.edges[node_id] = {}
    
    def add_edge(self, node1, node2, weight=1):
        """Add an edge between two nodes (undirected)"""
        if node1 not in self.nodes:
            self.add_node(node1)
        if node2 not in self.nodes:
            self.add_node(node2)
        
        self.edges[node1][node2] = weight
        self.edges[node2][node1] = weight  # Undirected graph
    
    def remove_edge(self, node1, node2):
        """Remove edge between two nodes"""
        if node1 in self.edges and node2 in self.edges[node1]:
            del self.edges[node1][node2]
        if node2 in self.edges and node1 in self.edges[node2]:
            del self.edges[node2][node1]
    
    def remove_node(self, node_id):
        """Remove a node and all its edges"""
        if node_id in self.nodes:
            # Remove all edges to this node
            for neighbor in list(self.edges[node_id].keys()):
                self.remove_edge(node_id, neighbor)
            
            # Remove the node itself
            del self.nodes[node_id]
            del self.edges[node_id]
    
    def get_neighbors(self, node_id):
        """Get all neighbors of a node"""
        return list(self.edges.get(node_id, {}).keys())
    
    def get_edge_weight(self, node1, node2):
        """Get weight of edge between two nodes"""
        return self.edges.get(node1, {}).get(node2, float('inf'))
    
    def has_edge(self, node1, node2):
        """Check if edge exists between two nodes"""
        return node2 in self.edges.get(node1, {})
    
    def get_node_count(self):
        """Get total number of nodes"""
        return len(self.nodes)
    
    def get_edge_count(self):
        """Get total number of edges"""
        count = 0
        for node in self.edges:
            count += len(self.edges[node])
        return count // 2  # Divide by 2 since graph is undirected
    
    def is_connected(self):
        """Check if the graph is connected using BFS"""
        if not self.nodes:
            return True
        
        start_node = next(iter(self.nodes))
        visited = set()
        queue = [start_node]
        visited.add(start_node)
        
        while queue:
            current = queue.pop(0)
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == len(self.nodes)
    
    def get_graph_info(self):
        """Get summary information about the graph"""
        return {
            "nodes": len(self.nodes),
            "edges": self.get_edge_count(),
            "is_connected": self.is_connected(),
            "node_list": list(self.nodes.keys())
        }