import os
import yaml
import networkx as nx
from logger import logger
import matplotlib.pyplot as plt
from preprocessor import GraphHandler
import random

# import config
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# configs go here (rn NONE)


# setup logger
logger = logger()
logger.user = "PathPlanner"
logger.log_info("PathPlanner initialized.")


class PathPlanner:
    def __init__(self, city_map=nx.graph):
        
        # Record the data, inputed to pathPlanner
        self.graph = city_map
        self.nodes = list(city_map.nodes)
        self.master_graph_hash = {}
        logger.log_info(f"Currently using {city_map}")


        # preprocess hash
        self.preprocess_map()

    def preprocess_map(self):
        # Generate all possible combinations of short node paths in map and store then

        for source in self.nodes:
            lengths, paths = nx.single_source_dijkstra(self.graph, source, weight='weight')
            for target, path in paths.items():
                self.master_graph_hash[(source, target)] = {
                    "path": path,
                    "cost": lengths[target]
                }
                logger.log_debug(f'Shortest Path for "{source}" to "{target}" added, cost = {lengths[target]}.')
        

        logger.log_info("Hash for shortest routes created.")
        # make a dict of all possible shortest path combos for each node combination

    def get_shortest_path(self, start, goal):
        """
        Return cached shortest path and its cost as a dict:
        {
            "path": [...],
            "cost": ...
        }
        """
        return self.master_graph_hash.get((start, goal), {"path": None, "cost": float('inf')})

    def get_edge_by_ID(self, edgeID):
        for u, v, data in self.graph.edges(data=True):
            if data['edgeID'] == edgeID:
                return u, v
        return None
    
    def add_edge_to_route(self, u, v, route):
        """
        Try inserting edge (u, v) into route between consecutive nodes i and i+1 with minimal cost.
        
        Returns:
            new_route: list of nodes with detour inserted
            detour_cost: cost added by detour
        """
        min_cost = float('inf')
        best_route = None

        # Iterate through consecutive pairs in route
        for i in range(len(route) - 1):
            start_node = route[i]
            end_node = route[i + 1]

            # Path costs for detour option 1: i -> u -> v -> i+1
            cost_i_u = self.master_graph_hash.get((start_node, u), None)
            cost_u_v = self.master_graph_hash.get((u, v), None)
            cost_v_end = self.master_graph_hash.get((v, end_node), None)

            if cost_i_u and cost_u_v and cost_v_end:
                cost_option1 = cost_i_u['cost'] + cost_u_v['cost'] + cost_v_end['cost']

                if cost_option1 < min_cost:
                    # Build new route for option 1
                    new_route_option1 = (
                        route[:i+1] +
                        cost_i_u['path'][1:] +  # exclude start_node duplicate
                        cost_u_v['path'][1:] +
                        cost_v_end['path'][1:] +
                        route[i+2:]
                    )
                    min_cost = cost_option1
                    best_route = new_route_option1

            # Path costs for detour option 2: i -> v -> u -> i+1
            cost_i_v = self.master_graph_hash.get((start_node, v), None)
            cost_v_u = self.master_graph_hash.get((v, u), None)
            cost_u_end = self.master_graph_hash.get((u, end_node), None)

            if cost_i_v and cost_v_u and cost_u_end:
                cost_option2 = cost_i_v['cost'] + cost_v_u['cost'] + cost_u_end['cost']

                if cost_option2 < min_cost:
                    # Build new route for option 2
                    new_route_option2 = (
                        route[:i+1] +
                        cost_i_v['path'][1:] +
                        cost_v_u['path'][1:] +
                        cost_u_end['path'][1:] +
                        route[i+2:]
                    )
                    min_cost = cost_option2
                    best_route = new_route_option2

        if best_route is None:
            # If no insertion found, just return original route and 0 cost increment
            return route, 0

        return best_route, min_cost


    def plan_path_mandatory(self, start, edge_list = {}, edge_rewards = {}):
        '''
        Go thru all the dict's edges and select mandatory visit and optional visit edges
        '''

        visited_edges = set()
        visited_nodes = set()

        total_cost = 0

        route = [start,start]

        # Find and add mandatory edges to graph
        must_visit_edges = {edge: edge_list[edge] for edge in edge_list if edge_list[edge] == 2}

        for edgeID in must_visit_edges:
            u, v = self.get_edge_by_ID(edgeID)
            if u is None or v is None:
                logger.log_error(f"Edge {edgeID} not found in graph.")
                continue

            # Add edge to visited edges
            visited_edges.add(edgeID)
            visited_nodes.add(u)
            visited_nodes.add(v)
            
            # Add the edge to the route
            route, cost_to_add = self.add_edge_to_route(u, v, route)
            total_cost += cost_to_add

        return {"route": route, "cost": total_cost}




# Create a 6-node undirected graph
G = nx.Graph()


random.seed(42)
# Add edges with random weights and edgeID
# Edges: 13 edges, manually defined
# Create empty graph
G = nx.Graph()

# Define 12 nodes
nodes = list(range(12))
G.add_nodes_from(nodes)

# Define 16 sparse edges (no full connectivity)
edges = [
    (0, 1), (0, 4), (1, 2), (1, 5),
    (2, 3), (3, 6), (4, 5), (4, 8),
    (5, 6), (6, 7), (7, 10), (8, 9),
    (9, 10), (10, 11), (3, 11), (2, 9)
]

# Add edges with weights and edgeIDs
for idx, (u, v) in enumerate(edges):
    G.add_edge(u, v, weight=random.randint(1, 10), edgeID=f"e{idx}")

# Print edge details
print("Sparse Graph (12 nodes, 16 edges):")
for u, v, data in G.edges(data=True):
    print(f"{u}-{v}: weight={data['weight']}, edgeID={data['edgeID']}")


selection_dict = {
    'e0': 2,  # must visit
    'e2': 2,
    'e4': 1,
    'e10': 1,
    'e12': 2
}

reward_dict = {
    'e4': 4.0,
    'e10': 5.5
}

# Define the planner class
planner = PathPlanner(G)


print(planner.get_shortest_path(10, 0))
plt.show()

result = planner.plan_path_mandatory(start=0, edge_list=selection_dict, edge_rewards=reward_dict)


print("\n=== Final Route Output ===")
print("Route:", result["route"])
print("Total Cost:", result["cost"])

nx.draw(G, with_labels=True)
plt.show()