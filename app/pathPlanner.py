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
    
    def route_has_mandatory_edges(self, u, v, route , edge_list):
        '''
        Check if route has mandatory edges
        '''
        for i in range(u, v):
            edgeID = self.graph[route[i]][route[i+1]]['edgeID']
            if edgeID in edge_list and edge_list[edgeID] == 2 or edge_list[edgeID] == 3:
                return True
            
        return False
    
    def add_edge_mandatory(self, u, v, route):
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
            

    def add_optional_edge(self, edgeID, route, edge_list):
        """
        Try inserting an optional edge as a detour in the existing route.
        Only insert if it doesn't skip any mandatory edges in the segment it replaces.
        Returns updated route if insertion is successful, else returns the original route.
        """
        u, v = self.get_edge_by_ID(edgeID)
        if u is None or v is None:
            print(f"EdgeID {edgeID} not found.")
            return route

        best_new_route = route
        min_extra_cost = float('inf')

        # Loop through all i, j pairs in route where i < j
        for i in range(len(route)):
            for j in range(i + 1, len(route)):
                start = route[i]
                end = route[j]

                # Check if inserting u-v between i and j skips mandatory edges
                if self.route_has_mandatory_edges(i, j, route, edge_list):
                    # logger.log_debug(f"Skipping detour for edge {edgeID} between {route[i]} and {route[j]} due to mandatory edges.")
                    continue

                # Generate detour via u-v in both possible ways
                path1 = self.get_shortest_path(start, u)["path"] + [v] + self.get_shortest_path(v, end)["path"][1:]
                cost1 = self.get_shortest_path(start, u)["cost"] + self.graph[u][v]['weight'] + self.get_shortest_path(v, end)["cost"]

                path2 = self.get_shortest_path(start, v)["path"] + [u] + self.get_shortest_path(u, end)["path"][1:]
                cost2 = self.get_shortest_path(start, v)["cost"] + self.graph[u][v]['weight'] + self.get_shortest_path(u, end)["cost"]

                original_cost = 0
                for k in range(i, j):
                    u1 = route[k]
                    u2 = route[k + 1]
                    original_cost += self.graph[u1][u2]['weight']

                # logger.log_debug(f"Evaluating detour for edge {edgeID} between {route[i]} and {route[j]}: Cost1 = {cost1}, Cost2 = {cost2}, Original Cost = {original_cost}")

                # Check if either path1 or path2 is better than original
                extra1 = cost1 - original_cost
                extra2 = cost2 - original_cost

                if extra1 < min_extra_cost:
                    min_extra_cost = extra1
                    best_segment = (i, j)
                    best_path = path1

                if extra2 < min_extra_cost:
                    min_extra_cost = extra2
                    best_segment = (i, j)
                    best_path = path2

        # Replace segment i to j in original route if improvement found
        if min_extra_cost < float('inf'):
            i, j = best_segment
            new_route = route[:i] + best_path + route[j+1:]
            return new_route, min_extra_cost

        return route, 0
    
    
    def plan_path_mandatory(self, start, edge_list):
        '''
        Go thru all the dict's edges and select mandatory visit and optional visit edges
        '''
        # Configurable scalers
        cost_scaler = config["path_planning"]["cost_scaler"]    # set from config

        total_cost = 0
        route = [start,start]

        # Find and add mandatory edges to graph
        must_visit_edges = {edge: edge_list[edge] for edge in edge_list if edge_list[edge] == 2}

        for edgeID in must_visit_edges:
            u, v = self.get_edge_by_ID(edgeID)
            if u is None or v is None:
                logger.log_error(f"Edge {edgeID} not found in graph.")
                continue

            
            # Add the edge to the route
            route, cost_to_add = self.add_edge_mandatory(u, v, route)
            total_cost += cost_to_add* cost_scaler
            logger.log_info(f"Added mandatory edge {edgeID} to route: {route}, Scaled Cost = {cost_to_add * cost_scaler}")

        return {"route": route, "cost": total_cost*cost_scaler}
    
    def plan_path_optional(self, route, total_cost, edge_list, edge_rewards):
        '''
        Add optional edges to the route if reward outweighs detour cost.
        All optional edges are evaluated in one round.
        '''
        optional_edges = {edge: edge_list[edge] for edge in edge_list if edge_list[edge] == 1}

        # Configurable scalers
        reward_scaler = config["path_planning"]["reward_scaler"]  # set from config
        cost_scaler = config["path_planning"]["cost_scaler"]    # set from config

        scaled_total_reward = 0
        scaled_total_cost = 0

        for optional_edge in optional_edges:
            reward = edge_rewards.get(optional_edge, 0)

            new_route, cost_diff = self.add_optional_edge(optional_edge, route, edge_list)
            logger.log_debug(f"Evaluating optional edge {optional_edge}: (Scaled) Reward = {reward*reward_scaler}, (Scaled) Extra Cost = {cost_diff*cost_scaler}")

            # If adding the edge is beneficial
            if new_route != route and reward_scaler * reward > cost_scaler * cost_diff:
                logger.log_info(
                    f"Added optional edge {optional_edge}: Scaled_Reward = {reward * reward_scaler}, Scaled_Extra Cost = {cost_diff * cost_scaler}"
                )
                logger.log_debug(f"Old route: {route}")
                logger.log_debug(f"New route: {new_route}")

                # Add the newly added optional edge so it doesn't get removed in the next iteration
                edge_list[optional_edge] = 3 # 3 = added optional edge
                route = new_route  # Accept this route with the optional edge
                total_cost += cost_diff 
                scaled_total_reward += reward_scaler * reward
                scaled_total_cost += cost_scaler * cost_diff
        return route, scaled_total_cost, scaled_total_reward


    def path_plan (self, start, edge_list, edge_rewards):
        '''
        Plan the path
        '''
        # Step 1: Add mandatory edges to the route
        result = self.plan_path_mandatory(start, edge_list)
        route = result["route"]
        total_cost = result["cost"]

        # Step 2: Add optional edges to the route
        route, scaled_cost, scaled_reward = self.plan_path_optional(route, total_cost, edge_list, edge_rewards)

        return {"route": route, "cost": scaled_cost, "reward": scaled_reward}


####################### SAMPLE TESTCASE #######################
""" 


# ---- Hardcoded Graph Setup ----
G = nx.Graph()

random.seed(42)
edges = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 3),
    (2, 4), (3, 4), (4, 5), (5, 0), (3, 5),
    (5, 11), (6, 7), (7, 8), (8, 9), (9, 10),
    (10, 11), (11, 6)
]

for idx, (u, v) in enumerate(edges):
    G.add_edge(u, v, weight=random.randint(1, 10), edgeID=f"e{idx}")

print("Graph edges and attributes:")
for u, v, data in G.edges(data=True):
    print(f"{u}-{v}: {data}")

# ---- Selection & Rewards ----
selection_dict = {
    'e0': 2,  # mandatory
    'e1': 0,
    'e2': 2,  # mandatory
    'e3': 0,
    'e4': 1,  # optional
    'e5': 1,  # optional
    'e6': 0,
    'e7': 0,
    'e8': 0,
    'e9': 0,
    'e10': 0,
    'e11': 0,
    'e12': 0,
    'e13': 2,
    'e14': 0,
    'e15': 0,
    'e16': 1,
    'e17': 0

}

reward_dict = {
    'e4': 15.0,
    'e5': 30.0,
    'e16': 20.0,
}

# ---- Visualize the Graph (Optional) ----
pos = nx.spring_layout(G, seed=42)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Graph with Edge IDs")

# ---- Initialize and Run PathPlanner ----
planner = PathPlanner(G)

# Full plan: includes mandatory + optional evaluation
final_result = planner.path_plan(start=0, edge_list=selection_dict, edge_rewards=reward_dict)

# ---- Output Result ----
print("\n=== Final Route Output ===")
print("Planned Route:", final_result["route"])
print("Total Cost:", final_result["cost"])
print("Total Reward:", final_result["reward"])
plt.show()


"""