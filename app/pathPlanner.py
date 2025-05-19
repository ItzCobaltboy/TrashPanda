import os
import yaml
import networkx as nx
from .preprocessor import GraphHandler
from .logger import logger

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
        self.G = city_map
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
                logger.log_debug(f'Shortest Path for "{source}" to "{target}" added, cost = {lengths[target]}. Path is {path}')
        

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
        Check if all mandatory edges (flagged 2 or 3 in edge_list) are visited at least once 
        outside the segment route[u:v].
        '''
        # Step 1: Get set of mandatory edgeIDs
        mandatory_edges = {
            eid for eid, val in edge_list.items() if val in (2, 3)
        }
        if not mandatory_edges:
            return False  # nothing to check

        # Step 2: Track which edgeIDs are visited outside the skipped segment
        visited_edges = set()

        for i in range(len(route) - 1):
            if u <= i < v :
                continue  # skip the excluded segment

            a, b = route[i], route[i + 1]

            # Check both directions in undirected graph
            if self.graph.has_edge(a, b):
                edgeID = self.graph[a][b].get('edgeID')
            elif self.graph.has_edge(b, a):
                edgeID = self.graph[b][a].get('edgeID')
            else:
                continue

            # if edgeID in mandatory_edges:
            visited_edges.add(edgeID)

        # Step 3: Return True only if all mandatory edges have been visited
        
        Truth = False

        if not mandatory_edges.issubset(visited_edges):
            Truth = True
            # logger.log_debug(f"Mandatory edges {mandatory_edges} visited outside segment {route[u:v]}: {visited_edges}")

        return Truth
    
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
            cost_v_end = self.master_graph_hash.get((v, end_node), None)

            if cost_i_u and cost_v_end:
                cost_option1 = cost_i_u['cost'] + cost_v_end['cost']

                if cost_option1 < min_cost:
                    # Build new route for option 1
                    new_route_option1 = (
                        route[:i+1] +
                        cost_i_u['path'][1:] +  # exclude start_node duplicate
                        [v] +
                        cost_v_end['path'][1:] +
                        route[i+2:]
                    )
                    min_cost = cost_option1
                    best_route = new_route_option1

            # Path costs for detour option 2: i -> v -> u -> i+1
            cost_i_v = self.master_graph_hash.get((start_node, v), None)
            cost_u_end = self.master_graph_hash.get((u, end_node), None)

            if cost_i_v  and cost_u_end:
                cost_option2 = cost_i_v['cost'] + cost_u_end['cost']

                if cost_option2 < min_cost:
                    # Build new route for option 2
                    new_route_option2 = (
                        route[:i+1] +
                        cost_i_v['path'][1:] +
                        [v] +
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
                if len(route) > 2:
                    for k in range(i, j):
                        u1 = route[k]
                        u2 = route[k + 1]
                        original_cost += self.graph[u1][u2]['weight']

                # logger.log_debug(f"Evaluating detour for edge {edgeID} between {route[i]} and {route[j]}: Cost1 = {cost1}, Cost2 = {cost2}, Original Cost = {original_cost}")

                # Check if either path1 or path2 is better than original
                extra1 = cost1 - original_cost
                extra2 = cost2 - original_cost

                if extra1 < 0 or extra2 < 0:
                    logger.log_error(f"Negative cost for detour: {edgeID} between {i} and {j}.")
                    logger.log_error(f"Path1: {path1}, Cost1: {cost1}, Path2: {path2}, Cost2: {cost2}, Original Cost: {original_cost}, Route: {route}")

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
    
    
    def plan_path_mandatory(self, start_node, edge_list):
        if start_node not in self.G:
            raise ValueError(f"Start node '{start_node}' not found in graph.")

        # Step 1: Extract required edges (flagged with 2)
        required_edge_ids = [eid for eid, val in edge_list.items() if val == 2]

        if not required_edge_ids:
            return [start_node], 0  # nothing to visit

        # Step 2: Map edgeID to (u, v)
        edge_id_map = {}
        for u in self.G.nodes:
            for v in self.G.nodes:
                try:
                    edge_id = self.G[u][v].get('edgeID')
                    if edge_id:
                        edge_id_map[edge_id] = (u, v)
                except KeyError:
                    continue
        

        required_edges = set()
        for eid in required_edge_ids:
            if eid not in edge_id_map:
                raise ValueError(f"EdgeID '{eid}' not found in the graph.")
            required_edges.add(edge_id_map[eid])

        print(f"Edge ID map: {required_edges}")
        # Step 3: Greedy routing
        current = start_node
        path = [current]
        total_cost = 0

        while required_edges:
            min_cost = float('inf')
            best_edge = None
            best_entry_path = None

            for u, v in required_edges:
                for target in (u, v):
                    try:
                        cost = self.master_graph_hash[current, target].get('cost', float('inf'))
                        # print(f"Cost from '{current}' to '{target}': {cost}")
                        if cost < min_cost:
                            min_cost = cost

                            if target == u:
                                best_edge = (u, v)
                            else:
                                best_edge = (v, u)
            
                            best_entry_path = self.master_graph_hash[current, target].get('path', [])
                            print(f"Best edge set to '{best_edge}' with cost {min_cost} and path {best_entry_path}")
                    except KeyError:
                        print(f"Path from '{current}' to '{target}' not found in the graph.")
                        continue  # skip unreachable paths

            if best_edge is None:
                raise RuntimeError(f"No path found from '{current}' to any remaining required edge.")

            print(f"Best path from '{current}' to '{target}': {best_entry_path}")
            # Traverse to required edge
            for i in range(1, len(best_entry_path)):
                a, b = best_entry_path[i - 1], best_entry_path[i]
                path.append(b)
                total_cost += self.G[a][b]['weight']

            # Traverse the required edge itself
            u, v = best_edge

            print(f"Traversing edge '{u}' to '{v}'")

            if path[-1] != u:
                path.append(u)
                total_cost += self.G[path[-2]][u]['weight']
            path.append(v)
            total_cost += self.G[u][v]['weight']

            current = v
            x, y = best_edge
            try:
                required_edges.remove((x, y))
            except KeyError:
                required_edges.remove((y, x))
            print(f"Currently Route: {path}")

        # Return to start node
        if current != start_node:
            try:
                return_path = self.master_graph_hash[current, start_node]['path']
                for i in range(1, len(return_path)):
                    a, b = return_path[i - 1], return_path[i]
                    path.append(b)
                    total_cost += self.G[a][b]['weight']
            except KeyError:
                raise RuntimeError(f"Cannot return from '{current}' to start node '{start_node}'.")

        return path, total_cost
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
        scaled_total_cost = total_cost * cost_scaler

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
        route, total_cost = self.plan_path_mandatory(start, edge_list)
        logger.log_debug(f"Route after mandatory edges: {route}, Total Cost: {total_cost}")    

        # Step 2: Add optional edges to the route
        route, scaled_cost, scaled_reward = self.plan_path_optional(route, total_cost, edge_list, edge_rewards)

        return {"route": route, "cost": scaled_cost, "reward": scaled_reward}
        # return route


############################ testing ###########################

# city_map_file = os.path.join(os.path.dirname(__file__), '..', 'tests', 'simulation_small_city', 'city_map.json')

# gh = GraphHandler(city_map_file)

# city_map = gh.Graph

# pp = PathPlanner(city_map)

# edge_listo = {'edge1': 0, 'edge2': 0, 'edge7': 2, 'edge8': 1, 'edge9': 0, 'edge10': 2, 'edge11': 2, 'edge12': 2, 'edge13': 2, 'edge14': 0, 'edge15': 0, 'edge19': 1}
# edge_rewardso = {'edge1': 0, 'edge2': 0, 'edge7': 0, 'edge8': 50, 'edge9': 0, 'edge10': 0, 'edge11': 0, 'edge12': 0, 'edge13': 0, 'edge14': 0, 'edge15': 0, 'edge19': 150}

# print(pp.master_graph_hash['Node2', 'Node1'])



# path = pp.path_plan("Node1", edge_listo, edge_rewardso)

# print(path)

# # print(pp.master_graph_hash)
