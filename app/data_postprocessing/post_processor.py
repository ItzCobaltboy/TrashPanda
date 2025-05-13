import json
import os

def post_process_data(edges_to_visit):
    """
    Post-process the data to create a JSON file for the client.
    """
    json_output = json.dumps(edges_to_visit, indent=4)
    return json_output

# Testing script
test = {123, 456, 789}

print(post_process_data(test))