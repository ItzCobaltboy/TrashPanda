from fastapi import FastAPI, Form, UploadFile, File, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
import os
import yaml
import shutil
import json
from app.logger import logger
from app.telemetry import db_log_upload_telemetry, retrieve_latest_files
from app.preprocessor import validate_trashcan_data, validate_city_map, validate_traffic_data
from app.edgeSelector import EdgeSelector
from app.pathPlanner import PathPlanner
import uuid
import uvicorn
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# create the FastAPI app instance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

################### Load Config ##########################
config_file = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')

def load_config():
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

online_mode = config["database"]["online_mode"]
app_version = "v1.0.0-alpha"

host = config["server"]["host"]
port = config["server"]["port"]
visualise = config["server"]["visualise"]

##########################################################

##################### Setup Logger #######################
logger = logger()
log_info = logger.log_info  
log_error = logger.log_error
logger.user = "Main"
###########################################################

# Loading Parameters from config
city_map_url = config["uploads"]["map_upload_dir"]
trashcan_data_url = config["uploads"]["trash_data_dir"]
traffic_data_url = config["uploads"]["traffic_data_dir"]

city_map_url = os.path.join(os.path.dirname(__file__), "uploads", city_map_url)
trashcan_data_url = os.path.join(os.path.dirname(__file__), "uploads", trashcan_data_url)
traffic_data_url = os.path.join(os.path.dirname(__file__), "uploads", traffic_data_url)

# Create directories if they do not exist 
if not os.path.exists(city_map_url):
    logger.log_debug(f"Creating directory: {city_map_url}")
    os.makedirs(city_map_url)
if not os.path.exists(trashcan_data_url):
    logger.log_debug(f"Creating directory: {trashcan_data_url}")
    os.makedirs(trashcan_data_url)
if not os.path.exists(traffic_data_url):
    logger.log_debug(f"Creating directory: {traffic_data_url}")
    os.makedirs(traffic_data_url)


# Initialize global variables
latest_city_map = None
latest_trashcan_data = None

# If database mode is online, query the latest files from the database
if online_mode == True:
    latest_city_map, latest_trashcan_data = retrieve_latest_files()
    if latest_city_map is None or latest_trashcan_data is None:
        logger.log_warning("No Latest Files found in the database")
        latest_city_map = None
        latest_trashcan_data = None
    else:
        logger.log_info(f"Last used files retreived from database: {latest_city_map}, {latest_trashcan_data}")
        logger.log_info(f"Using these files to train the system")

################################ Setup Endpoints ##################################
@app.post("/upload")
def upload_files(city_map: UploadFile = File(...), trashcan_data: UploadFile = File(...)):

    random_id = str(uuid.uuid4())
    city_file = os.path.join(city_map_url, f"city_map_{random_id}.json")
    trash_file = os.path.join(trashcan_data_url, f"trashcan_data_{random_id}.csv")

    city_file_name = os.path.basename(city_file)
    trash_file_name = os.path.basename(trash_file)

    logger.log_debug(f"Recieved files: {city_map.filename}, {trashcan_data.filename}")
    logger.log_debug(f"Saving files with filename: city_map_{random_id}.json, trashcan_data_{random_id}.csv")
    # Store both files in respective directories after renaming to something unique
     # Save city_map
    with open(city_file, "wb") as f:
        shutil.copyfileobj(city_map.file, f)

    # Save trashcan_data
    with open(trash_file, "wb") as f:
        shutil.copyfileobj(trashcan_data.file, f)

    # Validate city_map
    if not city_map.filename.endswith(".json"):
        log_error("Invalid file type. Only JSON files are allowed.")
        logger.log_debug(f"File name: {city_map.filename}")
        os.delete(city_file)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. JSON FIles expected")
    
    if validate_city_map(city_file_name) == False:
        log_error("Invalid city map structure.")
        os.delete(city_file)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid city map structure. Some edges are not present in map")
    
    # Validate Trashcan_data
    if not trashcan_data.filename.endswith(".csv"):
        log_error("Invalid file type. Only CSV files are allowed.")
        logger.log_debug(f"File name: {trashcan_data.filename}")
        os.remove(trash_file)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. CSV FIles expected")

    if validate_trashcan_data(trashcan_data_file=trash_file_name, city_map_file=city_file_name) == False:
        log_error("Invalid trashcan structure.")
        os.remove(trash_file)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid trashcan structure. Some edges are not present in map")
    


    # Otherwise save the files for next call
    global latest_city_map, latest_trashcan_data

    latest_city_map = f"city_map_{random_id}.json"
    latest_trashcan_data = f"trashcan_data_{random_id}.csv"
    db_log_upload_telemetry(latest_city_map, latest_trashcan_data)

    return JSONResponse(content={"INFO": "Files uploaded Successfully"})


es = None
pp = None

@app.get("/train")  
def train_model():
    # Check if files are present
    if latest_city_map is None or latest_trashcan_data is None:
        log_error("No files found. Please use /upload to upload files.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files found. Please use /upload to upload files.")
    
    # Initilize EdgeSelector
    global es, pp
    es = EdgeSelector(city_map_file=latest_city_map, trashcan_data_file=latest_trashcan_data)

    try:
        truth = es.train_models()
    except Exception as e:
        log_error(f"Error in training models: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"failed to train models. {e}")

    return JSONResponse(content = {"INFO": "Models trained successfully", "training_time": truth})

@app.post("/predict")
def predict_trashcan_status(start_node : str = Form(...),day_name: str = Form(...),latest_data_file: UploadFile = File(...)):

    start_location = "Node1"
    # Validate Start Location

    # Validate the file
    if not latest_data_file.filename.endswith(".json"):
        log_error("Invalid file type. Only JSON files are allowed.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. JSON Files expected")
    
    latest_data = json.loads(latest_data_file.file.read())
    
    if es is None:
        log_error("EdgeSelector not initialized. Please train the model first.")
        raise HTTPException(status_code=status.HTTP_428_PRECONDITION_REQUIRED, detail="EdgeSelector not initialized. Please train the model first.")
    else:
        if not es.validate_latest_data(latest_data):
            log_error("Invalid latest data structure.")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid latest data structure. Some trashcanID: fill% values aare not present")
    global pp
    # Initialize PathPlanner
    Graph = es.GraphHandler.Graph
    if pp is None:
        pp = PathPlanner(city_map=Graph)

    if start_location is None:
        log_error("Please provide a start location")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please provide a start location")
    if str(start_location) not in Graph.nodes:
        log_error("Invalid start location. Please check the city map.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid start location. Please check the city map.")



    selected_cans, predicted_values_cans = es.select_trashcans(latest_trashcan_data=latest_data, day_name= day_name)
    if selected_cans is None or predicted_values_cans is None:
        log_error("Error in selecting trashcans.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error in selecting trashcans.")
    
    logger.log_debug(f"Selected trashcans: {selected_cans}")
    logger.log_debug(f"Predicted values: {predicted_values_cans}")

    selected_edges, edge_rewards = es.select_edges(selected_cans, predicted_values_cans)
    if selected_edges is None or edge_rewards is None:
        log_error("Error in selecting edges.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error in selecting edges.")
    
    logger.log_debug(f"Selected edges: {selected_edges}")
    logger.log_debug(f"Edge rewards: {edge_rewards}")

    try:
        path = pp.path_plan(start=start_location, edge_list=selected_edges, edge_rewards=edge_rewards)
    except Exception as e:
        log_error(f"Error in path planning: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error in path planning: {e}")
    # handle prediction request for a specific trashcan
    # This is a placeholder for the actual prediction logic
    logger.log_debug(f"Path planned: {path}")
    if visualise:
        draw_graph_with_route_and_trash(Graph, path['route'], latest_trashcan_data, route_color='red', default_color='lightgray')
    return path

@app.get("/logs")
def get_logs():
    # Return the path to the log file
    log_file_path = logger.retrieve_log()
    log_file_path = os.path.join(os.path.dirname(__file__), log_file_path)

    print(f"Log file path: {log_file_path}")

    if os.path.exists(log_file_path):
        return FileResponse(log_file_path, filename="latest_log.txt")
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log file not found")



if __name__ == "__main__":

    log_info("\n\n" \
        "========================================================\n\n" \
        f"Welcome to TrashPanda! {app_version}\n" \
        "This is a Trash Collection Route Optimization System, Please read documentation for usage and working\n" \
        "This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. Please find LICENSE file in the root directory.\n\n" \
        
        "Github Link: https://github.com/ItzCobaltboy/TrashPanda/tree/main \n\n"\
        "========================================================\n")



    log_info(f"FastAPI server started. Listening on port {port}.")
    log_info("Endpoints: /upload, /train, /predict, /logs")


    uvicorn.run("main:app", host=host, port=port)

    log_info("\n\n" \
        "========================================================\n\n" \
        f"Thanks for using TrashPanda! {app_version}\n" \
        "System shutdown successfully... \n\n"\
        "========================================================\n")


def draw_graph_with_route_and_trash(G, route_nodes, trashcan_csv_path, route_color='red', default_color='lightgray'):
# Step 1: Generate list of edges in the route (pairs of consecutive nodes)
    route_edges = [(route_nodes[i], route_nodes[i+1]) for i in range(len(route_nodes) - 1)]
    trashcan_csv_path = os.path.join(os.path.dirname(__file__), 'uploads',trashcan_data_url, trashcan_csv_path)
    # Step 2: Read CSV and create edgeID → TrashcanID mapping
    df = pd.read_csv(trashcan_csv_path)
    edge_to_trashcan = dict(zip(df['edgeID'], df['trashcanID']))

    # Step 3: Assign edge colors and labels
    edge_colors = []
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        edge_id = data.get('edgeID', None)
        trashcan_id = edge_to_trashcan.get(edge_id, 'N/A')
        label = f"ID: {edge_id}\nCan: {trashcan_id}"

        edge_labels[(u, v)] = label
        if (u, v) in route_edges or (v, u) in route_edges:
            edge_colors.append(route_color)
        else:
            edge_colors.append(default_color)

    # Step 4: Draw the graph
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color=edge_colors, width=2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Graph with Highlighted Route and Trashcan Mapping")
    plt.tight_layout()
    plt.show()