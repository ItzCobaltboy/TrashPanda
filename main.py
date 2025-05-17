from fastapi import FastAPI, UploadFile, File, status, HTTPException
import os
import yaml
import shutil
import json
from app.logger import logger
# from app.telemetry import db_log_upload_telemetry, db_log_ping, retrieve_latest_files
from app.preprocessor import validate_trashcan_data, validate_city_map, validate_traffic_data
from app.edgeSelector import EdgeSelector
from app.pathPlanner import PathPlanner
import uuid

# create the FastAPI app instance
app = FastAPI()

################### Load Config ##########################
config_file = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')

def load_config():
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

online_mode = config["database"]["online_mode"]
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
    # latest_city_map, latest_trashcan_data = retrieve_latest_files()
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
    # db_log_upload_telemetry(latest_city_map, latest_trashcan_data)

    return {"INFO": "File uploaded successfully"}


es = None
pp = None

@app.post("/train")  
def train_model():
    # Check if files are present
    if latest_city_map is None or latest_trashcan_data is None:
        log_error("No files found. Please use /upload to upload files.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files found. Please use /upload to upload files.")
    
    # Initilize EdgeSelector
    global es, pp
    es = EdgeSelector(city_map_file=latest_city_map, trashcan_data_file=latest_trashcan_data)
    truth = es.train_models()
    if truth == False:
        log_error("Error in training models. Please check the logs.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="failed to train models. Please check the logs.")
    return {"INFO": "Models trained successfully"}

@app.post("/predict")
def predict_trashcan_status(latest_data_file: UploadFile = File(...)):

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



    selected_cans, predicted_values_cans = es.select_trashcans(latest_data)
    if selected_cans is None or predicted_values_cans is None:
        log_error("Error in selecting trashcans.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error in selecting trashcans.")
    
    selected_edges, edge_rewards = es.select_edges(selected_cans, predicted_values_cans)
    if selected_edges is None or edge_rewards is None:
        log_error("Error in selecting edges.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error in selecting edges.")

    path = pp.path_plan(start=start_location, edge_list=selected_edges, edge_rewards=edge_rewards)

    # handle prediction request for a specific trashcan
    # This is a placeholder for the actual prediction logic

    return path

log_info("FastAPI server started. Listening on port 8000.")
log_info("Endpoints: /city_map, /trashcan_data, /road_data, /train, /predict")