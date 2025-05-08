from fastapi import FastAPI, UploadFile, File, status, HTTPException
import os
import yaml
import shutil
from app.logger import log_info, log_warning, log_error, log_debug
from app.telemetry import db_log_launch_telemetry, db_log_ping

# create the FastAPI app instance
app = FastAPI()

# Load configuration from YAML file
def load_config():
    with open("config\config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

# Loading Parameters from config
city_map_url = config["uploads"]["map_upload_dir"]
trashcan_data_url = config["uploads"]["trash_data_dir"]
traffic_data_url = config["uploads"]["traffic_data_dir"]

# Create directories if they do not exist 
if not os.path.exists(city_map_url):
    os.makedirs(city_map_url)
if not os.path.exists(trashcan_data_url):
    os.makedirs(trashcan_data_url)
if not os.path.exists(traffic_data_url):
    os.makedirs(traffic_data_url)

latest_city_map = None
latest_trashcan_data = None
latest_traffic_data = None


# Setup Endpoints
@app.post("/city_map")
def get_city_map(file: UploadFile = File(...)):
    # handle city map request
    if (file.filename.endswith(".json") == False):
        log_error("Invalid file type. Only JSON files are allowed.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Only JSON files are allowed.")
    
    # Save the uploaded file to the specified directory
    # file_path = uploads
    file_path = os.path.join(city_map_url, file.filename)
    db_log_ping(file.filename, "city_map", 200)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)   

    # Now reopen file and parse it and save it to the database
    latest_city_map = file.filename
    # keep record of latest file in the database

    log_info(f"File {file.filename} uploaded successfully to {city_map_url}.")
    return {"INFO": "File uploaded successfully"}

@app.post("/trashcan_data")
async def get_trashcan_data(file: UploadFile = File(...)):
    # handle trashcan data request
    if (file.filename.endswith(".csv") == False):
        log_error("Invalid file type. Only CSV files are allowed.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Only CSV files are allowed.")
    
    # Save the uploaded file to the specified directory
    # file_path = uploads
    file_path = os.path.join(trashcan_data_url, file.filename)
    db_log_ping(file.filename, "trashcan_data", 200)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Now reopen file and parse it and save it to the database
    latest_trashcan_data = file.filename
    # keep record of latest file in the database


    log_info(f"File {file.filename} uploaded successfully to {trashcan_data_url}.")
    return {"INFO": "File uploaded successfully"}

@app.post("/road_data")
def get_road_data(file: UploadFile = File(...)):    
    # handle road data request
    if (file.filename.endswith(".csv") == False):
        log_error("Invalid file type. Only CSV files are allowed.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Only CSV files are allowed.")

    # Save the uploaded file to the specified directory
    # file_path = uploads
    file_path = os.path.join(traffic_data_url, file.filename)
    db_log_ping(file.filename, "road_data", 200)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


    # Now reopen file and parse it and save it to the database
    latest_traffic_data = file.filename
    # keep record of latest file in the database

    log_info(f"File {file.filename} uploaded successfully to {traffic_data_url}.")
    return {"INFO": "File uploaded successfully"}

@app.post("/train")  
def train_model():
    # handle model training request
    # Record entry for kicking in training model
    db_log_launch_telemetry(latest_city_map, latest_trashcan_data, latest_traffic_data)
    log_info("Training model with the latest data files.")

    # stop the current model, refresh the variables and start training
    # This is a placeholder for the actual training logic

    return {"INFO": "Model training started."}

@app.post("/predict")
def predict_trashcan_status():
    # handle prediction request for a specific trashcan
    # This is a placeholder for the actual prediction logic

    return

log_info("FastAPI server started. Listening on port 8000.")
log_info("Endpoints: /city_map, /trashcan_data, /road_data, /train, /predict")