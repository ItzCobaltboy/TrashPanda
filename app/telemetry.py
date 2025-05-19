# Handler for Telemetry events
import os
import yaml
import psycopg2
from .logger import logger

# Load config
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


config = load_config()
online_mode = config["database"]["online_mode"]


# setup logger
logger = logger()
log_info = logger.log_info
log_error = logger.log_error
logger.user = "Telemetry"

# Load parameters
DB_USER = config["database"]["db_username"]
DB_PASSWORD = config["database"]["db_password"]
DB_HOST = config["database"]["db_host"]
DB_PORT = config["database"]["db_port"]
DB_NAME = config["database"]["db_name"]

if online_mode:
    try:
        # Connect to database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cur = conn.cursor()
        log_info("Connected to the database successfully at IP: " + DB_HOST + " and Port: " + str(DB_PORT))

    except Exception as e:
        log_error(f"Error connecting to the database: {e}")
        conn = None

    # Create Master Table for Launch events, tagging all the individual launch events
    try:
        cur.execute('''
        CREATE TABLE IF NOT EXISTS telemetry_events (
            id SERIAL PRIMARY KEY,
            city_map_file TEXT,
            trashcan_data_file TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
        conn.commit()
        log_info("Master events table connected successfully.")
    except Exception as e:
        log_error(f"Error creating master_events table: {e}")
        conn.rollback()


    # Create Table for recording pings
    # will record the pings from the clients

    try:
        cur.execute('''
        CREATE TABLE IF NOT EXISTS pings (
            id SERIAL PRIMARY KEY,
            client_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            endpoint TEXT,
            event_data TEXT,
            status_code INTEGER    )
    ''')
        conn.commit()
        log_info("Pings table connected successfully.")
    except Exception as e:
        log_error(f"Error creating pings table: {e}")
        conn.rollback()


    def db_log_upload_telemetry(city_map_file, trashcan_data_file):
        """
        Log telemetry data for the launch event.
        """
        try:
            # Insert telemetry data into the database
            cur.execute('''
                INSERT INTO telemetry_events (city_map_file, trashcan_data_file)
                VALUES (%s, %s, %s)
            ''', (city_map_file, trashcan_data_file))
            conn.commit()
            log_info(f"New City Set loaded: {city_map_file}, {trashcan_data_file}")
        except Exception as e:
            log_error(f"Error logging telemetry data: {e}")
            conn.rollback()  # Rollback in case of error

    def db_log_ping(client_id, endpoint, event_data, status_code):
        '''
        Log telemetry data for the ping event.
        
        '''
        try:
            # Insert telemetry data into the database
            cur.execute('''
                INSERT INTO pings (client_id, endpoint, event_data, status_code)
                VALUES (%s, %s, %s, %s)
            ''', (client_id, endpoint, event_data, status_code))
            conn.commit()
            log_info( f"New ping event logged: {client_id}, {endpoint}, {event_data}, {status_code}")
        except Exception as e:
            log_error( f"Error logging ping data: {e}")
            conn.rollback()  # Rollback in case of error

    def retrieve_latest_files():
        """
        Retrieve the latest files from the database.
        """
        try:
            cur.execute('''
                SELECT city_map_file, trashcan_data_file
                FROM telemetry_events
                ORDER BY timestamp DESC
                LIMIT 1
            ''')
            result = cur.fetchone()
            if result:
                city_map_file, trashcan_data_file = result
                log_info(f"Latest files retrieved: {city_map_file}, {trashcan_data_file}")
                return city_map_file, trashcan_data_file
            else:
                log_info("No telemetry data found.")
                return None, None
        except Exception as e:
            log_error(f"Error retrieving latest files: {e}")
            return None, None
else:

    log_info("Running in offline mode. No database connection established.")
    def db_log_upload_telemetry(city_map_file, trashcan_data_file):
        logger.log_debug("Cannot Log Launch Telemetry in offline mode.")
        return
    def db_log_ping(client_id, endpoint, event_data, status_code):
        logger.log_debug("Cannot Log Ping Telemetry in offline mode.") 
        return
    def retrieve_latest_files():
        return None, None