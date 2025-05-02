# TrashPanda
A Repository for an ML based solution to Trash Truck routing for real world applications. A Custom model defined on Tensorflow for accurate prediction of Trash Truck's routing for a city, depending on parameters of how fast the trashcans at each street fill up and traffic, such that the operating costs for the company is minimal.

The software is built to be ready to deploy with API endpoints built using FastAPI and telemetry implemented using PostgreSQL.

The repository also contains some sample input data for testing the system
<br>

# Core requirements

- Python 3.10.x (otherwise Tensorflow will break)
- Tensorflow 2.12
- FastAPI
- Numpy/Pandas
- PostgreSQL for database

#### Dev Tools
- Postman (for pinging)

<br>

# Installation from Source
## Clone Repository and Setup virtual environment

```
git clone <repository_link>
cd TrashPanda
python -m venv venv
venv\Scripts\activate
```
-- Replace `<repository_link>` with the correct link
-- use `deactivate` command to deactivate Virtual envirnment

## Dependancies

```
pip install -r requirements.txt
```
This will install all the required dependancies 

# Features

## API Endpoints

1) `/city_map`: POST upload `.json` file for city map as value for 'file' attribute in body, will be stored in `uploads` folder of the system locally and parsed later on.
2) `/trashcan_data`: POST upload `.csv` file for city map as value for 'file'.
3) `/traffic_data`: POST upload `.csv` fi;e for city map as value for 'file'.
4) `/train`: ping this with a GET request to trigger stopping of current model, resetting the weights and execute training process on latest available city_map, trashcan and traffic data.
5) `/predict`: ping this with current traffic and trashcan data as json value to get output from trained model, this data will be logged.

## General

- all the input files are stored in `/uploads` for future reference.
- The inputs are logged into database, currently PostgreSQL.
- Dynamic retraining process.
- All the requests are logged, including `/predict` pings, so the model can expand on the training set time to time
- Runtime logs can be found in `/logs`.
