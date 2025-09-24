from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import pickle, json
import joblib

#base path
BASE = Path(__file__).resolve().parent.parent


#paths
FLIGHT_MODEL_PATH = BASE/ "models"/ "flight_price_pipeline.pkl"
FLIGHT_FEATURES_PATH = BASE/ "config" / "flight_columns_info.json"

HOTEL_MODEL_PATH = BASE/ "models"/ "hotel_price_pipeline.pkl"
HOTEL_FEATURES_PATH = BASE/ "config" / "hotels_columns_info.json"


#def load pickle and json
def load_pickle(path):
    with open(path, "rb") as f:
        return joblib.load(f)

def load_json(path):
    with open (path, "r") as f:
        return json.load(f)


# try excpet
try:
    flight_model = load_pickle(FLIGHT_MODEL_PATH)
except FileNotFoundError:
    flight_model = None

try:
    hotel_model = load_pickle(HOTEL_MODEL_PATH)
except FileNotFoundError:
    hotel_model = None

try:
    flight_features = load_json(FLIGHT_FEATURES_PATH)
except FileNotFoundError:
    flight_features = []

try:
    hotel_features = load_json(HOTEL_FEATURES_PATH)
except FileNotFoundError:
    hotel_features = []



#initialize
app = FastAPI(title="Travel Prediction API")


#Allow frontend calls

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods
    allow_headers=["*"]   # allow all headers
)


#request model

class PredictRequest(BaseModel):
    data : dict



def build_input_vector(data: dict, features: list):
    missing = [f for f in features if f not in data]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return [data[f] for f in features]
    

@app.post("/predict/flight")
def predict_flight(req: PredictRequest):
    if flight_model is None:
        raise HTTPException(500, "Flights model not found.")
    try:
        X = build_input_vector(req.data, flight_features)
        pred = flight_model.predict([X])
        return {"prediction":float(pred[0])}
    except Exception as e:
        raise HTTPException(400, str(e))
    
@app.post("/predict/hotels")
def predict_hotels(req: PredictRequest):
    if hotel_model is None:
        raise HTTPException(500, "Hotels model not found.")
    try:
        X = build_input_vector(req.data, hotel_features)
        pred = hotel_model.predict([X])
        return {"prediction":float(pred[0])}
    except Exception as e:
        raise HTTPException(400, str(e))
    

    
