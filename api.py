from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator
import joblib
from ml_soilModel import get_soil_analysis
from monthly_data import get_last_30_day_weather
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import requests

load_dotenv()

#price prediction API configuration
API_KEY = os.getenv("API_KEY")
RESOURCE_ID = os.getenv("RESOURCE_ID")
BASE_URL = f'https://api.data.gov.in/resource/{RESOURCE_ID}'


app = FastAPI()
# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Replace "*" with the frontend URL for better security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the trained models and dataset
try:
    trained_crop_model = joblib.load("trained_models/crop_recommendation_model.pkl")
    label_encoder = joblib.load("trained_models/label_encoder.pkl")
    area_model = joblib.load("trained_models/area_model.pkl")
    prod_model = joblib.load("trained_models/production_model.pkl")
    encoder = joblib.load("trained_models/encoder.pkl")
   
    df = pd.read_csv("Datasets/Crop_recommendation.csv").dropna()
    fertilizer = pd.read_csv("Datasets/fertilizer_recommendation.csv").dropna()
    fertilizer.columns = fertilizer.columns.str.strip()
    fertilizer['Crop Type'] = fertilizer['Crop Type'].str.strip().str.lower()
    fertilizer['Fertilizer Name'] = fertilizer['Fertilizer Name'].str.strip()
except Exception as e:
    raise RuntimeError(f"Failed to load resources: {e}")



# Define the request schema
class CropRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    rainfall: float

class CropDetailRequest(BaseModel):
    crop : str
    
class CityRequest(BaseModel):
    city: str 


class PriceRequest(BaseModel):
    state: str
    commodity: str
    limit: int = 10

    @field_validator("commodity", mode="before")
    @classmethod
    def capitalize_commodity(cls, v):
        return v.capitalize()
    
class Prod_Input(BaseModel):
    state_name: str
    district_name: str
    crop: str
    season: str
# Request schema end


# Setting up the home API
@app.get("/")
def working():
    return { "Working"}

# API to predict crops, works using the request taken
@app.post("/predict_crop")
def predict_crop(request: CropRequest):
    """
    Predict the top crops based on soil and climate features.
    """
    top_n = 6

    # Prepare input data
    input_df = pd.DataFrame([[
        request.N,
        request.P,
        request.K,
        request.temperature,
        request.humidity,
        request.rainfall
    ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'rainfall'])

    # Predict probabilities
    probas = trained_crop_model.predict_proba(input_df)[0]

    # Get top N crop indices
    top_indices = np.argsort(probas)[-top_n:][::-1]
    
    # Get crop names and probabilities
    top_crops = label_encoder.inverse_transform(top_indices)
    top_probabilities = probas[top_indices]

    # Return results
    return [{"crop": crop, "probability": float(f"{prob:.2f}")} for crop, prob in zip(top_crops, top_probabilities)]


# Getting the required Crop details, requests taken : State name and Crop name
@app.post("/crop_details")
def crop_details(request: CropDetailRequest):
    """
    Fetch detailed information for a specific crop.
    """
    crop = request.crop.strip().lower()

    # Filter rows by crop name only
    filtered = df[df['label'].str.lower() == crop]

    if filtered.empty:
        raise HTTPException(status_code=404, detail=f"No data found for crop '{crop}'.")

    # Calculate feature averages
    crop_details = {
        "crop": crop.title(),
        "N_mean": round(filtered['N'].mean(), 2),
        "P_mean": round(filtered['P'].mean(), 2),
        "K_mean": round(filtered['K'].mean(), 2),
        "temperature_mean": round(filtered['temperature'].mean(), 2),
        "humidity_mean": round(filtered['humidity'].mean(), 2),
        "rainfall_mean": round(filtered['rainfall'].mean(), 2)
    }

    return {"crop_details": crop_details}



# Api to get soil analysis, like types of soil and the percentage values they are found in
@app.get("/soil_analysis")
def soil_analysis(state: str):
    """
    :param state: Name of the state (case-insensitive)
    :return: Soil types and NPK values for the given state
    """
    result = get_soil_analysis(state)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result

# API to get Weekly average temprature and Rainfall data
@app.post("/monthly-avg")
def weather_summary(request: CityRequest):
    try:
        result = get_last_30_day_weather(request.city)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except EnvironmentError as ee:
        raise HTTPException(status_code=500, detail=str(ee))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
    

# API for fertilizer recommendations based on crop type
@app.post("/get-fertilizer")
def get_fertilizer(data: CropDetailRequest):
    crop = data.crop.strip().lower()
    matched = fertilizer[fertilizer['Crop Type'] == crop]

    if matched.empty:
        raise HTTPException(status_code=404, detail=f"No fertilizer data found for crop '{data.crop}'")

    results = []
    for _, row in matched.iterrows():
        fert_data = {
            "fertilizer_name": row['Fertilizer Name'],
            "nitrogen": row['Nitrogen'] if row['Nitrogen'] != 0 else None,
            "phosphorus": row['Phosphorus'] if row['Phosphorus'] != 0 else None,
            "potassium": row['Potassium'] if row['Potassium'] != 0 else None,
            "soil_moisture": row['Soil Moisture']
        }
        results.append(fert_data)

    return {"crop": data.crop.title(), "recommendations": results}

# Api for price prediction
@app.post("/get_price_data")
def get_price_data(request: PriceRequest):
    params = {
        "api-key": API_KEY,
        "format": "json",
        "filters[state.keyword]": request.state,
        "filters[commodity.keyword]": request.commodity,
        "limit": request.limit
    }
    
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch data from data.gov.in API")
    
    data = response.json()
    
    # Check if records exist
    if "records" not in data or not data["records"]:
        raise HTTPException(status_code=404, detail=f"No price data found for {request.commodity} in {request.state}")
    
    # Return the list of records directly
    return {"price_data": data["records"]}

@app.post("/prod_prediction")
async def predict_crop_output(input_data: Prod_Input):
    try:
        
        input_dict = {
            'State_Name': input_data.state_name.strip().lower(),
            'District_Name': input_data.district_name.strip().lower(),
            'Crop': input_data.crop.strip().lower(),
            'Season': input_data.season.strip().lower()
        }

        # Prepare input DataFrame
        input_df = pd.DataFrame([input_dict])

        # One-hot encode
        encoded_input = encoder.transform(input_df)

        # Predict
        pred_area = area_model.predict(encoded_input)[0]
        pred_prod = prod_model.predict(encoded_input)[0]

        return {
            "estimated_area": round(float(pred_area), 2),
            "estimated_production": round(float(pred_prod), 2),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")