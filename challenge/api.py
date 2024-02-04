from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List
import pandas as pd
from model import DelayModel
import os

# check if model exists otherwise create it
if not os.path.exists("xgb_model_2.model"):
    data = pd.read_csv(filepath_or_buffer="./data/data.csv")
    model = DelayModel()
    features, target = model.preprocess(data=data, target_column="delay")
    model.fit(features, target)
    model._model.save_model("xgb_model_2.model")
else:
    model = DelayModel()
    model._model.load_model("xgb_model_2.model")

class FlightData(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

class InputData(BaseModel):
    flights: List[FlightData]

    @validator('flights')
    def validate_flights(cls, v):
        for flight in v:
            if flight.TIPOVUELO not in ['N', 'I']:
                raise ValueError('TIPOVUELO must be either "N" or "I"')
            if not (1 <= flight.MES <= 12):
                raise ValueError('MES must be an integer between 1 and 12')
        return v

app = FastAPI()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(input_data: InputData) -> dict:
    preds = []
    for flight in input_data.flights:
        data = pd.DataFrame([flight.dict()],index=[0])
        data = model.preprocess(data=data, train=False)
        prediction = model.predict(data.iloc[0].values.reshape(1, -1))
        preds.append(prediction[0])
    return preds