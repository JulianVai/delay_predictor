from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import xgboost as xgb

# Sample model loading, you should load your trained model here
model = xgb.Booster()
model.load_model("your_model_path.model")

app = FastAPI()

class InputData(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: str

@app.post("/predict/")
async def predict_delay(input_data: InputData):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Perform one-hot encoding
        input_df = pd.get_dummies(input_df)

        # Ensure input features match model's expected features
        # You might need to adjust this based on your model's requirements
        missing_cols = set(['OPERA_0', 'TIPOVUELO_0', 'MES_0']) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0

        # Reorder columns to match model's input order
        input_df = input_df[['OPERA_0', 'OPERA_1', 'TIPOVUELO_0', 'TIPOVUELO_1', 'MES_0', 'MES_1', 'MES_2']]

        # Predict using the loaded XGBoost model
        prediction = model.predict(xgb.DMatrix(input_df))[0]

        # Threshold the prediction
        delay_marker = 1 if prediction >= 0.5 else 0

        return {"delay_marker": delay_marker}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))