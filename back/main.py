from fastapi import FastAPI, Request
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

app = FastAPI()

model_path = 'eurlbp_lstm_model.h5'
model = load_model(model_path)

class PredictRequest(BaseModel):
    input: list

def preprocess_input(input_data, time_steps=60):
    input_array = np.array(input_data)
    if input_array.shape[1:] != (time_steps, 14):
        raise ValueError("Input array must have shape (samples, 60, 14)")
    return input_array

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        input_data = preprocess_input(request.input)
    except ValueError as e:
        return {"error": str(e)}
    
    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)