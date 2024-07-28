import requests
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Tuple, List, Dict, Any
import plotly.express as px
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import plotly
import json

API_URL = "https://lstm-api-ce674ddbb5fc.herokuapp.com/predict"
CSV_FILE_PATH = "EURLBPX2.csv"
TIME_STEP = 60  # Adjust as needed

app = FastAPI()

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index("Date", inplace=True)
    return data

def add_features(data: pd.DataFrame) -> pd.DataFrame:
    data["Log Returns"] = np.log(data["Close"] / data["Close"].shift(1))
    data["3D MA"] = data["Close"].rolling(window=3).mean()
    data["7D MA"] = data["Close"].rolling(window=7).mean()
    data["15D MA"] = data["Close"].rolling(window=15).mean()
    data["30D MA"] = data["Close"].rolling(window=30).mean()
    data["Bollinger High 7D"] = data["7D MA"] + (data["Close"].rolling(window=7).std() * 2)
    data["Bollinger Low 7D"] = data["7D MA"] - (data["Close"].rolling(window=7).std() * 2)
    data["Bollinger High 30D"] = data["30D MA"] + (data["Close"].rolling(window=30).std() * 2)
    data["Bollinger Low 30D"] = data["30D MA"] - (data["Close"].rolling(window=30).std() * 2)
    data["RSI"] = ta.momentum.rsi(data["Close"], window=14)
    data["MACD"] = ta.trend.macd(data["Close"])
    data["MACD Signal"] = ta.trend.macd_signal(data["Close"])
    data["MACD Hist"] = ta.trend.macd_diff(data["Close"])
    data["ATR"] = ta.volatility.average_true_range(data["High"], data["Low"], data["Close"], window=14)
    
    decomposition = seasonal_decompose(data["Close"], model="multiplicative", period=30)
    data["Trend"] = decomposition.trend
    data["Seasonal"] = decomposition.seasonal
    data["Resid"] = decomposition.resid

    data["Day"] = data.index.day
    data["Month"] = data.index.month
    data["Year"] = data.index.year
    data["Lag1"] = data["Close"].shift(1)
    data["Lag2"] = data["Close"].shift(2)
    
    data.dropna(inplace=True)
    return data

def normalize_data(data: pd.DataFrame, features: List[str]) -> np.ndarray:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])
    return scaled_data

def split_data(data: np.ndarray, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

def create_dataset(dataset: np.ndarray, time_step: int) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i : (i + time_step)])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

def add_additional_feature(X: np.ndarray) -> np.ndarray:
    additional_feature = np.zeros((X.shape[0], X.shape[1], 1))
    X = np.concatenate((X, additional_feature), axis=2)
    return X

def get_predictions(api_url: str, payload: Dict[str, Any]) -> List[float]:
    response = requests.post(api_url, json=payload)
    predictions = response.json()['prediction']
    return [item[0] for item in predictions]

def plot_predictions(train_actual: np.ndarray, train_predict: List[float], test_actual: np.ndarray, test_predict: List[float], time_step: int) -> Tuple[str, str]:
    train_actual = train_actual[time_step:time_step + len(train_predict)]
    test_actual = test_actual[:len(test_predict)]
    
    train_df = pd.DataFrame({
        'Time': range(time_step, time_step + len(train_predict)),
        'Actual': train_actual,
        'Predicted': train_predict
    })

    test_df = pd.DataFrame({
        'Time': range(len(train_actual) + time_step, len(train_actual) + time_step + len(test_predict)),
        'Actual': test_actual,
        'Predicted': test_predict
    })

    fig_train = px.line(train_df, x='Time', y=['Actual', 'Predicted'], title='Actual vs. Predicted Prices - Training Data')
    fig_train.update_layout(yaxis_title='Normalized Price', xaxis_title='Time')

    fig_test = px.line(test_df, x='Time', y=['Actual', 'Predicted'], title='Actual vs. Predicted Prices - Testing Data')
    fig_test.update_layout(yaxis_title='Normalized Price', xaxis_title='Time')

    train_html = fig_train.to_html(full_html=False)
    test_html = fig_test.to_html(full_html=False)

    return train_html, test_html

@app.get("/dashboard", response_class=HTMLResponse)
def predict():
    data = load_and_preprocess_data(CSV_FILE_PATH)
    data = add_features(data)

    features = ["Close", "RSI", "MACD", "MACD Signal", "MACD Hist", "Lag1", "Lag2", "7D MA", "30D MA", "Bollinger High 7D", "Bollinger Low 7D", "ATR", "Log Returns"]
    scaled_data = normalize_data(data, features)
    
    train_data, test_data = split_data(scaled_data)
    train_size = int(len(scaled_data) * 0.8)
    
    X_train, y_train = create_dataset(train_data, TIME_STEP)
    X_test, y_test = create_dataset(test_data, TIME_STEP)

    X_train = add_additional_feature(X_train)
    X_test = add_additional_feature(X_test)
    
    train_payload = {"input": X_train.tolist()}
    test_payload = {"input": X_test.tolist()}
    
    train_predict = get_predictions(API_URL, train_payload)
    test_predict = get_predictions(API_URL, test_payload)

    train_actual = scaled_data[:train_size, 0]
    test_actual = scaled_data[train_size:, 0]

    train_html, test_html = plot_predictions(train_actual, train_predict, test_actual, test_predict, TIME_STEP)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Predictions</title>
    </head>
    <body>
        <h1>Training Data</h1>
        {train_html}
        <h1>Testing Data</h1>
        {test_html}
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
