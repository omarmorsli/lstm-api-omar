import requests
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
import urllib.parse
import re


API_URL = "https://lstm-api-ce674ddbb5fc.herokuapp.com/predict"


eurzmw_data = pd.read_csv("EURLBPX2.csv")

eurzmw_data.dropna(inplace=True)
eurzmw_data["Date"] = pd.to_datetime(eurzmw_data["Date"])
eurzmw_data.set_index("Date", inplace=True)
eurzmw_data["Log Returns"] = np.log(
    eurzmw_data["Close"] / eurzmw_data["Close"].shift(1)
)
eurzmw_data["3D MA"] = eurzmw_data["Close"].rolling(window=3).mean()
eurzmw_data["7D MA"] = eurzmw_data["Close"].rolling(window=7).mean()
eurzmw_data["15D MA"] = eurzmw_data["Close"].rolling(window=15).mean()
eurzmw_data["30D MA"] = eurzmw_data["Close"].rolling(window=30).mean()
# Calcul des bandes de Bollinger pour la moyenne mobile sur 7 jours
eurzmw_data["Bollinger High 7D"] = eurzmw_data["7D MA"] + (
    eurzmw_data["Close"].rolling(window=7).std() * 2
)
eurzmw_data["Bollinger Low 7D"] = eurzmw_data["7D MA"] - (
    eurzmw_data["Close"].rolling(window=7).std() * 2
)

# Calcul des bandes de Bollinger pour la moyenne mobile sur 15 jours
eurzmw_data["Bollinger High 30D"] = eurzmw_data["30D MA"] + (
    eurzmw_data["Close"].rolling(window=15).std() * 2
)
eurzmw_data["Bollinger Low 30D"] = eurzmw_data["30D MA"] - (
    eurzmw_data["Close"].rolling(window=15).std() * 2
)
# Calcul du RSI (Relative Strength Index)
eurzmw_data["RSI"] = ta.momentum.rsi(eurzmw_data["Close"], window=14)
# Calcul des MACD (Moving Average Convergence Divergence)
eurzmw_data["MACD"] = ta.trend.macd(eurzmw_data["Close"])
eurzmw_data["MACD Signal"] = ta.trend.macd_signal(eurzmw_data["Close"])
eurzmw_data["MACD Hist"] = ta.trend.macd_diff(eurzmw_data["Close"])
# Ajouter RSI et MACD
eurzmw_data["RSI"] = ta.momentum.rsi(eurzmw_data["Close"], window=14)
eurzmw_data["MACD"] = ta.trend.macd(eurzmw_data["Close"])
eurzmw_data["MACD Signal"] = ta.trend.macd_signal(eurzmw_data["Close"])
eurzmw_data["MACD Hist"] = ta.trend.macd_diff(eurzmw_data["Close"])
eurzmw_data["7D MA"] = eurzmw_data["Close"].rolling(window=7).mean()
eurzmw_data["30D MA"] = eurzmw_data["Close"].rolling(window=30).mean()
eurzmw_data["Bollinger High"] = ta.volatility.bollinger_hband(
    eurzmw_data["Close"], window=20
)
eurzmw_data["Bollinger Low"] = ta.volatility.bollinger_lband(
    eurzmw_data["Close"], window=20
)
eurzmw_data["ATR"] = ta.volatility.average_true_range(
    eurzmw_data["High"], eurzmw_data["Low"], eurzmw_data["Close"], window=14
)
eurzmw_data["Log Returns"] = np.log(
    eurzmw_data["Close"] / eurzmw_data["Close"].shift(1)
)
# Décomposition de la série temporelle (période de 30 jours)
decomposition = seasonal_decompose(
    eurzmw_data["Close"], model="multiplicative", period=30
)
eurzmw_data["Trend"] = decomposition.trend
eurzmw_data["Seasonal"] = decomposition.seasonal
eurzmw_data["Resid"] = decomposition.resid
# Extraction des caractéristiques temporelles depuis l'index
eurzmw_data["Day"] = eurzmw_data.index.day
eurzmw_data["Month"] = eurzmw_data.index.month
eurzmw_data["Year"] = eurzmw_data.index.year
# Ajouter des décalages temporels comme caractéristiques
eurzmw_data["Lag1"] = eurzmw_data["Close"].shift(1)
eurzmw_data["Lag2"] = eurzmw_data["Close"].shift(2)
# Supprimer les valeurs manquantes après avoir ajouté toutes les caractéristiques
eurzmw_data.dropna(inplace=True)

# Normalisation des données
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(
    eurzmw_data[
        [
            "Close",
            "RSI",
            "MACD",
            "MACD Signal",
            "MACD Hist",
            "Lag1",
            "Lag2",
            "7D MA",
            "30D MA",
            "Bollinger High",
            "Bollinger Low",
            "ATR",
            "Log Returns",
        ]
    ]
)

# Division en train et test sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]


# Création des séquences de données pour LSTM
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i : (i + time_step)])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)


time_step = 60  # Ajustement du time_step
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape des données pour LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])


payload_train = {"data": X_train.tolist()}
payload_test = {"data": X_test.tolist()}

headers = {"Content-Type": "application/json"}

response_train = requests.post(API_URL, data=payload_train, headers=headers)
response_test = requests.post(API_URL, data=payload_test, headers=headers)

train_predict = response_train.json()['detail'][0]['ctx']['doc']
test_predict = response_test.json()['detail'][0]['ctx']['doc']


# decoded_data = urllib.parse.unquote(train_predict)
# numbers = re.findall(r"[\d\.]+", decoded_data)
# floats = [float(x) for x in numbers]
# print(decoded_data)


# Replace plt.plot with plotly
# fig = go.Figure()

# # Add real values
# fig.add_trace(go.Scatter(
#     x=eurzmw_data.index,
#     y=eurzmw_data['Close'],
#     mode='lines',
#     name='Valeurs Réelles'
# ))

# # Add train predictions
# fig.add_trace(go.Scatter(
#     x=eurzmw_data.index[time_step:len(train_predict) + time_step],
#     y=train_predict,
#     mode='lines',
#     name='Prédictions Train'
# ))

# # Add test predictions
# fig.add_trace(go.Scatter(
#     x=eurzmw_data.index[len(train_predict) + (2 * time_step) + 1:len(train_predict) + (2 * time_step) + 1 + len(test_predict)],
#     y=test_predict,
#     mode='lines',
#     name='Prédictions Test'
# ))

# # Set title and labels
# fig.update_layout(
#     title='Prédictions du Modèle LSTM',
#     xaxis_title='Date',
#     yaxis_title='Prix de Clôture',
#     legend_title='Légende'
# )

# fig.show()