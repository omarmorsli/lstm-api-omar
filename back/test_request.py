import requests

url = "https://lstm-api-ce674ddbb5fc.herokuapp.com/predict"

sample_input = {
    "input": [
        [[0.1]*14]*60
    ]
}

response = requests.post(url, json=sample_input)
print(response.json())