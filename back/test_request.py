import requests

url = "http://127.0.0.1:8000/predict"

sample_input = {
    "input": [
        [[0.1]*13]*60
    ]
}

response = requests.post(url, json=sample_input)
print(response.json())