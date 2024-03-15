import requests

url = 'http://localhost:5000/api/chat'  # Updated endpoint
data = {'question': 'I am having cough'}
response = requests.post(url, json=data)

print(response.json())
