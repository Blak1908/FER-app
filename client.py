import requests
import json

url = "http://127.0.0.1:8001/api/v1/object-analysis/"

payload = json.dumps({
  "driverUrl": "string"
})
headers = {
  'accept': 'application/json',
  'Content-Type': 'application/json'
}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)
