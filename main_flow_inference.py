import sounddevice as sd
import json
import requests
import base64

fs = 48000  # Sample rate
seconds = 5  # Duration of recording
seconds_statement = 10

url = "http://127.0.0.1:8000/api/v1/speech2text"


print("Start listening in 5s...")
audio_recorded = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
# Convert NumPy array to bytes
audio_recorded_bytes = audio_recorded.tobytes()
# Encode bytes as base64 string
audio_recorded_base64 = base64.b64encode(audio_recorded_bytes).decode('utf-8')
payload = json.dumps({
    "audio": audio_recorded_base64
    })
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
    }
response = requests.request("POST", url, headers=headers, data=payload)

if response.status_code == 200:
    # Parse the JSON response
    json_response = response.json()

    # Access the 'result' value from the JSON response
    result_text = json_response.get('text')

    # Use the 'result_text' as needed
    print("Speech-to-text text:")
    print(result_text)
else:
    # Handle the case when the request was not successful
    print(f"API request failed with status code: {response.status_code}")
