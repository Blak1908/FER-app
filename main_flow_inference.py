import sounddevice as sd
import json
import requests
import base64
import asyncio
import cv2
from app.core.settings import get_settings
from app.core.utils import remove_special_characters

settings = get_settings()

sample_rate = 48000  # Sample rate
seconds = 5  # Duration of recording
seconds_statement = 10



url_s2t = "http://127.0.0.1:8000/api/v1/speech2text"
url = "http://http://0.0.0.0:8001/api/v1/speech2text"
# url_s2t = get_settings

headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
        }

async def consume_messages():
    while True:
        print("Start listening in 5s...")
        audio_recorded = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()  # Wait until recording is finished
        # Convert NumPy array to bytes
        audio_recorded_bytes = audio_recorded.tobytes()
        # Encode bytes as base64 string
        audio_recorded_base64 = base64.b64encode(audio_recorded_bytes).decode('utf-8')
        payload = json.dumps({
            "audio": audio_recorded_base64
            })
        
        response = requests.request("POST", url_s2t, headers=headers, data=payload)

        if response.status_code == 200:
            # Parse the JSON response
            json_response = response.json()

            # Access the 'result' value from the JSON response
            result_text = json_response.get('text')
            
            # Use the 'result_text' as needed
            print("Speech-to-text text:")
            print(result_text)
            if remove_special_characters(result_text.strip().lower()) == "xin chào":
                print("start object analysis")
                binary_images = [] 
                cap = cv2.VideoCapture(1)
                if not cap.isOpened():
                    print("Camera not opened. Check if it is available and accessible.")
                    exit(1)
                idx = 0 
                while True:
                    # Read the frame from the video capture
                    ret, frame = cap.read()
                    if ret:  # Check if the frame is successfully captured
                        # Encode the frame to base64
                        _, buffer = cv2.imencode('.jpg', frame)
                        encoded_image = base64.b64encode(buffer).decode("utf-8")
                        binary_images.append(encoded_image)
                        
                    else:
                        print("Error capturing frame.")
                        break
                    
                    if idx == 5:
                        break
                    idx += 1
                # Release the video capture
                cap.release()
                    
                payload = json.dumps({
                    "images": binary_images,
                    "isRequire_analys": "true"
                    })
                
                response = requests.request("POST", url, headers=headers, data=payload)
                json_response = response.json()
                print(json_response)
                
            else:
                continue
            
            if remove_special_characters(result_text.strip().lower()) == "tạm biệt":
                print("Stop system")
                break
        else:
            # Handle the case when the request was not successful
            print(f"API request failed with status code: {response.status_code}")
            
asyncio.run(consume_messages())