import requests
import json
import glob
import base64

image_paths = glob.glob("/home/dattran/datadrive/AI-Chatbot-Synthesis/Source_tiktok/*.png")  # Replace with your actual image paths

binary_images = []

for path in image_paths:
    with open(path, "rb") as file:
        binary_data = file.read()
        encoded_image = base64.b64encode(binary_data).decode("utf-8")
        binary_images.append(encoded_image)
# import pdb; pdb.set_trace()

url = "http://127.0.0.1:8000/api/v1/object-analysis/"

payload = json.dumps({
  "images": binary_images
})
headers = {
  'accept': 'application/json',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
