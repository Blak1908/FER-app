from deepface import DeepFace
import os
from collections import Counter
import json

models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
]

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe',
  'yolov8',
]

distances = [
    'cosine',
    'euclidean',
    'euclidean_l2'
]


def calculate_mean(lst):
    if not lst:
        return None
    total_sum = sum(lst)
    mean = total_sum / len(lst)
    return mean

def most_common_element(lst):
    # Create a Counter object to count the occurrences of each element
    counter = Counter(lst)

    # Use the most_common() method to get a list of tuples with elements and their counts
    most_common_items = counter.most_common()

    # The most common item will be the first element in the most_common_items list
    most_common_element, count = most_common_items[0]

    return most_common_element


num_frames = 5  # Number of frames to capture
output_folder = "frames"  # Folder to save the frames

frames = []


while len(frames) < num_frames:


    # Generate a unique filename for the frame
    filename = f"frame{len(frames)+1}.jpg"
    filepath = os.path.join(output_folder, filename)

    # Append the file path to the list of frames
    frames.append(filepath)

# identities person
db_path = "/home/cuongacpe/Documents/AI-Chatbot-Synthesis/app/modules/user_recognition/user/database"
file_name = f"representations_{models[1]}.pkl"
file_name = file_name.replace("-", "_").lower()

names_list = [] # User's name

max = 0
# Face verification
for frame_path in frames:
    sub_names_list = []
    result_find = DeepFace.find(img_path = frame_path,
        db_path = db_path, 
        model_name = models[1],
        detector_backend = backends[2],
        distance_metric = distances[2]
    )
    for df_data in result_find:
        name = df_data['label'][0]
        name = name.split("-")
        name = name[0]
        sub_names_list.append(name)
    if len(result_find) > max:
        max = len(result_find)
        names_list = sub_names_list

# Face analysis

emotions_lists = [[] for _ in range(max)]
ages_lists = [[] for _ in range(max)]
genders_lists = [[] for _ in range(max)]
races_lists = [[] for _ in range(max)]

for frame_path in frames:
    result = DeepFace.analyze(frame_path, actions=['emotion', 'age', 'race', 'gender'], enforce_detection=False, detector_backend = backends[2])
    if len(result) == max:
        for i in range(max):
            emotions_lists[i].append(result[i]['dominant_emotion'])
            ages_lists[i].append(result[i]['age'])
            genders_lists[i].append(result[i]['dominant_gender'])
            races_lists[i].append(result[i]['dominant_race'])

result_emotions = []
result_ages = []
result_genders = []
result_races = []

for i in range(max):
    result_emotions.append(most_common_element(emotions_lists[i]))
    result_ages.append(calculate_mean(ages_lists[i]))
    result_genders.append(most_common_element(genders_lists[i]))
    result_races.append(most_common_element(races_lists[i]))

import pdb; pdb.set_trace()
flag = True

if flag:
    
    default_details = {
        "name": names_list,
        "emotion": result_emotions
    }

    # Save the emotion details to a JSON file
    with open("default_details.json", 'w') as file:
        json.dump(default_details, file, indent=4)

    print("Emotion details saved to default_details.json")
else:
    request = input("Enter the parameter(s) you want to choose (age, gender, race), separated by spaces: ").split()
    request_details = {}

    # Include the chosen parameters and their details in the request details
    for param in request:
        if param == 'age':
            request_details['age'] = result_ages
        elif param == 'gender':
            request_details['gender'] = result_genders
        elif param == 'race':
            request_details['race'] = result_races

    # Include emotion details in the request details
    request_details['name'] = names_list
    request_details['emotion'] = result_emotions

    if len(request_details) > 0:
        with open("request_details.json", 'w') as file:
            json.dump(request_details, file, indent=4)
        print("Request details saved to request_details.json")
    else:
        print("No requested parameters found in the result.")
