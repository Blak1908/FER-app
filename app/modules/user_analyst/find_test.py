from deepface import DeepFace
import os
import shutil
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

db_path = "/home/cuongacpe/Documents/AI-Chatbot-Synthesis/app/modules/user_recognition/user/database"
file_name = f"representations_{models[1]}.pkl"
file_name = file_name.replace("-", "_").lower()

num_frames = 5  # Number of frames to capture
output_folder = "frames"  # Folder to save the frames

frames = []


while len(frames) < num_frames:


    # Generate a unique filename for the frame
    filename = f"frame{len(frames)+1}.jpg"
    filepath = os.path.join(output_folder, filename)

    # Append the file path to the list of frames
    frames.append(filepath)
names_list = []
results_list = []
max = 0
#face verification
for frame_path in frames:
  sub_names_list = []
  result = DeepFace.find(img_path = frame_path,
        db_path = db_path, 
        model_name = models[1],
        detector_backend = backends[2],
        distance_metric = distances[2]
  )
  for df_data in result:
        name = df_data['label'][0]
        name = name.split("-")
        name = name[0]
        sub_names_list.append(name)
  if len(result) > max:
      max = len(result)
      names_list = sub_names_list
  results_list.append(result)
import pdb; pdb.set_trace()
print(names_list)
