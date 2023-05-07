import cv2
import os
import pandas as pd
import glob

from app.core.modules.emotic.utils.yolo_utils import prepare_yolo, get_bbox
from app.core.settings import get_settings

settings = get_settings()

model_path = settings.MODEL_PATH
DEVICE = settings.DEVICE

device = DEVICE
yolo = prepare_yolo(model_path)
yolo = yolo.to(device)
yolo.eval()

data_root_path = '/home/cuongacpe/workspace/FER-app/data'
video_root = '/home/cuongacpe/workspace/FER-app/video_data'
temp_path = '/home/cuongacpe/workspace/FER-app/data/temp'
context_path = '/home/cuongacpe/workspace/FER-app/data/context'
body_path = '/home/cuongacpe/workspace/FER-app/data/body'
data_csv_path = 'asian.csv'

if not os.path.exists(data_root_path):
    os.system(f"mkdir {data_root_path}")

if not os.path.exists(video_root):
    os.system(f"mkdir {video_root}")

if not os.path.exists(temp_path):
    os.system(f"mkdir {temp_path}")

if not os.path.exists(context_path):
    os.system(f"mkdir {context_path}")

if not os.path.exists(body_path):
    os.system(f"mkdir {body_path}")

if not os.path.isfile(data_csv_path):
    print("File data does not exist.")
    print("Create csv file: ", data_csv_path)
    # initialize data of lists.
    data = {'Folder': [],
            'Filename': [],
            'Image size': [],
            'BBox': [],
            'Categorical_Labels': [],
            'Continuous_Labels': [],
            'Gender': [],
            'Age': []}
    
    # Create a DataFrame with index column and header
    df = pd.DataFrame(data)
    df.index.name = "Index"

    # Print the DataFrame
    df.to_csv('asian.csv')




df = pd.read_csv(data_csv_path)
# Video capture:
videos_path = sorted(glob.glob(os.path.join(video_root,'*.mp4')))
for vid_path in videos_path:
    
    base_name = vid_path.split("/")[-1]

    base_name = base_name.split('.')[0]
    
    
    cap = cv2.VideoCapture(vid_path)

    count = 0

    while True:
        
        ret , frame = cap.read()
        
        if not ret:
            break
        
        cv2.imwrite(f'{temp_path}/{base_name}_{count}.jpg', frame)
        count += 1        
        
    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

    frames_path = sorted(glob.glob(os.path.join(temp_path,'*.jpg')))
    print("Len frames: ", len(frames_path))
    file_names = df['Filename'].tolist()
    flag = False
    for i in range(0, len(frames_path)):
        if i % 10 == 0 or flag:
            print("i: ", i)
            frame = cv2.imread(frames_path[i])
            base_name = frames_path[i].split("/")[-1]
            base_name = base_name.split('.')[0]
            if f'{base_name}.jpg' in file_names:
                print("Image name is duplicated.")
                continue
            try:
                box = get_bbox(yolo,device=device,image_context=frame)
                cropped_body = frame[box[0][1]:box[0][3], box[0][0]:box[0][2]]
                cv2.imwrite(f'{body_path}/{base_name}.jpg',cropped_body)
                cv2.imwrite(f'{context_path}/{base_name}.jpg',frame)

                tmp = context_path.split("/")

                folder = str(os.path.join(tmp[5], tmp[6]))
                file_name = f'{base_name}.jpg'
                img_size = f'[{frame.shape[1::-1]}]'
                bbox = f'[{box[0][0]}, {box[0][1]}, {box[0][2]}, {box[0][3]}]'
                Categorical_Labels = ''
                Continuous_Labels = ''
                Gender = ''
                Age = ''

                index = len(df.index)
                df.loc[len(df.index)] = [index, folder, file_name, img_size, bbox, Categorical_Labels, Continuous_Labels, Gender, Age] 

                flag = False      
            except Exception as e:
                print("Error: ",e)
                flag = True
                continue

            cmd_rm = f"rm {temp_path}/*.jpg"
            os.system(cmd_rm)

df.to_csv('asian.csv', index=False)

