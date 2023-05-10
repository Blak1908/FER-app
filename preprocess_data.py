import cv2
import os 
import glob

video_path = "/home/dattran/datadrive/FER-app/sample_1.mp4"

data_root_path = '/home/dattran/datadrive/FER-app/data'

video_root = '/home/dattran/datadrive/FER-app/video_data'



if not os.path.exists(video_root):
    os.system(f"mkdir {video_root}")

if not os.path.exists(data_root_path):
    os.system(f"mkdir {data_root_path}")

    
    
    
    
videos_path = sorted(glob.glob(os.path.join(video_root,'*.mp4')))

frames =  []
for vid_path in videos_path:
    
    base_name = vid_path.split("/")[-1]
    base_name = base_name.split('.')[0]
    
    folder_id_path = os.path.join(data_root_path,base_name)
    
    if not os.path.exists(folder_id_path):
        os.system(f"mkdir {folder_id_path}")
    
    cap = cv2.VideoCapture(vid_path)

    count = 0


    while True:
        
        ret , frame = cap.read()
        
        if not ret:
            break
        
        # cv2.imwrite(f'{folder_id_path}/{base_name}_{count}.jpg', frame)
        # count += 1
        frames.append(frame)
        
        
    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    

folders_id = sorted(glob.glob(os.path.join(data_root_path,'*')))


from app.core.modules.emotic.utils.yolo_utils import prepare_yolo, get_bbox
from app.core.settings import get_settings

settings = get_settings()

model_path = settings.MODEL_PATH
DEVICE = settings.DEVICE

device = DEVICE
yolo = prepare_yolo(model_path)
yolo = yolo.to(device)
yolo.eval()



data_root_body = '/home/dattran/datadrive/FER-app/data_body'
if not os.path.exists(data_root_body):
    os.system(f"mkdir {data_root_body}")


for folder_id in folders_id:
    print(f'Folder id: ', folder_id)
    count = 0
    base_name = folder_id.split("/")[-1]
    base_name = base_name.split('.')[0]
    
    
    folder_id_body = f'{data_root_body}/{base_name}_body'
    
    if not os.path.exists(folder_id_body):
        os.system(f"mkdir {folder_id_body}")
    
    import pdb; pdb.set_trace()
    frames_path = sorted(glob.glob(os.path.join(folder_id,'*.jpg')))
    print("Len frames: ", len(frames_path))
    for i, frame_p in enumerate(frames_path):
        if i % 1000 == 0:
            print("i: ", i)
            frame = cv2.imread(frame_p)
            base_name_bod = frame_p.split("/")[-1]
            base_name_bod = base_name_bod.split('.')[0]
            try:
                box = get_bbox(yolo,device=device,image_context=frame)
                cropped_body = frame[box[0][1]:box[0][3], box[0][0]:box[0][2]]
                cv2.imwrite(f'{folder_id_body}/{base_name_bod}.jpg',cropped_body)
                
            except Exception as e:
                print("Error: ",e)
                pass
            
            
        
        

        
        
    