import cv2
from app.core.modules.emotic.utils.yolo_utils import prepare_yolo, get_bbox
import torch
import numpy as np
import os
from app.core.modules.emotic.utils.inference import infer
from app.core.emotic import Emotic
from app.core.settings import get_settings
from app.core.utils import download_model


settings = get_settings()

MAT_PATH = settings.MAT_PATH
MODEL_PATH = settings.MODEL_PATH
DEVICE = settings.DEVICE

ID_MAT = settings.ID_MAT
ID_MODEL = settings.ID_MODEL


# Download weights on google drive
if not os.path.exists(MAT_PATH):
    os.system(f"mkdir {MAT_PATH}")
    print("Downloading mat file on Drive...")
    status = download_model(ID_MAT,MAT_PATH)
    if not status:
        print("Cannot download mat file on Drive")

if not os.path.exists(MODEL_PATH):
    os.system(f"mkdir {MODEL_PATH}")
    print("Downloading model on Drive...")
    status = download_model(ID_MODEL,MODEL_PATH)
    if not status:
        print("Cannot download model file on Drive")

class EMOTIC():
    def __init__(self, mat_path, model_path, device):
        print("Start loading models...")

        self.device = device
        self.yolo = prepare_yolo(model_path)
        self.yolo = self.yolo.to(device)
        self.yolo.eval()

        self.thresholds = torch.FloatTensor(np.load(os.path.join(mat_path, 'val_thresholds.npy'))).to(device) 
        self.model_context = torch.load(os.path.join(model_path,'model_context1.pth')).to(device)
        self.model_body = torch.load(os.path.join(model_path,'model_body1.pth')).to(device)
        self.emotic_model = torch.load(os.path.join(model_path,'model_emotic1.pth')).to(device)
        self.models = [self.model_context, self.model_body, self.emotic_model]

    def frame_predict(self ,context_norm ,body_norm, ind2cat, ind2vad, image_context):
       
        try: 
            bbox_yolo = get_bbox(self.yolo,self.device, image_context)
            for pred_idx, pred_bbox in enumerate(bbox_yolo):
                pred_cat, pred_cont = infer(context_norm, body_norm, ind2cat, ind2vad, self.device, self.thresholds, self.models, image_context=image_context, bbox=pred_bbox, to_print=False)
                write_text_vad = list()
                for continuous in pred_cont:
                    write_text_vad.append(str('%.1f' %(continuous)))

                write_text_vad = 'vad ' + ' '.join(write_text_vad)

                # Draw a retangle into frame
                image_context = cv2.rectangle(image_context, (pred_bbox[0], pred_bbox[1]),(pred_bbox[2] , pred_bbox[3]), (255, 0, 0), 3)

                # Write predict categories into frame
                cv2.putText(image_context, write_text_vad, (pred_bbox[0], pred_bbox[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                for i, emotion in enumerate(pred_cat):
                    cv2.putText(image_context, emotion, (pred_bbox[0], pred_bbox[1] + (i+1)*12), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                    
        except Exception as e:
            print("Error: ", e)
            pass
        return image_context

emotic_model = EMOTIC(mat_path=MAT_PATH,model_path=MODEL_PATH, device=DEVICE)

if __name__=='__main__':
   print("Init Emotic model ... ")