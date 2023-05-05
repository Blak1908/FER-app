import argparse 
import cv2
import numpy as np 
import os

from app.core.settings import get_settings
from app.core.modules import emotic

settings = get_settings()

modes = settings.MODES
result_path = settings.RESULT_PATH
temp_path = settings.TEMP_PATH

if not os.path.exists(result_path):
  os.system(f"mkdir {result_path}")
if not os.path.exists(temp_path):
  os.system(f"mkdir {temp_path}")


def detect(context_norm, body_norm, ind2cat, ind2vad, args):

  camera_mode, video_mode, image_mode = False, False, False 
  if args.source == "0":
    camera_mode = True
    # Capture frame from camera
    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture(int(args.source))
    while True:
      ret, frame = cap.read()
      
      if not ret:
        print("End process! ")
        break
      
      image_context = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      
    
      pred = emotic.emotic_model.frame_predict(context_norm ,body_norm, ind2cat, ind2vad, image_context)

      if camera_mode:                                         
        # Show real-time capture                                         )
        cv2.imshow('Emotion Detector',pred)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    
  elif str(args.source).endswith(".mp4"):
    
    base_name = str(args.source).split('/')[-1]
    base_name = base_name.split('.')[0]
    
    video_mode = True
    # capture video
    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    idx = 0 
    
    while True:
      ret, frame = cap.read()
      
      if not ret:
        print("End process! ")
        break
      
      image_context = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      
      
      pred = emotic.emotic_model.frame_predict(context_norm ,body_norm, ind2cat, ind2vad, image_context)
      
      cv2.imwrite(f'{temp_path}/{idx+1:06d}.jpg', pred)
      idx += 1
        
    # Write video prediction
    if video_mode:   
      cmd_export_video = f'''ffmpeg -framerate {fps} -i {temp_path}/%06d.jpg -c:v libx264 -profile:v high -crf {fps} -pix_fmt yuv420p {result_path}/{base_name}_result.mp4''' 
      os.system(cmd_export_video)
      cmd_delete_tmp = f'rm {temp_path}/*.jpg'
      os.system(cmd_delete_tmp)
      
  else:
    base_name = str(args.source).split('/')[-1]
    base_name = base_name.split('.')[0]
    
    # Read image from source
    img = cv2.imread(args.source)
    
    # Convert image original to RGB 
    image_context = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Make forward prediction
    pred = emotic.emotic_model.frame_predict(context_norm ,body_norm, ind2cat, ind2vad, image_context)
    
    # Write image to result path
    cv2.imwrite(f"{result_path}/{base_name}.jpg", pred)
    
  
  

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--experiment_path', type=str, help='Path of experiment files (results, models, logs)')
    parser.add_argument('--model_dir', type=str, default='models', help='Folder to access the models')
    parser.add_argument('--result_dir', type=str, default='results', help='Path to save the results')
    parser.add_argument('--inference_file', type=str, help='Text file containing image context paths and bounding box')
    parser.add_argument('--source', help='0 camera, video file path, image file path')
    parser.add_argument('--imgsz',default=(640, 640), help='0 camera, video file path, image file path')
    parser.add_argument('--stride',default=32, help='The number of frame per thread')
    parser.add_argument('--vid_stride',default=1, help='video stride')
    parser.add_argument('--pt',default=True, help='video stride')
    # Generate args
    args = parser.parse_args()
    return args


if __name__=='__main__':
  args = parse_args()

  cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection', \
          'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear','Happiness', \
          'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
  cat2ind = {}
  ind2cat = {}
  for idx, emotion in enumerate(cat):
      cat2ind[emotion] = idx
      ind2cat[idx] = emotion
  
  vad = ['Valence', 'Arousal', 'Dominance']
  ind2vad = {}
  for idx, continuous in enumerate(vad):
      ind2vad[idx] = continuous
  
  context_mean = [0.4690646, 0.4407227, 0.40508908]
  context_std = [0.2514227, 0.24312855, 0.24266963]
  body_mean = [0.43832874, 0.3964344, 0.3706214]
  body_std = [0.24784276, 0.23621225, 0.2323653]
  context_norm = [context_mean, context_std]
  body_norm = [body_mean, body_std]
  if args.source is not None:
    print (f'inference over source: {args.source}')
    detect(context_norm, body_norm, ind2cat, ind2vad, args)
  else:
    print('source input invalid !!')



# python app/core/detect.py --source 0 --experiment_path /home/dattran/Documents/dattran/FER-app/weights/model_saved