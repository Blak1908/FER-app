import argparse 
import cv2
import numpy as np 
import os


import torch 
from torchvision import transforms

from emotic import Emotic
from app.core.utils.inference import infer
from app.core.utils.yolo_utils import prepare_yolo, rescale_boxes, non_max_suppression





def get_bbox(yolo_model, device, image_context, yolo_image_size=416, conf_thresh=0.8, nms_thresh=0.4):

  test_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
  image_yolo = test_transform(cv2.resize(image_context, (416, 416))).unsqueeze(0).to(device)

  with torch.no_grad():
    detections = yolo_model(image_yolo)
    nms_det  = non_max_suppression(detections, conf_thresh, nms_thresh)[0]
    det = rescale_boxes(nms_det, yolo_image_size, (image_context.shape[:2]))
  
  bboxes = []
  for x1, y1, x2, y2, _, _, cls_pred in det:
    if cls_pred == 0:  # checking if predicted_class = persons. 
      x1 = int(min(image_context.shape[1], max(0, x1)))
      x2 = int(min(image_context.shape[1], max(x1, x2)))
      y1 = int(min(image_context.shape[0], max(15, y1)))
      y2 = int(min(image_context.shape[0], max(y1, y2)))
      bboxes.append([x1, y1, x2, y2])
  return np.array(bboxes)


def yolo_infer(images_list, result_path, model_path, context_norm, body_norm, ind2cat, ind2vad, args):
  ''' Infer on a list of images defined in images_list text file to obtain bounding boxes of persons in the images using yolo model.
  :param images_list: Text file specifying the images to conduct inference. A row in the file is Path_of_image. 
  :param result_path: Directory path to save the results (images with the predicted emotion categories and continuous emotion dimesnions).
  :param model_path: Directory path to load models and val_thresholds to perform inference.
  :param context_norm: List containing mean and std values for context images. 
  :param body_norm: List containing mean and std values for body images. 
  :param ind2cat: Dictionary converting integer index to categorical emotion. 
  :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
  :param args: Runtime arguments.
  '''
  device = torch.device("cuda:%s" %(str(args.gpu)) if torch.cuda.is_available() else "cpu")
  yolo = prepare_yolo(model_path)
  yolo = yolo.to(device)
  yolo.eval()


  print("Start loading models...")
  thresholds = torch.FloatTensor(np.load(os.path.join(result_path, 'val_thresholds.npy'))).to(device) 
  model_context = torch.load(os.path.join(model_path,'model_context1.pth')).to(device)
  model_body = torch.load(os.path.join(model_path,'model_body1.pth')).to(device)
  emotic_model = torch.load(os.path.join(model_path,'model_emotic1.pth')).to(device)
  
  models = [model_context, model_body, emotic_model]

  with open(images_list, 'r') as f:
    lines = f.readlines()
  
  for idx, line in enumerate(lines):
    image_context_path = line.split('\n')[0].split(' ')[0]
    image_context = cv2.cvtColor(cv2.imread(image_context_path), cv2.COLOR_BGR2RGB)
    try:
      bbox_yolo = get_bbox(yolo, device, image_context)
      for pred_bbox in bbox_yolo:
        pred_cat, pred_cont = infer(context_norm, body_norm, ind2cat, ind2vad, device, thresholds, models, image_context=image_context, bbox=pred_bbox, to_print=False)
        write_text_vad = list()
        for continuous in pred_cont:
          write_text_vad.append(str('%.1f' %(continuous)))
        write_text_vad = 'vad ' + ' '.join(write_text_vad) 
        image_context = cv2.rectangle(image_context, (pred_bbox[0], pred_bbox[1]),(pred_bbox[2] , pred_bbox[3]), (255, 0, 0), 3)
        cv2.putText(image_context, write_text_vad, (pred_bbox[0], pred_bbox[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        for i, emotion in enumerate(pred_cat):
          cv2.putText(image_context, emotion, (pred_bbox[0], pred_bbox[1] + (i+1)*12), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    except Exception as e:
      print ('Exception for image ',image_context_path)
      print (e)
    cv2.imwrite(os.path.join(result_path, 'img_%r.jpg' %(idx)), cv2.cvtColor(image_context, cv2.COLOR_RGB2BGR))
    print ('completed inference for image %d'  %(idx))



def detect(result_path, model_path, context_norm, body_norm, ind2cat, ind2vad, args):

  device = torch.device("cuda:%s" %(str(args.gpu)) if torch.cuda.is_available() else "cpu")
  yolo = prepare_yolo(model_path)
  yolo = yolo.to(device)
  yolo.eval()

  thresholds = torch.FloatTensor(np.load(os.path.join(result_path, 'val_thresholds.npy'))).to(device) 
  model_context = torch.load(os.path.join(model_path,'model_context1.pth')).to(device)
  model_body = torch.load(os.path.join(model_path,'model_body1.pth')).to(device)
  emotic_model = torch.load(os.path.join(model_path,'model_emotic1.pth')).to(device)

  model_context.eval()
  model_body.eval()
  emotic_model.eval()
  models = [model_context, model_body, emotic_model]


  # Capture frame from camera
  cv2.ocl.setUseOpenCL(False)
  cap = cv2.VideoCapture(int(args.source))

  while True:
    _, frame = cap.read()
    image_context = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try: 
      bbox_yolo = get_bbox(yolo, device, image_context)
      for pred_idx, pred_bbox in enumerate(bbox_yolo):
        pred_cat, pred_cont = infer(context_norm, body_norm, ind2cat, ind2vad, device, thresholds, models, image_context=image_context, bbox=pred_bbox, to_print=False)
        write_text_vad = list()
        for continuous in pred_cont:
          write_text_vad.append(str('%.1f' %(continuous)))
        write_text_vad = 'vad ' + ' '.join(write_text_vad) 
        image_context = cv2.rectangle(image_context, (pred_bbox[0], pred_bbox[1]),(pred_bbox[2] , pred_bbox[3]), (255, 0, 0), 3)
        cv2.putText(image_context, write_text_vad, (pred_bbox[0], pred_bbox[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        for i, emotion in enumerate(pred_cat):
          cv2.putText(image_context, emotion, (pred_bbox[0], pred_bbox[1] + (i+1)*12), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.imshow('Emotion Detector',image_context)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break  
    except Exception:
      pass

  


def check_paths(args):
  ''' Check (create if they don't exist) experiment directories.
  :param args: Runtime arguments as passed by the user.
  :return: result_dir_path, model_dir_path.
  '''
  if args.inference_file is not None: 
    if not os.path.exists(args.inference_file):
      raise ValueError('inference file does not exist. Please pass a valid inference file')

  model_path = os.path.join(args.experiment_path, args.model_dir)
  if not os.path.exists(model_path):
    raise ValueError('model path %s does not exist. Please pass a valid model_path' %(model_path))
  result_path = os.path.join(args.experiment_path, args.result_dir)
  if not os.path.exists(result_path):
    os.makedirs(result_path)
  return result_path, model_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--experiment_path', type=str, required=True, help='Path of experiment files (results, models, logs)')
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

  result_path, model_path = check_paths(args)

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
    detect(result_path, model_path, context_norm, body_norm, ind2cat, ind2vad, args)
  else:
    print('source input invalid !!')



# python app/detect.py --source 0 --experiment_path /home/dattran/Documents/dattran/FER-app/weights/model_saved