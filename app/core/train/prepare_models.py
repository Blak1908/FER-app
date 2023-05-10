import os
import torch
from torch.autograd import Variable as V
import torchvision.models as models
import pickle
from functools import partial
from app.core.utils import download_model
from app.core.settings import get_settings


settings = get_settings()


model_name = settings.CONTEXT_MODEL

model_dir = settings.MODEL_PATH
model_resnet18 = settings.RESNET18_MODEL
model_resnet18_py26 = settings.RESNET18_MODEL_PY26


id_resnet18 = settings.ID_RESNET18
id_resnet18_py26 = settings.ID_RESNET18_PY26



def prep_models(context_model='resnet18', body_model='resnet18'):
  ''' Download imagenet pretrained models for context_model and body_model.
  :param context_model: Model to use for conetxt features.
  :param body_model: Model to use for body features.
  :param model_dir: Directory path where to store pretrained models.
  :return: Yolo model after loading model weights
  '''
  
  # Check folder exist
  if not os.path.exists(model_dir):
    os.system(f"mkdir {model_dir}")

  # Check checkpoint resnet18
  if not os.path.isfile(model_resnet18):
    print("Downloading resnet checkpoint path on Drive...")
    
    status = download_model(id_resnet18, model_resnet18)
    if not status:
        print("Cannot download resnet checkpoint path on Drive")
        # download_command = 'wget ' + 'http://places2.csail.mit.edu/models_places365/' + model_name +' -O ' + model_file
        # os.system(download_command)
        
  print("===== Starting create the network architecture =====")
  
  save_file = model_resnet18_py26
  
  pickle.load = partial(pickle.load, encoding="latin1")
  pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
  model = torch.load(model_resnet18, map_location=lambda storage, loc: storage, pickle_module=pickle)
  torch.save(model, save_file)

  # create the network architecture
  model_context = models.__dict__[context_model](num_classes=365)
  checkpoint = torch.load(save_file, map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!
  if context_model == 'densenet161':
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    state_dict = {str.replace(k,'norm.','norm'): v for k,v in state_dict.items()}
    state_dict = {str.replace(k,'conv.','conv'): v for k,v in state_dict.items()}
    state_dict = {str.replace(k,'normweight','norm.weight'): v for k,v in state_dict.items()}
    state_dict = {str.replace(k,'normrunning','norm.running'): v for k,v in state_dict.items()}
    state_dict = {str.replace(k,'normbias','norm.bias'): v for k,v in state_dict.items()}
    state_dict = {str.replace(k,'convweight','conv.weight'): v for k,v in state_dict.items()}
  else:
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()} # the data parallel layer will add 'module' before each layer name
  model_context.load_state_dict(state_dict)
  model_context.eval()
  model_context.cpu()
  torch.save(model_context, os.path.join(model_dir, 'context_model' + '.pth'))
  
  print ('completed preparing context model')

  model_body = models.__dict__[body_model](pretrained=True)
  model_body.cpu()
  torch.save(model_body, os.path.join(model_dir, 'body_model' + '.pth'))

  print ('completed preparing body model')
  return model_context, model_body



if __name__ == "__main__":
  print("Test prepare model: ")
  try:
    context_model, body_model = prep_models(model_name, model_name)
    print("Load model success! ")
  except Exception as e:
    print("Error: ", e)
