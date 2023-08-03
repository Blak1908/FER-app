import os, sys
sys.path.append("app/modules/speech2text/")
from whisper import load_model
from whisper.audio import load_audio as load_audio_whisper

from app.core.settings import get_settings
from app.core.utils import download_model
settings = get_settings()

s2t_model_name = settings.WHISPER_MODEL_PATH_NAME
id_s2t = settings.ID_WHISPER_CHECKPOINT
device = settings.DEVICE
weights_path = settings.CHECKPOINT_PATH

if not os.path.exists(weights_path):
    os.makedirs(weights_path)
# Download checkpoint whisper
s2t_model_path = os.path.join(weights_path, s2t_model_name)
if not os.path.isfile(s2t_model_path):
    print("Download whisper model.")
    download_model(id_s2t, s2t_model_path)

class SPEECH2TEXT_TRANSFORMER():
    def __init__(self,device, s2t_model_path):
        print("Init SPEECH2TEXT_TRANSFORMER")
        self.device = device
        self.model = model = load_model("base", s2t_model_path=s2t_model_path).to(device)

    def forward(self, audio):
        audio = load_audio_whisper(audio)
        result = self.model.transcribe(audio, language="vi",fp16=False)
        return result
    
        

s2t_transformer =  SPEECH2TEXT_TRANSFORMER(device=device, s2t_model_path=s2t_model_path)

def s2t_processing(audio):
    result = s2t_transformer.forward(audio)
    return result

# uvicorn s2t_api:app --host 172.17.12.221 --port 8000