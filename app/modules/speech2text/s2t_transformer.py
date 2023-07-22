import os, sys
sys.path.append("app/modules/speech2text/")
from whisper import load_model
from whisper.audio import load_audio as load_audio_whisper

from app.core.settings import get_settings

settings = get_settings()

s2t_model_path = settings.WHISPER_PATH
id_s2t = settings.ID_WHISPER_CHECKPOINT
device = settings.DEVICE

# Download checkpoint whisper
if os.path.isfile(s2t_model_path):
    print()

class SPEECH2TEXT_TRANSFORMER():
    def __init__(self,device, t2s_model_path):
        print("Init SPEECH2TEXT_TRANSFORMER")
        self.device = device
        self.model = model = load_model("base").to(device)

    def forward(self, audio):
        audio = load_audio_whisper(audio)
        result = self.model.transcribe(audio, language="vi")
        return result
    
        

s2t_transformer =  SPEECH2TEXT_TRANSFORMER(device=device, t2s_model_path=s2t_model_path)

def s2t_processing(audio):
    result = s2t_transformer.forward(audio)
    return result
         