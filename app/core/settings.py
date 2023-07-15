import os
from pydantic import BaseSettings
from starlette.config import Config


config = Config("app/environment/environment.env")
class Settings(BaseSettings):
    environment: str = os.getenv("BULD_ENVIRONMENT", "dev")
    DEVICE = config("device", default='')
    CHECKPOINT_PATH = config("checkpoint_path", default="")
    MAT_PATH = config("mat_path", default="")
    MODEL_PATH = config("model_path", default="")
    RESULT_PATH = config("result_path", default="")
    RESNET18_MODEL = config("resnet18_model", default="")
    RESNET18_MODEL_PY26 = config("resnet18_model_py26", default="")
    CONTEXT_MODEL = BODY_MODEL = config("context_model",default="")
    WHISPER_PATH = config("whisper_path", default="")
    
    ID_MAT =config("id_mat", default="")
    ID_MODEL = config("id_model", default="")
    ID_RESNET18 = config("id_resnet18", default="")
    ID_RESNET18_PY26 = config("id_resnet18_py36", default="")
    ID_WHISPER_CHECKPOINT = config("id_whisper_checkpoint", default="")                    
    TEMP_PATH = config("temp_path", default="")
    SCR_FOLDER_NAME = config("scr_folder_name", default="")
    
    
def get_settings():
    settings = Settings()
    return settings