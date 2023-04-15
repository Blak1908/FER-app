import os
from pydantic import BaseSettings
from starlette.config import Config
from app.core.utils import get_mode_list


config = Config("app/environment/environment.env")
class Settings(BaseSettings):
    environment: str = os.getenv("BULD_ENVIRONMENT", "dev")
    MODES = get_mode_list(config("modes", default=""))
    DEVICE = config("device", default='')
    MAT_PATH = config("mat_path", default="")
    MODEL_PATH = config("model_path", default="")
    RESULT_PATH = config("result_path", default="")
    ID_MAT =config("id_mat", default="")
    ID_MODEL = config("id_model", default="")                    

def get_settings():
    settings = Settings()
    return settings