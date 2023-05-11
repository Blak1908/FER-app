from app.core.settings import get_settings
import gdown
import os

settings = get_settings()

checkpoint_path = settings.CHECKPOINT_PATH
models_path = settings.MODEL_PATH

if not os.path.exists(checkpoint_path):
    os.system(f"mkdir {checkpoint_path}")
    
if not os.path.exists(models_path):
    os.system(f"mkdir {models_path}")

def download_folder_model(id, output_file):

    status = gdown.download_folder(id=id, output=output_file ,quiet=True, use_cookies=False)

    return status


def download_model(id, output_file):
    url = f"https://drive.google.com/uc?id={id}"
    status = gdown.download(url, output_file, quiet=False)
    
    return status