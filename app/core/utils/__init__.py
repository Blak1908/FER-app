from app.core.settings import get_settings
import gdown
import os
import string

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


def get_activation_phrase():
    return

def remove_special_characters(input_string):
    # Khởi tạo một chuỗi chứa các ký tự đặc biệt
    special_characters = string.punctuation
    
    # Tạo một chuỗi mới chỉ chứa các ký tự không phải là ký tự đặc biệt
    result_string = ''.join(char for char in input_string if char not in special_characters)
    
    return result_string