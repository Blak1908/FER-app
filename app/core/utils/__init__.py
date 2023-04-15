import gdown

def get_mode_list(mode: str):
    modes = mode.strip().split(",")
    return modes

def download_model(id, output_file):
    status = False
    try:
        gdown.download_folder(id=id, output=output_file ,quiet=True, use_cookies=False)
        status = True
    except Exception as e:
        print("Error :", e)

    return status