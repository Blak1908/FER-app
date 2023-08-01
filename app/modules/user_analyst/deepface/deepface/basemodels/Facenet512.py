import os
import gdown
from app.modules.user_recognition.deepface.deepface.basemodels import Facenet
from app.modules.user_recognition.deepface.deepface.commons import functions


def loadModel(
    # url="https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5",
    checkpoint_path,
):

    model = Facenet.InceptionResNetV2(dimension=512)

    # -------------------------

    # home = functions.get_deepface_home()

    # if os.path.isfile(home + "/.deepface/weights/facenet512_weights.h5") != True:
    #     print("facenet512_weights.h5 will be downloaded...")

    #     output = home + "/.deepface/weights/facenet512_weights.h5"
    #     gdown.download(url, output, quiet=False)

    # -------------------------

    model.load_weights(checkpoint_path)

    # -------------------------

    return model
