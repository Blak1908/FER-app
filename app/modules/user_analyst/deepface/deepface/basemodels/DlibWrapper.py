from app.modules.user_recognition.deepface.deepface.basemodels.DlibResNet import DlibResNet


def loadModel(checkpoint_path):
    return DlibResNet(checkpoint_path)
