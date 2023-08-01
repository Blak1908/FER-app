from app.modules.user_analyst.deepface.deepface.basemodels.DlibResNet import DlibResNet


def loadModel(checkpoint_path):
    return DlibResNet(checkpoint_path)
