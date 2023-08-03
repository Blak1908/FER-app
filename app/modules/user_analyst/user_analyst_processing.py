from app.modules.user_analyst.deepface.deepface import DeepFace
from app.modules.user_analyst.deepface.deepface.detectors import FaceDetector, DlibWrapper
import os
import glob
from app.modules.user_analyst.ultils import calculate_mean, most_common_element
from app.core.settings import get_settings
from app.modules.user_analyst.deepface.deepface.DeepFace import build_model
from app.core.utils import download_model

settings = get_settings()

src_imgs_path = settings.SCR_FOLDER_NAME
weights_path = settings.CHECKPOINT_PATH

id_facenet_model = settings.ID_FACENET_MODEL
id_dlib_model = settings.ID_DLIB_MODEL
id_emotion_model = settings.ID_EMOTION_MODEL
id_age_model = settings.ID_AGE_MODEL
id_gender_model = settings.ID_GENDER_MODEL
id_race_model = settings.ID_RACE_MODEL

detector_backend_checkpoint_name = settings.DECTECOR_MODEL_CHECKPOINT_NAME
detector_backend_name = settings.DETECTOR_BACKEND_NAME
face_analyst_model_checkpoint_name = settings.FACE_ANALYST_MODEL_CHECKPOINT_NAME
face_analyst_model_name = settings.FACE_ANALYST_MODEL_NAME
emotion_checkpoint_name = settings.EMOTION_CHECKPOINT_NAME
emotion_model_name = settings.EMOTION_MODEL_NAME
age_checkpoint_name = settings.AGE_CHECKPOINT_NAME
age_model_name = settings.AGE_MODEL_NAME
gender_checkpoint_name = settings.GENDER_CHECKPOINT_NAME
gender_model_name = settings.GENDER_MODEL_NAME
race_checkpoint_name = settings.RACE_CHECKPOINT_NAME
race_model_name = settings.RACE_MODEL_NAME
database_image_path = settings.DATABASE_IMAGE_PATH
deepface_distance = settings.DEEPFACE_DISTANCE

if not os.path.exists(weights_path):
    os.makedirs(weights_path)

face_analyst_model_checkpoint_path = os.path.join(weights_path,face_analyst_model_checkpoint_name)
if not os.path.isfile(face_analyst_model_checkpoint_path):
    print("Download facenet model.")
    download_model(id_facenet_model, face_analyst_model_checkpoint_path)
detector_backend_checkpoint_path = os.path.join(weights_path,detector_backend_checkpoint_name)
if not os.path.isfile(detector_backend_checkpoint_path):
    print("Download dlib model.")
    download_model(id_dlib_model, detector_backend_checkpoint_path)
emotion_checkpoint_path = os.path.join(weights_path, emotion_checkpoint_name)
if not os.path.isfile(emotion_checkpoint_path):
    print("Download emotion model.")
    download_model(id_emotion_model, emotion_checkpoint_path)
age_checkpoint_path = os.path.join(weights_path, age_checkpoint_name)
if not os.path.isfile(age_checkpoint_path):
    print("Download age model.")
    download_model(id_age_model, age_checkpoint_path)
gender_checkpoint_path = os.path.join(weights_path, gender_checkpoint_name)
if not os.path.isfile(gender_checkpoint_path):
    print("Download gender model.")
    download_model(id_gender_model, gender_checkpoint_path)
race_checkpoint_path = os.path.join(weights_path, race_checkpoint_name)
if not os.path.isfile(race_checkpoint_path):
    print("Download race model.")
    download_model(id_race_model, race_checkpoint_path)

class DeepFaceProcessingModel:
    def __init__(self):
        self.face_detector =  DlibWrapper.build_model(detector_backend_checkpoint_path)
        self.face_respresent = build_model(face_analyst_model_name,face_analyst_model_checkpoint_path)
        self.user_analysis_model = {}
        self.user_analysis_model["emotion"] = build_model(emotion_model_name, emotion_checkpoint_path)
        self.user_analysis_model["age"] = build_model(age_model_name,age_checkpoint_path)
        self.user_analysis_model["gender"] = build_model(gender_model_name, gender_checkpoint_path)
        self.user_analysis_model["race"] = build_model(race_model_name, race_checkpoint_path)
        self.database_image_path = database_image_path
        self.distance_calculator = deepface_distance
        self.face_analyst_model = face_analyst_model_name
        self.max = 0

    def get_images_list_path(self, src_folder_path):
        imgs_path_list = sorted(glob.glob(os.path.join(src_folder_path,"*.jpg")))
        return imgs_path_list

    def face_verification(self, images_path_list, name=None):
        names_list = []  # User's name
        max_count = 0
        flag = True
        # Face verification
        for frame_path in images_path_list:
            sub_names_list = []
            if name != None:
                result_find = DeepFace.find(
                    img_path=frame_path,
                    db_path=self.database_image_path,
                    model_name=self.face_analyst_model,
                    detector_backend=detector_backend_name,
                    detector_model = self.face_detector,
                    predict_model = self.face_respresent,
                    distance_metric=self.distance_calculator
                )
            else:
                result_find = DeepFace.find(
                    img_path=frame_path,
                    db_path=self.database_image_path,
                    model_name=self.face_analyst_model,
                    detector_backend=detector_backend_name,
                    detector_model = self.face_detector,
                    predict_model = self.face_respresent,
                    distance_metric=self.distance_calculator,
                    silent=True
                    
                )

            for df_data in result_find:
                name = df_data['label'][0]
                name = name.split("-")[0]
                sub_names_list.append(name)
            if len(result_find) > max_count:
                max_count = len(result_find)
                names_list = sub_names_list

        self.max = max_count  # Set max to the maximum number of faces recognized
        return names_list
    
    def analyze_faces(self, images_path_list):
        result_emotions = []
        result_ages = []
        result_genders = []
        result_races = []

        temp_emotions_lists = [[] for _ in range(self.max)]
        temp_ages_lists = [[] for _ in range(self.max)]
        temp_genders_lists = [[] for _ in range(self.max)]
        temp_races_lists = [[] for _ in range(self.max)]

        for frame_path in images_path_list:
            result = DeepFace.analyze(frame_path, actions=['emotion', 'age', 'race', 'gender'],
                                    enforce_detection=False, detector_backend=detector_backend_name,
                                    detector_model = self.face_detector, analysis_model=self.user_analysis_model)
            if len(result) == self.max:
                for i in range(self.max):
                    temp_emotions_lists[i].append(result[i]['dominant_emotion'])
                    temp_ages_lists[i].append(result[i]['age'])
                    temp_genders_lists[i].append(result[i]['dominant_gender'])
                    temp_races_lists[i].append(result[i]['dominant_race'])

        for i in range(self.max):
            result_emotions.append(most_common_element(temp_emotions_lists[i]))
            result_ages.append(calculate_mean(temp_ages_lists[i]))
            result_genders.append(most_common_element(temp_genders_lists[i]))
            result_races.append(most_common_element(temp_races_lists[i]))

        return result_emotions, result_ages, result_genders, result_races
    def get_final_results(self, flag, names_list, result_emotions, result_ages, result_genders, result_races):
        if flag:
            details = {
                "name": names_list,
                "emotion": result_emotions
            }
        else:
            request = input("Enter the parameter(s) you want to choose (age, gender, race), separated by spaces: ").split()
            details = {"name": names_list,
                       "emotion": result_emotions}
            for param in request:
                if param == 'age':
                    details['age'] = result_ages
                elif param == 'gender':
                    details['gender'] = result_genders
                elif param == 'race':
                    details['race'] = result_races

        return details

    def forward(self, flag, src_folder_path, name):
        images_path_list  = self.get_images_list_path(src_folder_path)
        names_list = self.face_verification(images_path_list, name)
        emotion, age, gender, race = self.analyze_faces(images_path_list)
        return self.get_final_results(flag, names_list, emotion, age, gender, race)
    
deepface_process = DeepFaceProcessingModel()

def deepface_processing(flag=True, src_folder_path=None, name=None):
    result = deepface_process.forward(flag, src_folder_path, name)
    return result
