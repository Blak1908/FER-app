from app.modules.user_recognition.deepface.deepface import DeepFace
from app.modules.user_recognition.deepface.deepface.detectors import FaceDetector, DlibWrapper
import os
import glob
from app.modules.user_recognition.sub_function import calculate_mean, most_common_element
from app.core.settings import get_settings
from app.modules.user_recognition.deepface.deepface.DeepFace import build_model

settings = get_settings()

src_imgs_path = settings.SCR_FOLDER_NAME
weights_path = settings.CHECKPOINT_PATH

detector_backend_checkpoint_name = settings.DECTECOR_MODEL_CHECKPOINT_NAME
detector_backend_name = settings.DETECTOR_BACKEND_NAME
deepface_model_checkpoint_name = settings.DEEPFACE_MODEL_CHECKPOINT_NAME
deepface_model_name = settings.DEEPFACE_MODEL_NAME
emotion_checkpoint_name = settings.EMOTION_CHECKPOINT_NAME
emotion = settings.EMOTION
age_checkpoint_name = settings.AGE_CHECKPOINT_NAME
age = settings.AGE
gender_checkpoint_name = settings.GENDER_CHECKPOINT_NAME
gender = settings.GENDER
race_checkpoint_name = settings.RACE_CHECKPOINT_NAME
race = settings.RACE

if not os.path.exists(weights_path):
    os.makedirs(weights_path)

detector_backend_checkpoint_path = os.path.join(weights_path,detector_backend_checkpoint_name)
deepface_model_checkpoint_path = os.path.join(weights_path,deepface_model_checkpoint_name)
emotion_checkpoint_path = os.path.join(weights_path, emotion_checkpoint_name)
age_checkpoint_path = os.path.join(weights_path, age_checkpoint_name)
gender_checkpoint_path = os.path.join(weights_path, gender_checkpoint_name)
race_checkpoint_path = os.path.join(weights_path, race_checkpoint_name)

class DeepFaceProcessingModel:
    def __init__(self, temp_imgs_path):
        self.temp_imgs_path = temp_imgs_path
        self.face_detector =  DlibWrapper.build_model(detector_backend_checkpoint_path)
        self.face_respresent = build_model(deepface_model_name,deepface_model_checkpoint_path)
        self.user_analysis_model = {}
        self.user_analysis_model["emotion"] = build_model(emotion, emotion_checkpoint_path)
        self.user_analysis_model["age"] = build_model(age,age_checkpoint_path)
        self.user_analysis_model["gender"] = build_model(gender, gender_checkpoint_path)
        self.user_analysis_model["race"] = build_model(race, race_checkpoint_path)
        self.max = 0

    def get_images_list_path(self):

        # while len(self.frames) < self.num_frames:
        #     # Generate a unique filename for the frame
        #     filename = f"frame{len(self.frames) + 1}.jpg"
        #     filepath = os.path.join(self.output_folder, filename)

        #     # Append the file path to the list of frames
        #     self.frames.append(filepath)
        imgs_path_list = sorted(glob.glob(os.path.join(self.temp_imgs_path,"*.jpg")))
        return imgs_path_list

    def face_verification(self, db_path, models, distances, images_path_list):
        names_list = []  # User's name
        max_count = 0

        # Face verification
        for frame_path in images_path_list:
            sub_names_list = []
            result_find = DeepFace.find(
                img_path=frame_path,
                db_path=db_path,
                model_name=models,
                detector_backend=detector_backend_name,
                detector_model = self.face_detector,
                predict_model = self.face_respresent,
                distance_metric=distances
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

        # output_file = "default_details.json" if flag else "request_details.json"

        # with open(output_file, 'w') as file:
        #     json.dump(details, file, indent=4)

        return details

    def forward(self, db_path, models, distances, flag):
        images_path_list  = self.get_images_list_path()
        names_list = self.face_verification(db_path, models, distances, images_path_list)
        emotion, age, gender, race = self.analyze_faces(images_path_list)
        return self.get_final_results(flag, names_list, emotion, age, gender, race)
    
deepface_process = DeepFaceProcessingModel(temp_imgs_path=src_imgs_path)

def deepface_processing(db_path, models, distances, flag):
    result = deepface_process.forward(db_path, models, distances, flag)
    return result