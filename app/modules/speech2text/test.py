import whisper
import sounddevice as sd
import torch
import numpy as np
import librosa
from scipy.io.wavfile import write
import speech_recognition as sr
from df.enhance import enhance, init_df, load_audio, save_audio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_df, df_state, _ = init_df()

model = whisper.load_model("base").to(device)
fs = 48000  # Sample rate
seconds = 3  # Duration of recording


# # Wait for spoken input
# def get_activation_phrase(language_code):
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         print(f"Say 'Xin chào Mai' to start:")
#         audio = r.listen(source)
#     try:
#         activation_phrase = r.recognize_google(audio, language=language_code)
#         print("You said:", activation_phrase)
#         return activation_phrase
#     except sr.UnknownValueError:
#         print("Sorry, I could not understand what you said.")
#         return None
#     except sr.RequestError:
#         print("Sorry, I'm having trouble accessing the speech recognition service.")
#         return None

# # Select language
# language_code = "vi-VN"  
# activation_phrase = None

# while activation_phrase != "Xin chào Mai":
#     activation_phrase = get_activation_phrase(language_code)


# print("Recording...")
# myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
# sd.wait()  # Wait until recording is finished
audio_path = "/home/cuongacpe/test_opensource/whisper/output.wav"
# enhanced_audio = enhance(model_df, df_state, myrecording)

# write('output.wav', fs, enhanced_audio)
# print("Enhanced recording saved: output.wav")
audio, _ = load_audio(audio_path, sr=df_state.sr())
# # Convert stereo audio to mono
# mono_recording = np.mean(myrecording, axis=1)

# # Compute spectrogram
# spectrogram = librosa.feature.melspectrogram(y=mono_recording, sr=fs, n_fft=2048, hop_length=512, n_mels=80)

# # Convert to decibel scale
# spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

# # Transpose the spectrogram (optional, depending on your model's input requirements)
# spectrogram_db = spectrogram_db.T

# # Normalize the spectrogram (optional, depending on your model's input requirements)
# spectrogram_db = (spectrogram_db - np.mean(spectrogram_db)) / np.std(spectrogram_db)

# # Reshape the spectrogram to match the expected input shape of the model
# spectrogram_db = np.expand_dims(spectrogram_db, axis=0)

# # Convert numpy array to tensor
# spectrogram_tensor = torch.from_numpy(spectrogram_db)
# spectrogram_tensor = spectrogram_tensor.view(spectrogram_tensor.size(1), -1)
# spectrogram_tensor = spectrogram_tensor.to(device)
# Now you can pass the spectrogram to your model for further processing
enhanced_audio = enhance(model_df, df_state, audio)

import pdb;pdb.set_trace()
result = model.transcribe(enhanced_audio, language="vi")

print(result['text'])
