from whisper import load_model
from whisper.audio import load_audio as load_audio_whisper
import sounddevice as sd
import torch
import numpy as np
import librosa
from scipy.io.wavfile import write
import speech_recognition as sr
from time import time
from df.enhance import enhance, init_df, load_audio, save_audio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_df, df_state, _ = init_df()
model = load_model("base").to(device)
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
# write('output.wav', fs, myrecording)

start = time()
audio_path = "/home/cuongacpe/test_opensource/whisper/output1.wav"

audio, _ = load_audio(audio_path, sr=df_state.sr())

temp_path = "temp_audio.wav"
enhanced_audio = enhance(model_df, df_state, audio)
save_audio(temp_path, enhanced_audio, df_state.sr())
stop = time()
print("======", stop-start)
# import pdb;pdb.set_trace()
start = time()
audio_whisper = load_audio_whisper(temp_path)
result = model.transcribe(audio_whisper, language="vi")
stop = time()
print("======", stop-start)
print(result['text'])
