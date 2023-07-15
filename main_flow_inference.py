from app.modules.speech2text.s2t_transformer import s2t_processing
from app.core.utils import get_activation_phrase
import sounddevice as sd

fs = 48000  # Sample rate
seconds = 5  # Duration of recording
seconds_statement = 10

while True:
    print("Start listening in 5s...")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    result = s2t_processing(myrecording)
    if result == "xin chào mai":
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished
        statement_s2t = s2t_processing(myrecording)
    if result == "tạm biệt mai":
        print("Tạm biệt")
        break
        
