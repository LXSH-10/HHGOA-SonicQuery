import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000  # 16kHz — the sample rate Sarvam works best with
seconds = 5

print("Recording for 5 seconds... speak your test question now")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()
write("test.wav", fs, recording)
print("Saved as test.wav")