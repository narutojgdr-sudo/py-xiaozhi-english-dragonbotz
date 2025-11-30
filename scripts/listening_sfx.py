import wave, numpy as np, sounddevice as sd, sys
f = "src/sfx/ready.wav"
with wave.open(f, "rb") as wf:
    channels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    fr = wf.getframerate()
    frames = wf.readframes(wf.getnframes())

if sampwidth == 1:
    data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32); data = (data-128.0)/128.0
elif sampwidth == 2:
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)/32768.0
elif sampwidth == 4:
    data = np.frombuffer(frames, dtype=np.int32).astype(np.float32)/2147483648.0
else:
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)/32768.0

if channels > 1:
    data = data.reshape(-1, channels)

print("Playing via default output...")
sd.play(data, samplerate=fr)
sd.wait()
print("Done")