import numpy as np
import librosa
import matplotlib.pyplot as plt

# Load an audio file
y, sr = librosa.load(librosa.ex('trumpet'), sr=None)

# Parameters for framing
frame_length = int(0.025 * sr)  # 25 ms
hop_length = int(frame_length / 2)  # 50% overlap

# Create frames
frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)

# Apply Hamming window to each frame
hamming_window = np.hamming(frame_length)
windowed_frames = frames * hamming_window[:, np.newaxis]

# Plot the original and windowed frame
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(frames[:, 10])
plt.title('Original Frame')

plt.subplot(2, 1, 2)
plt.plot(windowed_frames[:, 10])
plt.title('Windowed Frame (Hamming)')

plt.tight_layout()
plt.show()
