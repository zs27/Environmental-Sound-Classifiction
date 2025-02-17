import sys
import gammatone.fftweight
import gammatone.gtgram
import nnAudio.features
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import gammatone
from gammatone.filters import make_erb_filters, erb_filterbank, centre_freqs
import gammatone.plot
import nnAudio
import torch


def gtsc(audio_file: str) -> None:
    # Load the audio signal
    y, sr = librosa.load(audio_file, sr=22050)
    print(len(y), sr)

    # y = y[2*sr:3*sr]
    # y = y[3*sr-551:3*sr]
    # y = y[:41]

    # Gammatone filterbank
    def apply_gammatone_filterbank(y, sr, num_bands=60, low_freq=20):
        # if high_freq is None:
        #     high_freq = sr / 2
        # centre_freqs = np.linspace(low_freq, high_freq, num_bands)

        centre_f = centre_freqs(sr, num_bands, low_freq)
        erb_filters = make_erb_filters(sr, centre_f)
        fcoefs = np.flipud(erb_filters)
        y_filtered = erb_filterbank(y, fcoefs)
        return y_filtered

        # twin = 0.025 # 0.08
        # thop = twin / 2
        # channels = 1024
        # fmin = 20
        # y_filtered = gammatone.gtgram.gtgram(y, sr, twin, thop, num_bands, low_freq)
        return y_filtered

    def gammatone_filterbank(audio_signal, num_filters, fs):
        # Create Gammatone filterbank
        filters = make_erb_filters(fs, num_filters)
        filtered_signals = erb_filterbank(audio_signal, filters)
        return filtered_signals

    # Half Wave Rectifier (HWR)
    def half_wave_rectifier(y):
        return np.maximum(y, 0)

    # Teager Energy Operator (TEO)
    def teager_energy_operator(y):
        energy = y[1:-1] ** 2 - y[:-2] * y[2:]
        # When the central sample is very small (or close to zero)
        # and the surrounding samples are NON-zero,
        # e.g. [1, 1, 0, 1, 1] -> [1, -1, 1]
        # So we replace the negative values with zero.
        energy[energy < 0] = 0
        return energy

    # Short-term Averaging
    # def short_term_averaging(y, frame_size=3, hop_size=1):
    # # def short_term_averaging(y, frame_size=1024, hop_size=512):
    #     return librosa.feature.rms(y=y, frame_length=frame_size, hop_length=hop_size).squeeze()

    # def short_term_averaging(energy_signal, window_size=3):
    #     # Apply short-term averaging
    #     averaged_energy = np.convolve(energy_signal, np.ones(window_size)/window_size, mode='same')
    #     return averaged_energy

    # def short_term_averaging(signal, window_size=1024, hop_size=512):
    #     window_count = (len(signal) - window_size) // hop_size + 1
    #     result = np.zeros(window_count)

    #     for i in range(window_count):
    #         start_index = i * hop_size
    #         end_index = start_index + window_size
    #         window = signal[start_index:end_index]
    #         result[i] = np.mean(window)

    #     return result

    def short_term_averaging(window_size: int, hop_size: int):
        def sta(signal: np.ndarray) -> np.ndarray:
            window_count = (len(signal) - window_size) // hop_size + 1
            result = np.zeros(window_count)

            for i in range(window_count):
                start_index = i * hop_size
                end_index = start_index + window_size
                window = signal[start_index:end_index]
                result[i] = np.mean(window)

            return result
        return sta

    # Logarithmic Compression
    def logarithmic_compression(y):
        return np.log1p(y)

    def normalise(y: np.ndarray):
        normalization_factor: float = 1 / np.max(np.abs(y))
        print("Normalisation factor:", normalization_factor)
        return y * normalization_factor # - 0.5

    # print(normalization_factor)

    w = 1
    FRAMES_PER_SEGMENT = 101 - 1 # 41  # - 1  # 41 frames ~= 950 ms segment length @ 22050 Hz
    WINDOW_SIZE = w * FRAMES_PER_SEGMENT  # 23 ms per frame @ 22050 Hz
    STEP_SIZE = w * FRAMES_PER_SEGMENT // (2 if False else 10)

    N_FFT = 1024
    HOP_LEN = 512

    y = normalise(y)

    # print(y.dtype)
    # y = torch.tensor(y)
    # print(y.dtype)

    # Apply Gammatone filterbank
    y_filtered = apply_gammatone_filterbank(y, sr)
    # y_filtered = gammatone_filterbank(y, 60, sr)
    # y_spec2 = gammatone.gtgram.gtgram(y, sr, 1024 / sr, 512 / sr, 60, 20) 
    # y_spec2 = gammatone.fftweight.fft_gtgram(y.numpy(), sr, 1024 / sr, 512 / sr, 60, 20) 
    # y_spec2 = nnAudio.features.Gammatonegram(sr=int(sr), n_bins=60, n_fft=1024)(y).squeeze().numpy()
    y_spec2 = nnAudio.features.Gammatonegram(sr=int(sr), n_bins=60, n_fft=N_FFT, hop_length=HOP_LEN)(torch.tensor(y).float()).squeeze().numpy()
    print(y_filtered.shape, np.min(y_filtered), np.max(y_filtered))

    # y_spec = librosa.power_to_db(y_filtered, top_db=None)
    # y_spec = np.sqrt(np.apply_along_axis(short_term_averaging, 1, np.square(y_filtered)))
    # y_spec = librosa.power_to_db(y_spec, ref=np.max)
    # y_spec = np.log1p(y_filtered)
    y_spec = y_spec2; 
    # librosa.feature.melspectrogram(
    #     y=y, sr=sr, n_fft=WINDOW_SIZE, hop_length=STEP_SIZE, n_mels=60
    # )

    y_spec = librosa.power_to_db(y_spec, ref=np.max, top_db=None)
    y_spec = y_spec[:, 1:-1]
    y_spec /= np.max(np.abs(y_spec))
    y_spec += 1

    
    print(y_spec.shape, np.min(y_spec), np.max(y_spec))
    # np.savetxt("spec.txt", y_spec)
    # np.savetxt("spec2.txt", y_spec2)

    # Apply Half Wave Rectifier
    y_hwr = half_wave_rectifier(y_filtered)
    print(y_hwr.shape, np.min(y_hwr), np.max(y_hwr))

    # Apply Teager Energy Operator
    y_teo = np.apply_along_axis(teager_energy_operator, 1, y_hwr)
    print(y_teo.shape, np.min(y_teo), np.max(y_teo))

    # Apply Short-term Averaging
    y_sta = np.apply_along_axis(short_term_averaging(N_FFT, HOP_LEN), 1, y_teo)
    # y_sta = librosa.feature.rms(S=y_teo, frame_length=3, hop_length=1)
    print(y_sta.shape, np.min(y_sta), np.max(y_sta))

    # Apply Logarithmic Compression
    y_log = librosa.power_to_db(y_sta, ref=np.max, top_db=None) # np.log1p(y_sta * ((np.e - 1) / np.max(y_sta)))
    y_log /= np.max(np.abs(y_log))
    y_log += 1
    # y_log = librosa.power_to_db(y_sta, ref=np.max)
    # y_log = librosa.power_to_db(y_sta, top_db=None)
    # y_log = np.apply_along_axis(librosa.power_to_db, 1, y_sta)
    print(y_log.shape, np.min(y_log), np.max(y_log))

    y_spec = y_spec # normalise(y_spec)
    y_log = y_log # normalise(y_log)

    fused = y_log # * 0.5 + y_spec * 0.5
    # fused = librosa.power_to_db(y_sta * 0.5 + y_filtered[:, 1:-1] * 0.5)
    print(fused.shape, np.min(fused), np.max(fused))

    # print(y_spec[30] == np.min(y_spec[30]))
    # print(y_log[30] == np.min(y_log[30]))

    n = 9

    # Plot the results
    plt.figure(figsize=(6, n + 1))

    plt.subplot(n, 1, 1)
    plt.plot(y)
    plt.title("Audio Signal")

    plt.subplot(n, 1, 2)
    plt.imshow(y_filtered, aspect="auto", origin="lower")
    plt.title("Gammatone Filterbank (Waveforms)")

    plt.subplot(n, 1, 3)
    plt.imshow(y_spec, aspect="auto", origin="lower")
    plt.title("GTSC")

    plt.subplot(n, 1, 4)
    plt.imshow(y_hwr, aspect="auto", origin="lower")
    plt.title("Half Wave Rectifier (HWR)")

    plt.subplot(n, 1, 5)
    plt.imshow(y_teo, aspect="auto", origin="lower")
    plt.title("TEO (Teager Energy Operator)")

    plt.subplot(n, 1, 6)
    plt.imshow(y_sta, aspect="auto", origin="lower")
    plt.title("Short-term Averaging")

    plt.subplot(n, 1, 7)
    plt.imshow(y_log, aspect="auto", origin="lower")
    plt.title("Logarithmic Compression")

    plt.subplot(n, 1, 8)
    plt.imshow((fused), aspect="auto", origin="lower")
    plt.title("Fused")

    plt.subplot(n, 1, 9)
    plt.imshow(librosa.feature.delta(fused), aspect="auto", origin="lower")
    plt.title("Delta")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 3 + 1))
    plt.subplot(3, 1, 1)
    plt.imshow(librosa.power_to_db(y_spec), aspect="auto", origin="lower")
    plt.title("Pow2DB GTSC (Easier to See)")
    plt.subplot(3, 1, 2)
    plt.imshow(librosa.power_to_db(y_log), aspect="auto", origin="lower")
    plt.title("Pow2DB TEO-GTSC (Easier to See)")
    plt.subplot(3, 1, 3)
    plt.imshow(librosa.power_to_db(fused), aspect="auto", origin="lower")
    plt.title("Pow2DB Fused (Easier to See)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(np.concatenate((y_spec, y_log, fused), axis=0), aspect="auto", origin="lower")
    plt.title("On the same scale")
    plt.tight_layout()
    plt.subplot(1, 2, 2)
    plt.imshow(librosa.power_to_db(np.concatenate((y_spec, y_log, fused), axis=0)), aspect="auto", origin="lower")
    plt.title("On the same scale (Pow2DB)")
    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(6, 3))
    # plt.subplot(1, 2, 1)
    # plt.imshow(y_spec, aspect="auto", origin="lower")
    # plt.title("Spec")
    # plt.tight_layout()
    # plt.subplot(1, 2, 2)
    # plt.imshow(y_spec2, aspect="auto", origin="lower")
    # plt.title("Spec2")
    # plt.tight_layout()
    # plt.show()

    gammatone.plot.render_audio_from_file(audio_file, 5, gammatone.fftweight.fft_gtgram)
    s = 0
    segments = []
    while (logspec := fused[:, s * STEP_SIZE : s * STEP_SIZE + WINDOW_SIZE]).shape[
        1
    ] == WINDOW_SIZE:
        # wv = y[s * STEP_SIZE : s * STEP_SIZE + WINDOW_SIZE]
        segments.append(
            (
                s * STEP_SIZE,
                s * STEP_SIZE + WINDOW_SIZE,
                logspec.shape,
                np.mean(logspec),
                np.mean(logspec) > 0.05, # -60,
            )
        )  # 0.0005)) # -85.0
        s += 1

    if fused.shape[1] % STEP_SIZE != 0:
        # wv = y[s * STEP_SIZE : s * STEP_SIZE + WINDOW_SIZE]
        segments.append(
            (
                fused.shape[1] - WINDOW_SIZE,
                fused.shape[1],
                logspec.shape,
                np.mean(logspec),
                np.mean(logspec) > -60,
            )
        )  # 0.0005)) # -85.0
        s += 1
        

    print(len(segments), segments)


def log_mel(audio_file):
    # Load the audio signal
    y, sr = librosa.load(audio_file, sr=22050)
    print(len(y), sr)

    melspec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=60
    )

    logspec = librosa.power_to_db(melspec)

    delta = librosa.feature.delta(logspec)

    print(logspec.shape)

    plt.figure(figsize=(6, 4))

    plt.subplot(3, 1, 1)
    plt.imshow(melspec, aspect="auto", origin="lower")
    plt.title("Mel Spec")

    plt.subplot(3, 1, 2)
    plt.imshow(logspec, aspect="auto", origin="lower")
    plt.title("Log Mel")

    plt.subplot(3, 1, 3)
    plt.imshow(delta, aspect="auto", origin="lower")
    plt.title("Delta")

    plt.tight_layout()
    plt.show()


def main() -> None:
    if len(sys.argv) != 2:
        print("No file name specified!")
    filename = sys.argv[1]
    gtsc(filename)
    log_mel(filename)
    return


if __name__ == "__main__":
    main()
