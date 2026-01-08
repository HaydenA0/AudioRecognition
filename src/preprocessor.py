import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_audio(audio_path: str):
    y, sr = librosa.load(audio_path, sr=None)
    return y, sr

def plot_signal_time(y, sr):
    t = np.arange(len(y)) / sr
    plt.figure()
    plt.plot(t, y)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Time Domain Signal")
    plt.show()

def plot_signal_frequency(y, sr):
    Y = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(Y), 1 / sr)
    magnitude = np.abs(Y)

    plt.figure()
    plt.plot(freqs[:len(freqs)//2], magnitude[:len(magnitude)//2])
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title("Frequency Domain Signal")
    plt.show()


def print_general_info(y, sr):
    duration = len(y) / sr
    print(f"Duration: {duration:.2f} s")
    print(f"Sampling rate: {sr} Hz")

def normalize_signal(y):
    y_norm = librosa.util.normalize(y)
    return y_norm



def remove_silence(y, top_db=20):
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed

def get_frames(y, sr, frame_ms=25, overlap=0.5):
    frame_length = int(sr * frame_ms / 1000)
    hop_length = int(frame_length * (1 - overlap))
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    return frames


def preprocess_audio(y, sr, show = False):
    if show :
        print_general_info(y, sr)
    y_norm = normalize_signal(y)
    y_voiced = remove_silence(y_norm, sr)
    frames = get_frames(y_voiced, sr)
    if show :
        plot_signal_time(y_voiced, sr)
        plot_signal_frequency(y_voiced, sr)
        print(f"frames = {len(frames)}")
    return y_voiced, sr

if __name__ == "__main__" :
    y, sr = load_audio(PROJECT_ROOT + "/data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac")
    preprocess_audio(y, sr, show=True)
