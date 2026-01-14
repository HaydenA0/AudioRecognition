import librosa
import numpy as np
import torch

def extract_features(y, sr):
    pitch = librosa.yin(y=y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    rms = librosa.feature.rms(y=y)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    return {
        "pitch": pitch,
        "rms": rms,
        "mfccs": mfccs,
        "contrast": contrast,
        "zcr": zcr
    }

def aggregate_features(feat_dict):
    final_vector = []
    for key in feat_dict:
        val = feat_dict[key]
        mean = np.mean(val, axis=1)
        std = np.std(val, axis=1)
        final_vector.extend(mean)
        final_vector.extend(std)
    return np.array(final_vector)

def get_char_vector(y, sr):
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    features = extract_features(y_trimmed, sr)  
    return aggregate_features(features)




def get_cnn_features(audio_path, max_len=256):
    # Load at 16kHz for consistency
    y, sr = librosa.load(audio_path, sr=16000)
    y, _ = librosa.effects.trim(y, top_db=20)
    y = librosa.util.normalize(y)
    
    # 1. MEL SPECTROGRAM (Instead of MFCC)
    # n_mels=128 gives way more frequency detail
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=256)
    
    # 2. CONVERT TO DECIBELS (This is how humans hear)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 3. PADDING / TRUNCATING
    if mel_db.shape[1] < max_len:
        pad_width = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :max_len]
        
    return torch.tensor(mel_db, dtype=torch.float32)

if __name__ == "__main__" :
    y, sr = load_audio(PROJECT_ROOT + "/data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac")
    print(get_char_vector(y, sr))

