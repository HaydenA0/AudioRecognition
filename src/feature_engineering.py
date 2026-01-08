import librosa
import numpy as np
from preprocessor import load_audio
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_features(y, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
    
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)
    
    features = np.vstack([
        mfccs, 
        mfccs_delta, 
        mfccs_delta2,
        spectral_centroid, 
        spectral_bandwidth
    ]).T
    
    return features



def aggregate_features(features):
    mean_feat = np.mean(features, axis=0)
    std_feat = np.std(features, axis=0)
    return np.hstack([mean_feat, std_feat])


def get_char_vector(y, sr, n_mfcc=13):
    features = extract_features(y, sr, n_mfcc)  
    return aggregate_features(features)





if __name__ == "__main__" :
    y, sr = load_audio(PROJECT_ROOT + "/data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac")
    print(get_char_vector(y, sr))

