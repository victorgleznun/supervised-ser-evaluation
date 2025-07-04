import os
import numpy as np
import librosa
from tqdm import tqdm
from Config.settings import RAVDESS_DIR, SAVEE_DIR, TESS_DIR

# Mapas de etiqueta por corpus
EMOTION_MAPS = {
    'ravdess': {'03':'alegría','04':'tristeza','05':'ira','06':'miedo'},
    'savee':   {'h':'alegría','s':'tristeza','a':'ira','f':'miedo'},
    'tess':    {'happy':'alegría','sad':'tristeza','angry':'ira','fear':'miedo'}
}

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    if len(audio) < 1000:
        return None
    feats = []
    # MFCC (13)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    feats.extend(np.mean(mfccs, axis=1))
    # Chroma (12)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    feats.extend(np.mean(chroma, axis=1))
    # RMS Energy
    feats.append(np.mean(librosa.feature.rms(y=audio)))
    # Zero Crossing Rate
    feats.append(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
    # Spectral Centroid
    feats.append(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    return np.array(feats)

def load_ravdess():
    X, y = [], []
    for actor in tqdm(os.listdir(RAVDESS_DIR), desc="RAVDESS"):
        dirp = os.path.join(RAVDESS_DIR, actor)
        if not os.path.isdir(dirp): continue
        for fn in os.listdir(dirp):
            if not fn.endswith('.wav'): continue
            code = fn.split('-')[2]
            emo  = EMOTION_MAPS['ravdess'].get(code)
            feat = extract_features(os.path.join(dirp, fn))
            if feat is not None:
                X.append(feat); y.append(emo)
    return np.array(X), np.array(y)

def load_savee():
    X, y = [], []
    for fn in tqdm(os.listdir(SAVEE_DIR), desc="SAVEE"):
        if not fn.endswith('.wav'): continue
        code = fn[3]
        emo  = EMOTION_MAPS['savee'].get(code)
        feat = extract_features(os.path.join(SAVEE_DIR, fn))
        if feat is not None:
            X.append(feat); y.append(emo)
    return np.array(X), np.array(y)

def load_tess():
    X, y = [], []
    for folder in tqdm(os.listdir(TESS_DIR), desc="TESS"):
        dirp = os.path.join(TESS_DIR, folder)
        if not os.path.isdir(dirp): continue
        label = folder.split('_')[-1].lower()
        emo   = EMOTION_MAPS['tess'].get(label)
        for fn in os.listdir(dirp):
            if not fn.endswith('.wav'): continue
            feat = extract_features(os.path.join(dirp, fn))
            if feat is not None:
                X.append(feat); y.append(emo)
    return np.array(X), np.array(y)