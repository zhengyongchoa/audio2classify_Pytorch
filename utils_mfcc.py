import numpy as np
import scipy.io.wavfile as wav
from scipy.io.wavfile import read
from python_speech_features import mfcc, delta,base
import librosa
import pdb


def computer_feature(filname):
        y, sr = librosa.load(filname)
        mfcc_feat = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=13)
        
        # Deltas
        d_mfcc_feat = delta(mfcc_feat, 2)
        # Deltas-Deltas
        dd_mfcc_feat = delta(d_mfcc_feat, 2)
        # concat above three features
        concat_mfcc_feat = np.concatenate(
            (mfcc_feat, d_mfcc_feat, dd_mfcc_feat))

        return concat_mfcc_feat

