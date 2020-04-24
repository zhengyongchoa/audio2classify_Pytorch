# encoding: utf-8
import os
import glob
import torch
import numpy as np
import scipy.io.wavfile as wav
from scipy.io.wavfile import read
from utils_mfcc import compute_mfcc
from torch.utils.data import Dataset

def padding(array):
    array = [array[_] for _ in range(array.shape[0])]
    size = array[0].shape
    for i in range(180000 - len(array)):
        array.append(np.zeros(size))
    return np.stack(array, axis=0)
    
class MyDataset_test(Dataset):
    def __init__(self):
        self.data = []
        path = '/home/momozyc/Documents/LMS_codes/speech-music-classify-Luo'
        test_files = glob.glob(os.path.join(path, '5000','music','*.wav'))

        for wav_file in test_files:
            fs, signal = wav.read(wav_file)
            if len(signal) < 180000:
            #     print(speech)
                signal = padding(signal)
                mfcc_features = compute_mfcc(signal, 16000)
                self.data.append((wav_file, mfcc_features, 0))
            else:
                signal_i = signal[0:180000]
                mfcc_features = compute_mfcc(signal_i, 16000)
                self.data.append((wav_file, mfcc_features, 0))
        print('the number of test_files for testing: {}'.format(len(self.data)))
        speech_data = len(self.data)
    
    def __getitem__(self, idx):
        (wav_file, mfcc_features, label) = self.data[idx]
        #print(mfcc_features.shape)
        return {'name':wav_file, 'inputs':torch.FloatTensor(mfcc_features), 'label':label}
            
    def __len__(self):
        return len(self.data)
