# encoding: utf-8
import os
import glob
import torch
import numpy as np
import scipy.io.wavfile as wav
from scipy.io.wavfile import read
from utils_mfcc import computer_feature
from torch.utils.data import Dataset
import pdb
class MyDataset_train(Dataset):
    def __init__(self):
        self.data = []
        tempdata =[]
        path = './5000'
        speech_wav_files = glob.glob(os.path.join(path, 'speech', '*.wav'))
        music_wav_files = glob.glob(os.path.join(path, 'music', '*.wav'))
        
        for speech in speech_wav_files:
            mfcc_features = computer_feature(speech)
            mfcc_features= np.transpose(mfcc_features)
            num=int(mfcc_features.shape[0]/20)
            for i in range(num):
              tempdata.append((mfcc_features[i*20:(i+1)*20], 0))
              self.data.append((mfcc_features[i*20:(i+1)*20], 0))
        
        print('the number of speech for training: {}'.format(len(self.data)))
        speech_data = len(self.data)

        ##music label: 1
        for music in music_wav_files:
            mfcc_features = computer_feature(music)
            mfcc_features= np.transpose(mfcc_features)
            num=int(mfcc_features.shape[0]/20)
            for i in range(num):
              self.data.append((mfcc_features[i*20:(i+1)*20], 1))
       
        print('the number of music for training: {}'.format(len(self.data)-speech_data)) 
    
    def __getitem__(self, idx):
        (m_feat, label) = self.data[idx]
        
        return {'inputs':torch.FloatTensor(m_feat), 'label':label}
            
    def __len__(self):
        return len(self.data)


class MyDataset_test(Dataset):
    def __init__(self):
        self.data = []
        path = './5000'
        speech_wav_files = glob.glob(os.path.join(path, 'speech-test', '*.wav'))
        music_wav_files = glob.glob(os.path.join(path, 'music-test', '*.wav'))
   
        ##speech label: 0
        for speech in speech_wav_files:
            mfcc_features = computer_feature(speech)
            mfcc_features=np.transpose(mfcc_features)
            num=int(mfcc_features.shape[0]/20)
            for i in range(num):
              self.data.append((mfcc_features[i*20:(i+1)*20], 0))
            
        print('the number of speech for testing: {}'.format(len(self.data)))
        speech_data = len(self.data)
        
        ##music label: 1
        for music in music_wav_files:
            mfcc_features = computer_feature(music)
            mfcc_features=np.transpose(mfcc_features)
            num=int(mfcc_features.shape[0]/20)
            for i in range(num):
              self.data.append((mfcc_features[i*20:(i+1)*20], 1))
            
        print('the number of music for testing: {}'.format(len(self.data)-speech_data)) 
    
    def __getitem__(self, idx):
        (m_feat, label) = self.data[idx]
  
        return {'inputs':torch.FloatTensor(m_feat), 'label':label}
            
    def __len__(self):
        return len(self.data)
