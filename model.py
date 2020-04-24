#coding:utf-8
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from python_speech_features import mfcc, delta
import options as opt


class Speech_Music_Classify(nn.Module):
    def __init__(self, inputDim=195,nClasses=2):
        super(Speech_Music_Classify, self).__init__()
        self.nClasses = nClasses
        self.conv1 = nn.Conv1d(in_channels=39,out_channels=1024,kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=1024,out_channels=512,kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=512,out_channels=256,kernel_size=2)
        self.conv4 = nn.Conv1d(in_channels=256,out_channels=128,kernel_size=2)
        self.conv5 = nn.Conv1d(in_channels=128,out_channels=64,kernel_size=2)
        #卷积的输出尺寸
        self.fc6 = nn.Sequential(nn.Linear(64*15, 128*2), nn.BatchNorm1d(128*2), nn.ReLU(True))
        self.fc7 = nn.Sequential(nn.Linear(128*2,128), nn.BatchNorm1d(128), nn.ReLU(True))
        self.fc8 = nn.Sequential(nn.Linear(128, self.nClasses))
        self._initialize_weights()
        
    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        #print(x.size())
        x = x.view(opt.batch_size, -1)
        x = self.fc6(x)
        x = self.fc7(x)
        #print(x.size())
        x = self.fc8(x)
        
        
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def speech_music_classify(inputDim=390, nClasses=2):
    model = Speech_Music_Classify(inputDim=inputDim,  nClasses=nClasses)
    return model



