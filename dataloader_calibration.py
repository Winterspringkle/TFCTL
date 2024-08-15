# -*- coding:utf-8 -*-
import numpy as np
from torch.utils import data
import os
import random
import python_speech_features as psf
def normalize(data_set):
    # this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the training set statistics.

    mean = np.mean(data_set, axis=(0, 1))
    max = data_set.max(axis=(0, 1))
    min = data_set.min(axis=(0, 1))
    data_set = (data_set - mean) / (max - min)
    return data_set
def get_LogFillterBank(x, sample_rate):
    """
    将信号变为log fillterbank特征
    :param x: signal
    :param sample_rate: 默认16000
    :return: 特征值
    """

    fb = psf.logfbank(x, nfilt=24,samplerate=sample_rate)
    fb = normalize(fb)
    fillterBank = np.array(fb)
    return fillterBank

def get_MFCC(x, sample_rate):
    """
    将信号变为log fillterbank特征
    :param x: signal
    :param sample_rate: 默认16000
    :return: 特征值
    """

    mfcc_feature = psf.mfcc(x, samplerate=sample_rate,numcep=20,nfilt=26)
    mfcc_list = np.array(mfcc_feature)
    return mfcc_list
def get_spectrogram(x, sample_rate):
    melspec = librosa.feature.melspectrogram(x, n_mels=80, sr=sample_rate).astype(np.float32).T
    melspec = np.log(np.maximum(1e-6, melspec))
    return melspec

class wav_pos_dataloader(data.Dataset):
    def __init__(self, wav_dir1,segment_length):
        self.wav_dir1 = wav_dir1
        self.data_list=[]
        self.segment_length=segment_length
        self.total_list=os.listdir(self.wav_dir1)
        for sp in os.listdir(self.wav_dir1):
            wav_folder=os.path.join(self.wav_dir1,sp)
            if "_1" in wav_folder:
                for wav in os.listdir(wav_folder):
                    label=int(sp[-1])
                    self.data_list.append((label,sp,os.path.join(wav_folder,wav)))
        self.num_wavs=len(self.data_list)

    def __getitem__(self, index):
        dataset = self.data_list
        label, sp, wav_filename = dataset[index]
        feature = np.load(wav_filename, allow_pickle=True)
        if "CMDC" in wav_filename:
            source = 1
        else:
            source = 0
        return feature,label,sp,source

    def __len__(self):
        """Return the number of images."""
        return self.num_wavs

class wav_neg_dataloader(data.Dataset):
    def __init__(self, wav_dir1,segment_length):
        self.wav_dir1 = wav_dir1
        self.data_list=[]
        self.segment_length=segment_length
        self.total_list=os.listdir(self.wav_dir1)
        for sp in os.listdir(self.wav_dir1):
            wav_folder=os.path.join(self.wav_dir1,sp)
            if "_0" in wav_folder:
                for wav in os.listdir(wav_folder):
                    label=int(sp[-1])
                    self.data_list.append((label,sp,os.path.join(wav_folder,wav)))
        self.num_wavs=len(self.data_list)

    def __getitem__(self, index):
        dataset = self.data_list
        label, sp, wav_filename = dataset[index]
        feature = np.load(wav_filename, allow_pickle=True)
        # feature=feature[np.newaxis, :]
        if "CMDC" in wav_filename:
            source = 1
        else:
            source = 0
        return feature,label,sp,source

    def __len__(self):
        """Return the number of images."""
        return self.num_wavs

class wav_dataloader(data.Dataset):
    def __init__(self, wav_dir1,segment_length):
        self.wav_dir1 = wav_dir1
        self.data_list=[]
        self.positive_list = []
        self.negative_list = []
        self.segment_length=segment_length
        self.total_list=os.listdir(self.wav_dir1)
        for sp in os.listdir(self.wav_dir1):
            wav_folder=os.path.join(self.wav_dir1,sp)
            for wav in os.listdir(wav_folder):
                label=int(sp[-1])
                self.data_list.append((label,sp,os.path.join(wav_folder,wav)))
                if label==0:
                    self.negative_list.append((label, sp, os.path.join(wav_folder, wav)))
                else:
                    self.positive_list.append((label, sp, os.path.join(wav_folder, wav)))
        self.num_wavs=len(self.data_list)

    def __getitem__(self, index):
        dataset = self.data_list
        label, sp, wav_filename = dataset[index]
        feature = np.load(wav_filename, allow_pickle=True)
        # feature=feature[np.newaxis, :]
        if label==1:
            positive_sample = random.sample(self.positive_list, 1)
            negative_sample = random.sample(self.negative_list, 1)
            positive_npy = np.load(positive_sample[0][2], allow_pickle=True)
            negative_npy = np.load(negative_sample[0][2], allow_pickle=True)
        else:
            positive_sample = random.sample(self.negative_list, 1)
            negative_sample = random.sample(self.positive_list, 1)
            positive_npy = np.load(positive_sample[0][2], allow_pickle=True)
            negative_npy = np.load(negative_sample[0][2], allow_pickle=True)
        if "CMDC" in wav_filename:
            source = 1
        else:
            source = 0
        return feature,label,sp,source,positive_npy,negative_npy

    def __len__(self):
        """Return the number of images."""
        return self.num_wavs