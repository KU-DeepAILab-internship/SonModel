import os
import csv
import numpy as np
import pickle
import torch
from CustomDataset import CustomDataset
from torch.utils.data import Dataset, DataLoader
from consts import *

dataset = CustomDataset()
dataloader = DataLoader(dataset = dataset, batch_size = TRAIN_BATCH_SIZE, shuffle=True, drop_last=False)

def load_data(prefix='train'):
    np_data = np.load('/content/drive/MyDrive/AlphabetDetection/character_font.npz')
    return np_data

def load_train_data():
    return load_data(prefix='train')


def load_test_data():
    return load_data(prefix='test')


def get_train_image_tensor(np_data, idx):
    pixel_data = np_data['images'][idx]
    # pixel_data = np.ravel(pixel_data, order='C').tolist()

    label = np_data['labels'][idx]

    pixel_data = np.expand_dims(pixel_data, 0)
    torch_pixel = torch.from_numpy(pixel_data)
    torch_float_pixel_data = torch_pixel.type(torch.FloatTensor)
    torch_float_pixel_data = torch_float_pixel_data / 255

    label = np.expand_dims(label, 0)
    torch_label = torch.from_numpy(label)
    torch_float_label_data = torch_label.type(torch.LongTensor)

    return torch_float_pixel_data, torch_float_label_data


def get_train_image_tensors(np_data, idxs):
    pixel_datas = []
    labels = []
    for idx in idxs:
        pixel_datas.append(np.ravel(np_data['images'][idx], order='C').tolist())
        labels.append(np_data['labels'][idx])
        

    pixel_data = np.array(pixel_datas)
    torch_pixel = torch.from_numpy(pixel_data)
    torch_float_pixel_data = torch_pixel.type(torch.FloatTensor)
    torch_float_pixel_data = torch_float_pixel_data / 255

    label = np.array(labels)
    torch_label = torch.from_numpy(label)
    torch_float_label_data = torch_label.type(torch.LongTensor)

    return torch_float_pixel_data, torch_float_label_data


def get_test_image_tensor(test_raw_data, idx):
    pixel_data = test_raw_data[idx][0:]
    pixel_data = np.expand_dims(pixel_data, 0)
    torch_pixel = torch.from_numpy(pixel_data)
    torch_float_pixel_data = torch_pixel.type(torch.FloatTensor)
    torch_float_pixel_data = torch_float_pixel_data / 255

    return torch_float_pixel_data


if __name__ == '__main__':
    load_train_data()