import glob
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from consts import *
import cv2
import random
import math

def get_patch(img, coord):
    weight = math.floor(PATCH_SIZE / 2)
    return img[coord[0]-weight:coord[0]+weight+1, coord[1]-weight:coord[1]+weight+1]

class CustomDataset(Dataset):
    def __init__(self, train=True, transform=None):
        
        self.svg_patches = []
        self.model_patches = []
        self.labels = []

        if train is True:
            for i in range(1, 90+1):
                file_num = f'{i}'.zfill(3)
                file_name = f'{file_num}.png'

                svg_img = cv2.imread(TRAIN_SVG_DIR+'/'+file_name)
                model_img = cv2.imread(TRAIN_MODEL_RES_DIR+'/'+file_name)
                label_img = cv2.imread(TRAIN_LABEL_DIR+'/'+file_name)
                for j in range(1024):
                    x = random.randrange(50, svg_img.shape[0]-50)
                    y = random.randrange(50, svg_img.shape[1]-50)
                    self.svg_patches.append(get_patch(svg_img, (x, y)))
                    self.model_patches.append(get_patch(model_img, (x, y)))
                    self.labels.append(1 if label_img[x, y, 1] == 255 else 0)
        else:
            for i in range(90, 100+1):
                file_num = f'{i}'.zfill(3)
                file_name = f'{file_num}.png'

                svg_img = cv2.imread(TRAIN_SVG_DIR+'/'+file_name)
                model_img = cv2.imread(TRAIN_MODEL_RES_DIR+'/'+file_name)
                label_img = cv2.imread(TRAIN_LABEL_DIR+'/'+file_name)
                for j in range(20):
                    x = random.randrange(50, svg_img.shape[0]-50)
                    y = random.randrange(50, svg_img.shape[1]-50)
                    self.svg_patches.append(get_patch(svg_img, (x, y)))
                    self.model_patches.append(get_patch(model_img, (x, y)))
                    self.labels.append(1 if label_img[x, y, 1] == 255 else 0)

        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):

        label = self.labels[idx]
        svg_patch = self.svg_patches[idx]
        model_patch = self.model_patches[idx]
        # pixel_data = np.ravel(pixel_data, order='C').tolist()

        # svg_pixel_data = np.expand_dims(svg_patch, 0)
        svg_torch_pixel = torch.from_numpy(svg_patch)
        torch_float_svg_data = svg_torch_pixel.type(torch.FloatTensor)
        torch_float_svg_data = torch_float_svg_data / 255

        # model_pixel_data = np.expand_dims(model_patch, 0)
        model_torch_pixel = torch.from_numpy(model_patch)
        torch_float_model_data = model_torch_pixel.type(torch.FloatTensor)
        torch_float_model_data = torch_float_model_data / 255

        label = np.expand_dims(label, 0)
        torch_label = torch.from_numpy(label)
        torch_float_label_data = torch_label.type(torch.LongTensor)

        return torch_float_svg_data, torch_float_model_data, torch_float_label_data

if __name__ == "__main__":
    dataset = CustomDataset()