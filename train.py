import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
import time
import math
import ModelDiscription
from ModelDiscription import MuxNetwork
from consts import *
from consts import TRAIN_BATCH_SIZE
from CustomDataset import CustomDataset
from torch.utils.data import Dataset, DataLoader

# raw_data = data_loader.load_train_data()


def train_model(n):
    print("Device : "+device)
    print("Generate Dataset")
    dataset = CustomDataset(train=True)
    validation_set = CustomDataset(train=False)
    dataloader = DataLoader(dataset = dataset, batch_size = TRAIN_BATCH_SIZE, shuffle=True, drop_last=False)
    validation_loader = DataLoader(dataset = validation_set, batch_size = 2*TRAIN_BATCH_SIZE, shuffle=False, drop_last=False)
    print("Generate Dataset Done")

    model = MuxNetwork()
    print(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    
    for epoch in range(1, 100+1):
        cost = 0
        model.train()
        # random_idxs = []
        # for ridx in range(batch_size):
        #     random_idxs.append(random.randint(0, 389764))
        for batch_idx, samples in enumerate(dataloader):
          optimizer.zero_grad()
          svg_patch, model_patch, label = samples
          

          svg_patch = svg_patch.to(device)
          model_patch = model_patch.to(device)
          label = label.to(device)
          # print(svg_patch.is_cuda)
          # print(model_patch.is_cuda)
          # print(label.is_cuda)
          
          pred = model(svg_patch, model_patch).to(device)
          # print(pred.shape)
          loss = criterion(pred, label)
          loss.backward()
          optimizer.step()
          if batch_idx %10 == 0:
            print(f'train {n}, epoch {epoch}, batch {batch_idx}/{len(dataloader)}, loss: {loss}')
          
          cost += loss

        model.eval()
        with torch.no_grad():
          valid_loss = sum(criterion(model(svg_patch.cuda(), model_patch.cuda()), label.cuda()) for svg_patch, model_patch, label in validation_loader)
        
        print(f'train {n}, epoch {epoch}, valid_loss : {valid_loss / len(validation_loader)}')
        # current_time = time.time() - start_time
        # x, y = so_compli.get_train_image_tensor(raw_data, idx)
        # x = x.to(device)
        # y = y.to(device)
        # prob = net(x).to(device)
        # loss = criterion(prob, y)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # study_rate = idx*batch_size/(current_time+0.00000001)
        # print(f'train {n}, {idx * batch_size},  time:{float(current_time):.5f},  loss: {loss}, rate: {float(study_rate):.5f}')
    return model
