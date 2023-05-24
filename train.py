import data_loader
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
    dataset = CustomDataset()
    validation_set = CustomDataset(train=False)
    dataloader = DataLoader(dataset = dataset, batch_size = TRAIN_BATCH_SIZE, shuffle=True, drop_last=False)
    validation_loader = DataLoader(dataset = validation_set, batch_size = 1, shuffle=False, drop_last=False)
    print("Generate Dataset Done")

    model = MuxNetwork()
    print(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    
    for epoch in range(1, 100+1):
        cost = 0
        model.train()
        # random_idxs = []
        # for ridx in range(batch_size):
        #     random_idxs.append(random.randint(0, 389764))
        for batch_idx, samples in enumerate(dataloader):
          svg_patch, model_patch, label = samples
          label = label.squeeze(dim=-1)
          svg_patch = svg_patch.reshape(TRAIN_BATCH_SIZE,3,PATCH_SIZE,PATCH_SIZE)
          model_patch = model_patch.reshape(TRAIN_BATCH_SIZE,3,PATCH_SIZE,PATCH_SIZE)

          svg_patch = svg_patch.to(device)
          model_patch = model_patch.to(device)
          label = label.to(device)
          pred = model(svg_patch, model_patch).to(device)
          # print(pred.shape)
          loss = criterion(pred, label)
          loss.backward()
          optimizer.step()
          
          cost += loss

          with torch.no_grad():
            tot = 0
            cor = 0
            for svg_patch, model_patch, label in validation_loader:
              svg_patch = svg_patch.reshape(1,3,PATCH_SIZE,PATCH_SIZE)
              model_patch = model_patch.reshape(1,3,PATCH_SIZE,PATCH_SIZE)
              output = model(svg_patch, model_patch)
              _, pred = torch.max(output.data, 1)
              tot += label.size(0)
              cor += (pred == label).sum()
          
          avg_cost = cost/len(dataloader)
          acc = 100*cor / tot

          print(f'train {n}, epoch {epoch}, batch {batch_idx}/{len(dataloader)}, loss: {loss}, acc : {acc}')
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
