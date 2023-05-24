import data_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
import time
import math
import model
from model import CustomNetwork
from consts import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from CustomDataset import CustomDataset
from torch.utils.data import Dataset, DataLoader


class History:
    def __init__(self, data, label, selected, overall, ans):
        self.data = data
        self.label = label
        self.selected = selected
        self.overall = overall
        self.ans = ans

models = []
history_list = []

dataset = CustomDataset()
dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle=True, drop_last=False)


for i in range(0, target_count):
    models.append(torch.load('/content/drive/MyDrive/AlphabetDetection/result_{}.pt'.format(i), map_location=torch.device('cpu')))
    models[i].eval()

correct = 0
test_size = 100

for i in range(0, test_size):
    print('test case : {}'.format(i))
    randidx = random.randint(0, 119928-1)
    raw_data, label = dataset.__getitem__(randidx)
    # print(raw_data.shape)
    data = raw_data.view(-1, 1, 32, 32)
    # print(data)
    result = torch.zeros([1, 26], dtype=torch.float64)

    selected_list = []
    for i in range(0, target_count):
        output = models[i](data)
        selected = output.argmax(dim=1, keepdim=True).numpy()
        selected = selected.squeeze()
        print('model {} : {}'.format(i, selected))
        result += output.squeeze()
        selected_list.append(selected)
    
    overall = result.argmax(dim=1, keepdim=True).numpy().squeeze()
    print(result)
    ans = label.numpy().squeeze()
    print('Overall result : {}'.format(overall))
    print('Answer : {}'.format(ans))

    if overall == ans:
        correct = correct + 1

    his = History(raw_data,label, selected_list,overall, ans)
    history_list.append(his)
    print()

# plt.close()
# fig = plt.figure(figsize=(13,13)) # Notice the equal aspect ratio
# ax = [plt.subplot(10,10,i+1) for i in range(10*10)]
# for i in range(0, len(history)):
#     ax[i].imshow(history[i].data.view(32,32))

# fig.show()

print('{}/{}, {}%'.format(correct, test_size, correct/test_size*100))


fig = plt.figure(figsize=(13,13))
for i in range(0, len(history_list)):
    a = fig.add_subplot(10,15,i+1)
    a.axis('off')
    a.imshow(history_list[i].data.view(32,32))
plt.subplots_adjust(bottom=0.2, top=1.2, hspace=0)


