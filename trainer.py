import train
import torch
from consts import *


def train_many():
    for idx in range(0, 3):
        filename = "/content/drive/MyDrive/EngineeringDesign/model/Result/"+str(idx)+".pt"
        net = train.train_model(idx)
        torch.save(net, filename)
        print("Model "+filename+" saved")


def train_only():
    print("Start Training")
    filename = "/content/drive/MyDrive/EngineeringDesign/model/Result/result_only.pt"
    net = train.train_model(1)
    torch.save(net, filename)
    print("Model " + filename + " saved")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device + " is available")
    train_only()