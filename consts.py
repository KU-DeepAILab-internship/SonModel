import torch

TRAIN_BATCH_SIZE = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_LABEL_DIR = '/content/drive/MyDrive/EngineeringDesign/model/dataset/label'
TRAIN_MODEL_RES_DIR = '/content/drive/MyDrive/EngineeringDesign/model/dataset/model'
TRAIN_SVG_DIR = '/content/drive/MyDrive/EngineeringDesign/model/dataset/svg'
PATCH_SIZE = 35