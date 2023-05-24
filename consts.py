import torch

TRAIN_BATCH_SIZE = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_LABEL_DIR = 'dataset/label'
TRAIN_MODEL_RES_DIR = 'dataset/model'
TRAIN_SVG_DIR = 'dataset/svg'
PATCH_SIZE = 100