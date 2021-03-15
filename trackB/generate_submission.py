import numpy as np
import os
import pandas as pd
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from xgboost import XGBClassifier


def load_data(public_test_dir, private_test_dir):
    X_crp = []
    fnames = []
    for filename in os.listdir(public_test_dir):
        img = Image.open(public_test_dir + filename)
        tensor = F.to_tensor(img)
        tr = F.crop(tensor, 192, 192, 192, 192)
        tr = F.resize(tr, [64, 64])
        img = tr.numpy()
        X_crp.append(img.flatten())
        fnames.append(filename.split('.')[0])
    for filename in os.listdir(private_test_dir):
        img = Image.open(private_test_dir + filename)
        tensor = F.to_tensor(img)
        tr = F.crop(tensor, 192, 192, 192, 192)
        tr = F.resize(tr, [64, 64])
        img = tr.numpy()
        X_crp.append(img.flatten())
        fnames.append(filename.split('.')[0])
    return X_crp, fnames


def load_models():
    pass


def make_csv(mode, dataloader, checkpoint_path, cfg):
    return 0


if __name__ == "__main__":
    public_test_dir = './tests/public_test/'
    private_test_dir = './tests/private_test/'
    X_crp, fnames = load_data(public_test_dir, private_test_dir)
    cls2, cls6 = load_models()
