
import numpy as np
import os
import pandas as pd
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch
from PIL import Image


def load_data(public_test_dir, private_test_dir):
    X_crp = []
    labels = []
    for filename in os.listdir(public_test_dir):
        img = Image.open(public_test_dir + filename)
        tensor = F.to_tensor(img)
        tr = F.crop(tensor, 192, 192, 192, 192)
        tr = F.resize(tr, [64, 64])
        img = tr.numpy()
        X_crp.append(img.flatten())
        labels.append(filename.split('.')[0])


def make_csv(mode, dataloader, checkpoint_path, cfg):
    return 0


def main(cfg):
    public_test_dir = './tests/public_test/'
    private_test_dir = './tests/private_test/'
    X_crp = load_data(public_test_dir, private_test_dir)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./config.ini")
    main(cfg=config)
