from matplotlib import pyplot as plt
import warnings
from PIL import Image
import numpy as np
import os
import pandas as pd
from skimage.io import imread, imsave
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch

torch.manual_seed(42)
warnings.filterwarnings('ignore')

er_trdat_path = './data/train/ER/'
nr_trdat_path = './data/train/NR/'
cropdat_path = './data/train/all_cropped/'
balanced_path = './data/train/all_balanced/'

if not os.path.exists(cropdat_path):
    os.mkdir(cropdat_path)
if not os.path.exists(balanced_path):
    os.mkdir(balanced_path)

transforms = torch.nn.Sequential(
    T.RandomAffine(degrees=45),
    T.RandomHorizontalFlip(p=0.5),
    T.GaussianBlur(3, sigma=(0.1, 2.0)),
)

counts = dict()
ii = 0
for filename in os.listdir(er_trdat_path):
    img = Image.open(er_trdat_path + filename)
    fn_parts = filename.split('_')
    ind = fn_parts.index('ER')
    regr = fn_parts[ind+1]
    tensor = F.to_tensor(img)
    n = 0
    if regr == '1':
        n = 700
    if regr == '6':
        n = 2000
    if regr == '20':
        n = 1000
    counts[f'ER-{regr}'] = counts.get(f'ER-{regr}', 0) + 1
    for i in range(n):
        tr = transforms(tensor)
        tr = F.crop(tr, 192, 192, 192, 192)
        tr = F.resize(tr, [64, 64])
        img = F.to_pil_image(tr)
        img.save(balanced_path + f'{ii}-ER-{regr}.png')
        ii += 1

print(counts)
