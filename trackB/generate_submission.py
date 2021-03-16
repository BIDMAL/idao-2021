import numpy as np
import os
import pandas as pd
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from catboost import CatBoostClassifier


def load_data(public_test_dir, private_test_dir):
    X_crp = []
    fnames = []
    for filename in os.listdir(public_test_dir):
        img = Image.open(public_test_dir + filename)
        tensor = F.to_tensor(img)
        tr = F.crop(tensor, 192, 192, 192, 192)
        tr = F.resize(tr, [64, 64])
        img = (tr.numpy()*255).astype(np.uint8)
        X_crp.append(img.flatten())
        fnames.append(filename.split('.')[0])
    for filename in os.listdir(private_test_dir):
        img = Image.open(private_test_dir + filename)
        tensor = F.to_tensor(img)
        tr = F.crop(tensor, 192, 192, 192, 192)
        tr = F.resize(tr, [64, 64])
        img = (tr.numpy()*255).astype(np.uint8)
        X_crp.append(img.flatten())
        fnames.append(filename.split('.')[0])
    X_crp = np.array(X_crp)
    return X_crp, fnames


def load_models(clf2_path, clf6_path):
    clf2_cat = CatBoostClassifier()
    clf2_cat.load_model(clf2_path)
    clf6_cat = CatBoostClassifier()
    clf6_cat.load_model(clf6_path)
    return clf2_cat, clf6_cat


def make_predictions(X_crp, cls2, cls6):
    preds2 = cls2.predict(X_crp)
    preds6 = cls6.predict(X_crp)
    return preds2, preds6


def make_csv(fnames, preds2, preds6):
    preds6 = np.stack(preds6, axis=1)[0]
    conv_six = {0: 1, 1: 3, 2: 6, 3: 10, 4: 20, 5: 30}
    df = pd.DataFrame(fnames, columns=['id'])
    df['classification_predictions'] = preds2
    df['regression_predictions'] = list(map(lambda x: conv_six[x], preds6))
    df.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    public_test_dir = './tests/public_test/'
    private_test_dir = './tests/private_test/'
    clf2_path = './saved_models/cat2.cbm'
    clf6_path = './saved_models/cat6.cbm'
    X_crp, fnames = load_data(public_test_dir, private_test_dir)
    cls2, cls6 = load_models(clf2_path, clf6_path)
    preds2, preds6 = make_predictions(X_crp, cls2, cls6)
    make_csv(fnames, preds2, preds6)
