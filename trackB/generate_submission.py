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


def load_models(clf2_path, clf6_path):
    clf2_xgb = XGBClassifier(
        n_estimators=500,
        colsample_bytree=1.0,
        gamma=1,
        max_depth=3,
        min_child_weight=1,
        subsample=1.0,
        eval_metric='auc',
        use_label_encoder=False,
        n_jobs=-1,
        random_state=125)
    clf2_xgb.load_model(clf2_path)
    clf6_xgb = XGBClassifier(
        objective='multi:softmax',
        num_classes=6,
        n_estimators=500,
        colsample_bytree=1.0,
        gamma=1,
        max_depth=3,
        min_child_weight=1,
        subsample=1.0,
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_jobs=-1,
        random_state=125)
    clf6_xgb.load_model(clf6_path)
    return clf2_xgb, clf6_xgb


def make_predictions(X_crp, cls2, cls6):
    preds2 = cls2.predict(X_crp)
    preds6 = cls6.predict(X_crp)
    return preds2, preds6


def make_csv(fnames, preds2, preds6):
    conv_six = {0: 1, 1: 3, 2: 6, 3: 10, 4: 20, 5: 30}
    df = pd.DataFrame(fnames, columns=['id'])
    df['classification_predictions'] = preds2
    df['regression_predictions'] = preds6
    df.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    public_test_dir = './tests/public_test/'
    private_test_dir = './tests/private_test/'
    clf2_path = './saved_models/xgb2'
    clf6_path = './saved_models/xgb6'
    X_crp, fnames = load_data(public_test_dir, private_test_dir)
    cls2, cls6 = load_models(clf2_path, clf6_path)
    preds2, preds6 = make_predictions(X_crp, cls2, cls6)
    make_csv(fnames, preds2, preds6)
