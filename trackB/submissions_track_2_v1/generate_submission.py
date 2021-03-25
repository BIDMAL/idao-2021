import configparser
import gc
import logging
import pathlib as path
import sys
from collections import defaultdict
from itertools import chain
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot as skplt
import torch
from more_itertools import bucket

from idao.data_module import IDAODataModule
from idao.model import ConvNetClassification, ConvNetRegression
from idao.utils import delong_roc_variance


dict_pred = defaultdict(list)


def make_csv(mode, dataloader, checkpoint_path, cfg):
    torch.multiprocessing.set_sharing_strategy("file_system")
    logging.info("Loading checkpoint")

    # загрузка моделей
    net_regression = torch.load('./idao/IDAO_nn_ENERJY_v_2_1.pt')

    net_regression.eval()

    net_classification = torch.load('./idao//IDAO_nn_NR_ER_v_2_1.pt')

    net_classification.eval()

    if mode == "classification":
        logging.info("Classification model loaded")
    else:
        logging.info("Regression model loaded")

    pred2class = [1.0, 10.0, 20.0, 3.0, 30.0, 6.0]
    # здесь заменил бейзлайн-предсказания на наши
    with torch.no_grad():
        for _, (img, name) in enumerate(iter(dataloader)):
            if mode == "classification":
                dict_pred["id"].append(name[0].split('.')[0])

                output = net_classification(img)
                _, predicted = torch.max(output.data, 1)
                dict_pred["classification_predictions"].append(predicted)
            else:
                output = net_regression(img)
                _, predicted = torch.max(output.data, 1)
                predicted = [pred2class[pred] for pred in predicted]
                dict_pred["regression_predictions"].append(predicted)


def main(cfg):
    PATH = path.Path(cfg["DATA"]["DatasetPath"])

    dataset_dm = IDAODataModule(
        data_dir=PATH, batch_size=64, cfg=cfg
    )

    dataset_dm.prepare_data()
    # dataset_dm.setup()
    dl = dataset_dm.test_dataloader()

    for mode in ["regression", "classification"]:
        if mode == "classification":
            model_path = cfg["REPORT"]["ClassificationCheckpoint"]
        else:
            model_path = cfg["REPORT"]["RegressionCheckpoint"]

        make_csv(mode, dl, model_path, cfg=cfg)

    data_frame = pd.DataFrame(dict_pred, columns=["id", "classification_predictions", "regression_predictions"])
    # здесь добавил замену 0 на 1 и 1 на 0 в столбце 'classification_predictions'
    data_frame['classification_predictions'] = data_frame['classification_predictions'].replace({0: 1, 1: 0})
    data_frame.to_csv('submission.csv', index=False, header=True)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./config.ini")
    main(cfg=config)
