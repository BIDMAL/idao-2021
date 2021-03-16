import numpy as np
import os
import pandas as pd


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
    fnames = []
    for filename in os.listdir(public_test_dir):
        fnames.append(filename.split('.')[0])
    for filename in os.listdir(private_test_dir):
        fnames.append(filename.split('.')[0])

    preds2 = [0 for _ in range(len(fnames))]
    preds6 = [1 for _ in range(len(fnames))]
    make_csv(fnames, preds2, preds6)
