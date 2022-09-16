import glob

import numpy as np
import pandas as pd
from raise_utils.data import Data
from raise_utils.transforms import Transform
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dodge import DODGELearner

base_path = '../../../DODGE Data/UCI/'
files = glob.glob(base_path + '*.csv')

win = {}
loss = {}

for file in files:
    print(file)

    df = pd.read_csv(file)
    print('Total size =', len(df))

    # Min of 90% of the data and data size rounded to the nearest 10
    for i in tqdm(range(50, min(int(0.9 * len(df)), round(len(df) / 10) * 10) + 1, 50)):
        X = df[df.columns[:-1]]
        y = df[df.columns[-1]]

        try:
            data = Data(*train_test_split(X, y, stratify=y, train_size=i))
        except ValueError:
            break
        data.x_train = np.array(data.x_train).astype(np.float32)
        data.y_train = np.array(data.y_train).astype(np.float32)

        dodge = DODGELearner(['f1'])
        dodge.set_data(*data)
        ghost_result = dodge.fit()

        dodge = DODGELearner(['f1'])
        transform = Transform('wfo')
        transform.apply(data)
        transform.apply(data)
        transform = Transform('smote')
        transform.apply(data)
        dodge.set_data(*data)
        dodge_result = dodge.fit()

        if ghost_result > dodge_result:
            if i not in win:
                win[i] = 1
            else:
                win[i] += 1
        else:
            if i not in loss:
                loss[i] = 1
            else:
                loss[i] += 1

print('Wins:', win)
print('Loss:', loss)
