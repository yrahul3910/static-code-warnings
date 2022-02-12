import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from raise_utils.data import Data
from ghost import BinaryGHOST
from dodge import DODGELearner


base_path = '../../../DODGE Data/defect/'
files = glob.glob(base_path + '*-*.*.csv')

win = {}
loss = {}

for file in files:
    print(file)

    df = pd.read_csv(file)
    df.drop(df.columns[:3], axis=1, inplace=True)

    # Min of 90% of the data and data size rounded to the nearest 10
    for i in tqdm(range(10, min(int(0.9 * len(df)), round(len(df) / 10) * 10) + 1, 10)):
        X = df.drop('bug', axis=1)
        y = df['bug']

        try:
            data = Data(*train_test_split(X, y, stratify=y, train_size=i))
        except ValueError:
            break
        data.x_train = np.array(data.x_train).astype(np.float32)
        data.y_train = np.array(data.y_train).astype(np.float32)

        ghost = BinaryGHOST(['f1'], name='experiment')
        ghost.set_data(*data)
        ghost_result = ghost.fit()

        dodge = DODGELearner(['f1'])
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
