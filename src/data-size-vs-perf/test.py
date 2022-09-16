import glob

import numpy as np
import pandas as pd
from raise_utils.data import Data
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
