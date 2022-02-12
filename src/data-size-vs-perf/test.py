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
        pass
