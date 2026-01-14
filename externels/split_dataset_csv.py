import numpy as np
import pandas as pd
import os

os.umask(0o000)  # Note the octal notation with leading 0o
np.random.seed(3407)

ratio = 0.15    # validation set ratio
CSV_PATH = 'train.csv'


df_csv = pd.read_csv(CSV_PATH)
csv_len = len(df_csv)
np_indices = np.array(range(0, csv_len))
np.random.shuffle(np_indices)
pick_len = int(csv_len * (1 - ratio))
train_idx, val_idx = np.split(np_indices, [pick_len])

df_train = df_csv.iloc[train_idx]
df_val = df_csv.iloc[val_idx]

df_train.to_csv('train.csv', index=False, encoding='utf-8-sig')
df_val.to_csv('valid.csv', index=False, encoding='utf-8-sig')
