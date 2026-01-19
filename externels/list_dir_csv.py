import os

import pandas as pd

# should be absolute path
DATASET_PATH = "/path/to/train/input"

os.umask(0o000)  # Note the octal notation with leading 0o

fns = [os.path.join(dir_path, f) for (dir_path, dir_names, fn) in os.walk(DATASET_PATH) for f in fn]
fns = sorted(fns)
pd.DataFrame({"input": fns}).to_csv("train.csv", encoding="utf-8-sig", index=False)
