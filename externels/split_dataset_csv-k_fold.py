import numpy as np
import pandas as pd

np.random.seed(3407)

K_FOLDS = 5
CSV_PATH = "train.csv"


def main():
    df = pd.read_csv(CSV_PATH)
    df = df.sample(frac=1)

    lefts = len(df) % K_FOLDS

    if lefts != 0:
        df_folds = np.split(df[:-lefts], K_FOLDS)
        df_folds[-1] = pd.concat([df_folds[-1], (df[-lefts:])])
    else:
        df_folds = np.split(df, K_FOLDS)

    for i in range(K_FOLDS):
        val_input = df_folds[i]
        train_input = [df_folds[idx] for idx in range(K_FOLDS) if idx != i]
        train_input_unchained = pd.DataFrame()
        for item in train_input:
            train_input_unchained = pd.concat([train_input_unchained, item])

        train_input_unchained.to_csv(f"train-fold_{i}.csv", encoding="utf-8-sig", index=False)
        val_input.to_csv(f"valid-fold_{i}.csv", encoding="utf-8-sig", index=False)


if __name__ == "__main__":
    main()
