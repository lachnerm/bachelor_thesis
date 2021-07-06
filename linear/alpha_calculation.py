import argparse
import functools
import os
import sqlite3

import numpy as np
from sklearn.linear_model import RidgeCV

from linear.linear_attacks import get_training_data
from utils.utils import sort_raw_db_names


def main():
    """
    Computes the optimal alpha values for ridge regression for all datasets that can be found in the provided folder.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '--root_folder', required=True,
                        help="root folder directory where the folder for the datasets is stored")
    parser.add_argument('--folder', required=True,
                        help="folder name where the datasets for the attacks are stored")

    args = parser.parse_args()
    root_folder = args.root
    db_folder = args.folder

    _, _, db_names = next(os.walk(f"{root_folder}/{db_folder}"))
    db_names = list(map(lambda file: file.split(".")[0], db_names))
    db_names = sorted(db_names, key=functools.cmp_to_key(sort_raw_db_names))

    values = list(range(0, 2000))

    for db, value in zip(db_names, values):
        conn = sqlite3.connect(f"{root_folder}/{db_folder}/{db}.db")
        cursor = conn.cursor()

        do_crop = True
        img_size = 200 if db_folder == "real" else 512
        crop_size = 128
        c_bits = int(db.split('b')[0])

        c_train, r_train = get_training_data(cursor, img_size, do_crop, crop_size)
        c_train_quadratic = np.empty((c_train.shape[0], c_bits * (c_bits + 1) // 2))
        idx = 0
        for i in range(c_bits):
            for j in range(i + 1):
                c_train_quadratic[:, idx] = c_train[:, i] * c_train[:, j]
                idx += 1

        ridge_alphas = list(np.arange(value - 1, value + 1, 0.01))

        reg = RidgeCV(alphas=ridge_alphas).fit(c_train_quadratic, r_train)
        print(db, reg.alpha_)


if __name__ == "__main__":
    main()
