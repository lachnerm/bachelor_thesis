import argparse
import functools
import os
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.utils import sort_raw_db_names


def beautify_db_names(dbs):
    """
    Beautifies the names of the databases for plotting.

    :param dbs: databases to be renamed
    :return: list of beautified database names
    """
    renamed_dbs = []
    for db_name in dbs:
        blocks = int(db_name.split("b")[0])
        blocks_per_row = int(np.sqrt(blocks))
        mr = int(db_name.split("mr")[1].split("_")[0])
        mb = int(db_name.split("mb")[1].split("_")[0]) if len(db_name.split("mb")) > 1 else 0
        has_mb = mb > 0
        every_2nd = "2nd" in db_name

        if every_2nd:
            type = "B"
        elif has_mb and mb == blocks // 2:
            type = "C"
        elif has_mb:
            type = "D"
        else:
            type = "A"

        name = f"{blocks_per_row} x {blocks_per_row} - {f'(Type {type})'}"
        renamed_dbs.append(name)

    return renamed_dbs


def main():
    """
    Creates plots of the distributions of the number of bits that are activated within the challenges of the
    datasets that are found in the provided folder.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, default=1,
                        help="number of processes that will be run for the attacks")
    parser.add_argument('--root', '--root_folder', required=True,
                        help="root folder directory where the folder for the datasets is stored")
    parser.add_argument('--folder', required=True,
                        help="folder name where the datasets for the attacks are stored")

    args = parser.parse_args()
    db_folder = args.folder
    root = args.root

    _, _, db_names = next(os.walk(f"{root}/{db_folder}"))
    db_names = list(map(lambda file: file.split(".")[0], db_names))

    log_folder = f"../results/challenge_distribution_{db_folder}"

    Path(f"{log_folder}").mkdir(parents=True, exist_ok=True)

    db_names = sorted(db_names, key=functools.cmp_to_key(sort_raw_db_names))
    renamed_db_names = beautify_db_names(db_names)

    params = {
        'axes.labelpad': 15,
    }
    plt.rcParams.update(params)

    zipped_names = list(zip(renamed_db_names, db_names))
    for idx in range(0, len(db_names), 4):
        names = zipped_names[idx:idx + 4]
        fig, axs = plt.subplots(2, 2, figsize=(11, 8))
        for ax in axs.flatten():
            ax.set_xlabel("Number of activated bits")
            ax.set_ylabel("Number of challenges")
        for idx2, (title, db) in enumerate(names):
            conn = sqlite3.connect(f"{root}/data/{db_folder}/{db}.db")
            cursor = conn.cursor()
            table_name = cursor.execute("select name from sqlite_master where type = 'table';").fetchone()[0]
            challenges = cursor.execute(f"SELECT DISTINCT challenge FROM {table_name}").fetchall()
            c_bits = list(map(lambda c: np.sum([int(bit) for bit in c[0]]), challenges))

            if idx2 == 0:
                color = "mediumslateblue"
                ax = axs[0][0]
            elif idx2 == 1:
                color = "teal"
                ax = axs[0][1]
            elif idx2 == 2:
                color = "dodgerblue"
                ax = axs[1][0]
            else:
                color = "mediumblue"
                ax = axs[1][1]

            ax.set_title(f"{title}")
            ax.hist(c_bits, bins=len(set(c_bits)), color=color)
            fig.tight_layout()
            fig.savefig(f"{log_folder}/{db.split('_')[0]}_challenge_dist.png", bbox_inches="tight", pad_inches=0.2)
            plt.close("all")


if __name__ == "__main__":
    main()
