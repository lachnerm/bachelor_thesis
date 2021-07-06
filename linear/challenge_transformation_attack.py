import argparse
import functools
import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np

from linear_attacks import get_training_data, get_test_data, run_lr
from utils import utils


def main():
    """
    Runs the challenge transformation into a higher number of bits and uses the new challenges for a linear regression
    attack. Afterwards, the differences for the transformed and non-transformed data with respect to the FHD, PC and
    SSIM are plotted.
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
    db_names = sorted(db_names, key=functools.cmp_to_key(utils.sort_raw_db_names))

    mean_fhds = []
    mean_ssims = []
    mean_pcs = []
    for db in db_names:
        c_bits = int(db.split("b")[0])
        conn = sqlite3.connect(f"{root_folder}/{db_folder}/{db}.db")
        cursor = conn.cursor()

        do_crop = True
        img_size = 512
        crop_size = 128
        c_train, r_train = get_training_data(cursor, img_size, crop_size, do_crop)
        c_test, r_test = get_test_data(cursor, img_size, crop_size, do_crop)

        shape = int(np.sqrt(c_bits))

        for idx in range(0, 2):
            if idx > 0:
                c_train = np.array(
                    list(map(lambda c: c.reshape(shape, shape).repeat(2, 0).repeat(2, 1).flatten(), c_train)))
                c_test = np.array(
                    list(map(lambda c: c.reshape(shape, shape).repeat(2, 0).repeat(2, 1).flatten(), c_test)))
                shape *= 2

            args = [c_train, r_train, c_test, r_test, img_size, crop_size, do_crop, False]
            abs_diffs, pcs, ssims, fhds = run_lr(*args)

            fhd_mean = np.mean(fhds)
            pc_mean = np.mean(pcs)
            ssim_mean = np.mean(ssims)
            color = "blue" if idx == 0 else "red"

            mean_fhds.append({db: (fhd_mean, color)})
            mean_pcs.append({db: (pc_mean, color)})
            mean_ssims.append({db: (ssim_mean, color)})

    fig_fhd = plt.figure(111, figsize=(11, 10))
    ax_fhd = fig_fhd.gca()
    for entry in mean_fhds:
        db, (fhd_mean, color) = list(entry.items())[0]
        ax_fhd.scatter(db, fhd_mean, s=100, color=color)
        fig_fhd.savefig(fname="challenge_transformation_result_fhd.jpg", bbox_inches='tight', pad_inches=0)
    plt.close("all")

    fig_pc = plt.figure(111, figsize=(11, 10))
    ax_pc = fig_pc.gca()
    for entry in mean_pcs:
        db, (pc_mean, color) = list(entry.items())[0]
        ax_pc.scatter(db, pc_mean, s=100, color=color)
        fig_pc.savefig(fname="challenge_transformation_result_pc.jpg", bbox_inches='tight', pad_inches=0)
    plt.close("all")

    fig_ssim = plt.figure(111, figsize=(11, 10))
    ax_ssim = fig_ssim.gca()
    for entry in mean_ssims:
        db, (ssim_mean, color) = list(entry.items())[0]
        ax_ssim.scatter(db, ssim_mean, s=100, color=color)
        fig_ssim.savefig(fname="challenge_transformation_result_ssim.jpg", bbox_inches='tight', pad_inches=0)
    plt.close("all")


if __name__ == "__main__":
    main()
