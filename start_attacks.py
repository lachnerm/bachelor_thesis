import argparse
import os
from concurrent.futures import ProcessPoolExecutor as Pool
from pathlib import Path

from tqdm import tqdm

from linear.linear_attacks import run_linear
from non_linear.dl_attacks import run_dl
from visualization.create_attack_plots import create_attack_plots


def run_attacks(db_name, db_folder, args, root, log_folder):
    '''
    Initializes all attacks on the specified dataset.

    :param db_name: name of the db file that contains the data
    :param db_folder: name of the folder that contains the db file
    :param args: further arguments that specify the attack, details and restrictions (from argparse)
    :param root: root folder directory where the folder for the datasets is stored
    :param log_folder: folder where the results of the attacks will be stored
    :return: tuple of the db file name and the results of all attacks
    '''

    # real optical PUF data from the Italian research group has responses of size 200x200, the simulation of 512x512
    img_size = 200 if db_folder == "real" else 512
    crop_size = 128

    data = {} if args.odl else run_linear_attacks(db_name, img_size, crop_size, db_folder, args, root, log_folder)
    if not args.ol:
        dl_data = run_DL_attacks(db_name, img_size, crop_size, db_folder, args, root, log_folder)
        data.update(dl_data)

    return db_name, data


def run_DL_attacks(db_name, img_size, crop_size, db_folder, args, root, log_folder):
    '''
    Runs the specified DL attacks for the given dataset and returns the corresponding results.

    :param db_name: name of the db file that contains the data
    :param img_size: size of the responses
    :param crop_size: size to which the responses will be cropped if decided to
    :param db_folder: name of the folder that contains the db file
    :param args: further arguments that specify the attack, details and restrictions (from argparse)
    :param root: root folder directory where the folder for the datasets is stored
    :param log_folder: folder where the results of the attacks will be stored
    :return: results of the carried out DL attacks
    '''
    data = run_dl(args.oddl, args.oadl, args.ct2, args.c, db_name, db_folder, img_size, crop_size, root, log_folder,
                  args.g)
    return data


def run_linear_attacks(db_name, img_size, crop_size, db_folder, args, root, log_folder):
    '''
    Runs the specified linear attacks for the given dataset and returns the corresponding results.

    :param db_name: name of the db file that contains the data
    :param img_size: size of the responses
    :param crop_size: size to which the responses will be cropped if decided to
    :param db_folder: name of the folder that contains the db file
    :param args: further arguments that specify the attack, details and restrictions (from argparse)
    :param root: root folder directory where the folder for the datasets is stored
    :param log_folder: folder where the results of the attacks will be stored
    :return: results of all carried out linear attacks
    '''
    data = run_linear(db_name, img_size, crop_size, args.c, db_folder, root, log_folder, args.g, args.olr)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, default=1,
                        help="number of processes that will be run for the attacks")
    parser.add_argument('--root', '--root_folder', required=True,
                        help="root folder directory where the folder for the datasets is stored")
    parser.add_argument('--folder', required=True,
                        help="folder name where the datasets for the attacks are stored")
    parser.add_argument('--c', '--crop', action="store_true",
                        help="crop the responses and use the cropped generator type 1 for the DL attack")
    parser.add_argument('--ct2', '--crop-type-2', action="store_true",
                        help="use the cropped generator type 2 for the DL attack")
    parser.add_argument('--g', '--custom-gabor', action="store_true",
                        help="use the second gabor transformation for the FHD evaluation")
    parser.add_argument('--ol', '--only-linear', action="store_true",
                        help="run only the linear attacks")
    parser.add_argument('--olr', '--only-linear-regression', action="store_true",
                        help="run only the linear regression attack")
    parser.add_argument('--odl', '--only-deep-learning', action="store_true",
                        help="run only the deep learning attacks")
    parser.add_argument('--oddl', '--only-default-deep-learning', action="store_true",
                        help="run only the default generator DL attack")
    parser.add_argument('--oadl', '--only-advanced-deep-learning', action="store_true",
                        help="run only the advanced generator DL attack")

    args = parser.parse_args()
    root_folder = args.root
    db_folder = args.folder

    _, _, db_names = next(os.walk(f"{root_folder}/{db_folder}"))
    db_names = list(map(lambda file: file.split(".")[0], db_names))

    number_of_files = len(db_names)
    log_folder = f"results/attack_results_{db_folder}{'_custom_gabor' if args.g else ''}"

    Path(f"{log_folder}/tmp").mkdir(parents=True, exist_ok=True)
    Path(f"{log_folder}/plots/crop").mkdir(parents=True, exist_ok=True)
    Path(f"{log_folder}/plots/crop_type2").mkdir(parents=True, exist_ok=True)
    Path(f"{log_folder}/plots/normal").mkdir(parents=True, exist_ok=True)

    pool = Pool(max_workers=args.p)
    with tqdm(total=number_of_files) as progress:
        futures = []

        for db_name in db_names:
            future = pool.submit(run_attacks, db_name, db_folder, args, root=root_folder, log_folder=log_folder)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        results = [future.result() for future in futures]

    folder = f"{log_folder}/plots/{'crop' if args.c else 'normal'}"
    if args.ct2:
        folder += "_type2"

    create_attack_plots(results, folder, db_folder, args.c, args.ct2)


if __name__ == "__main__":
    main()
