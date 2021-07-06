import argparse
import os
from concurrent.futures import ProcessPoolExecutor as Pool

from tqdm import tqdm

from metrics.calc_entropy import calc_entropy
from metrics.calc_fhd import calc_fhd
from visualization.create_metric_plots import create_metric_plots


def calc_metrics(db_name, threshold, db_folder, log_folder, root, custom_gabor):
    """
    Computes the entropy and FHD used for the dataset evaluation for the given dataset.

    :param db_name: name of the db file that contains the data
    :param threshold: threshold until which entry in the db the fhd will be computed
    :param db_folder: name of the folder that contains the db file
    :param log_folder: folder where the results of the attacks will be stored
    :param root: root folder directory where the folder for the datasets is stored
    :param custom_gabor: whether to use the second gabor transformation
    :return: tuple of the database name and all computed metrics
    """
    if db_folder == "real":
        img_size = 200
    elif db_folder == "real_large":
        img_size = 400
    else:
        img_size = 512
    crop_size = 128

    fhd = calc_fhd(db_name, threshold, db_folder, log_folder, root, img_size, crop_size, custom_gabor=custom_gabor)
    entropy_values = calc_entropy(db_name, db_folder, log_folder, root, img_size, crop_size)
    entropy, crop_entropy = entropy_values
    return db_name, {"Entropy": entropy,
                     "Crop Entropy": crop_entropy,
                     "FHD": fhd}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, default=1,
                        help="number of processes that will be run for the attacks")
    parser.add_argument('--threshold', type=int, default=300,
                        help="number of CRPs that will be used for the FHD computation")
    parser.add_argument('--root', '--root_folder', required=True,
                        help="root folder directory where the folder for the datasets is stored")
    parser.add_argument('--folder', required=True,
                        help="folder name where the datasets for the attacks are stored")
    parser.add_argument('--g', '--custom-gabor', action="store_true",
                        help="use the second gabor transformation for the FHD evaluation")

    args = parser.parse_args()
    root = args.root
    db_folder = args.folder
    processes = args.p
    threshold = args.threshold

    do_rename = db_folder not in ["real", "real_large"]
    _, _, db_names = next(os.walk(f"{root}/{db_folder}"))
    db_names = list(map(lambda file: file.split(".")[0], db_names))

    log_folder = f"results/metric_results_{db_folder}" if not args.g else f"results/metric_results_{db_folder}_custom_gabor"
    necessary_dirs = [log_folder, f"{log_folder}/tmp"]

    for dir in necessary_dirs:
        if not os.path.isdir(dir):
            os.mkdir(dir)

    pool = Pool(max_workers=processes)
    with tqdm(total=len(db_names)) as progress:
        futures = []

        for db in db_names:
            future = pool.submit(calc_metrics, db, threshold, db_folder, log_folder, root, args.g)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        results = [future.result() for future in futures]

    create_metric_plots(results, log_folder, db_folder, do_rename)


if __name__ == "__main__":
    main()
