import json
import sqlite3

import numpy as np
from scipy.spatial.distance import hamming
from tqdm import tqdm

from utils import utils


def calc_fhd(db_name, threshold, db_folder, log_folder, root, img_size, crop_size, custom_gabor=False):
    """
    Computes the fractional hamming distance between all responses of the provided dataset up to a given threshold.

    :param db_name: name of the db file that contains the data
    :param threshold: threshold until which entry in the db the fhd will be computed
    :param db_folder: name of the folder that contains the db file
    :param log_folder: folder where the results of the attacks will be stored
    :param root: root folder directory where the folder for the datasets is stored
    :param img_size: size of the responses
    :param crop_size: size to which the responses will be cropped if decided to
    :param custom_gabor: whether to use the second gabor transformation
    :return: list of all fhds that were computed
    """
    do_crop = db_folder != "real"
    db_path = f"{root}/{db_folder}/{db_name}.db"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        table_name = cursor.execute("select name from sqlite_master where type = 'table';").fetchone()[0]
        rows = cursor.execute(
            f"SELECT DISTINCT challenge, response FROM {table_name} WHERE id <= {threshold}").fetchall()
    except Exception as e:
        print("Error on establishing db connection to ", db_name)
        print(e)
        return []

    conn.close()

    r = np.array(json.loads(rows[0][1])).astype(np.float)
    r.reshape(img_size, img_size)
    get_img_response = lambda response: np.array(json.loads(response)).astype(np.float)

    transform = lambda r, crop: (
        utils.custom_gabor_transform(r, crop, img_size, crop_size) if custom_gabor
        else utils.gabor_bit_transform(r, crop, img_size, crop_size)).flatten()

    print(f"Calculating FHD for {db_name}")
    r_fhds = []
    for ref_challenge, ref_response in tqdm(rows):
        ref_response = get_img_response(ref_response)
        ref_response_bitstring = transform(ref_response, do_crop)

        for challenge, response in rows:
            c_fhd = hamming(ref_challenge, challenge)
            if c_fhd == 0:
                continue

            response = get_img_response(response)
            response_bitstring = transform(response, do_crop)

            r_fhd = hamming(ref_response_bitstring, response_bitstring)
            r_fhds.append(r_fhd)

    with open(f'{log_folder}/tmp/{db_name}_fhd.json', 'w') as tmp:
        results = {"FHD": r_fhds}
        json.dump(results, tmp)

    return r_fhds
