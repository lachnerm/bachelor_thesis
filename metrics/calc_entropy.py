import json
import sqlite3

import numpy as np
from skimage.measure import shannon_entropy

from utils.utils import crop_center


def calc_entropy(db_name, db_folder, log_folder, root, img_size, crop_size):
    """
    Computes the shannon entropies for all responses and their cropped versions of the provided dataset.

    :param db_name: name of the db file that contains the data
    :param db_folder: name of the folder that contains the db file
    :param log_folder: folder where the results of the attacks will be stored
    :param root: root folder directory where the folder for the datasets is stored
    :param img_size: size of the responses
    :param crop_size: size to which the responses will be cropped if decided to
    :return: tuple of the list of entropies for the regular and the cropped responses
    """
    db_path = f"{root}/{db_folder}/{db_name}.db"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        table_name = cursor.execute("select name from sqlite_master where type = 'table';").fetchone()[0]
        data = cursor.execute(f"SELECT DISTINCT response FROM {table_name}")
    except Exception:
        print("Error on establishing db connection to ", db_name)
        exit()

    print(f"Calculating Entropy for {db_name}")

    entropies = []
    crop_entropies = []
    for r in data:
        r = np.array(json.loads(r[0])).astype(np.float)
        r = r.reshape(img_size, img_size)
        entropy = shannon_entropy(r)
        entropies.append(entropy)

        r = crop_center(r, crop_size, img_size).flatten()
        crop_entropy = shannon_entropy(r)
        crop_entropies.append(crop_entropy)

    conn.close()

    with open(f'{log_folder}/tmp/{db_name}_entropy.json', 'w') as tmp:
        results = {"Entropy": entropies, "Crop Entropy": crop_entropies}
        json.dump(results, tmp)

    return entropies, crop_entropies
