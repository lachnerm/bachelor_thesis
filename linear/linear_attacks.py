import json
import sqlite3

import numpy as np
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as calc_ssim
from sklearn.linear_model import LinearRegression, Ridge
from tqdm import tqdm

from utils import utils
from utils.utils import crop_center


def prepare_data(rows, img_size, crop_size, do_crop):
    """
    Prepares the data retrieved from the db file to be used for the linear attacks.

    :param rows: array of CRPs from the db
    :param img_size: size of the responses
    :param crop_size: size to which the responses will be cropped if decided to
    :param do_crop: whether to crop the responses
    :return: tuple of prepared challenges and responses as numpy arrays
    """
    challenges, responses = [], []

    for row in tqdm(rows, leave=True, position=0):
        challenge = np.array([int(bit) for bit in row[0]]).astype(np.uint8)
        response = np.array(json.loads(row[1])).astype(np.float32)
        response = response.flatten()

        if do_crop:
            response = crop_center(response, crop_size, img_size).flatten()

        challenges.append(challenge)
        responses.append(response)
    return np.array(challenges), np.array(responses)


def get_training_data(cursor, img_size, crop_size, do_crop):
    """
    Retrieves and prepares the data that will be used for the training of the linear attacks.

    :param cursor: cursor object from sqlite to access the data
    :param img_size: size of the responses
    :param crop_size: size to which the responses will be cropped if decided to
    :param do_crop: whether to crop the responses
    :return: prepared data as tuple of challenges and responses
    """
    table_name = cursor.execute("select name from sqlite_master where type = 'table';").fetchone()[0]
    size = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    threshold = int(size * 0.9)
    rows = cursor.execute(
        f"SELECT DISTINCT challenge, response FROM {table_name} WHERE id <= {threshold}").fetchall()
    return prepare_data(rows, img_size, crop_size, do_crop)


def get_test_data(cursor, img_size, crop_size, do_crop):
    """
    Retrieves and prepares the data that will be used for the testing of the linear attacks.

    :param cursor: cursor object from sqlite to access the data
    :param img_size: size of the responses
    :param crop_size: size to which the responses will be cropped if decided to
    :param do_crop: whether to crop the responses
    :return: prepared data as tuple of challenges and responses
    """
    table_name = cursor.execute("select name from sqlite_master where type = 'table';").fetchone()[0]
    size = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    threshold = int(size * 0.9)
    rows = cursor.execute(
        f"SELECT DISTINCT challenge, response FROM {table_name} WHERE id > {threshold} ").fetchall()
    return prepare_data(rows, img_size, crop_size, do_crop)


def run_lr(c_train, r_train, *args):
    """
    Fits the LR model and returns the results of the attack.

    :param c_train: training challenges
    :param r_train: training responses
    :param args: further arguments for the testing and evaluation of the attack
    :return: results of the attack for the test data
    """
    print("Starting linear regression...")
    reg = LinearRegression().fit(c_train, r_train)
    print("Fitting finished!")
    return evaluate_linear_attack(reg, *args)


def run_ridge(c_train, r_train, *args, alpha):
    """
    Fits the ridge model and returns the results of the attack.

    :param c_train: training challenges
    :param r_train: training responses
    :param args: further arguments for the testing and evaluation of the attack
    :param alpha: penalty term used for ridge
    :return: results of the attack for the test data
    """
    print(f"Starting Ridge...")
    reg = Ridge(alpha=alpha).fit(c_train, r_train)
    print("Fitting finished!")
    return evaluate_linear_attack(reg, *args)


def run_opr_lr(c_bits, c_train, r_train, c_test, r_test, *args):
    """
    Fits the OPR LR model using the quadratic challenge transformation and returns the results of the attack.

    :param c_bits: number of bits of a challenge
    :param c_train: training challenges
    :param r_train: training responses
    :param c_test: test challenges
    :param r_test: test responses
    :param args: further arguments for the testing and evaluation of the attack
    :return: results of the attack for the test data
    """
    c_train_quadratic = np.empty((c_train.shape[0], c_bits * (c_bits + 1) // 2))
    idx = 0
    for i in range(c_bits):
        for j in range(i + 1):
            c_train_quadratic[:, idx] = c_train[:, i] * c_train[:, j]
            idx += 1
    c_test_quadratic = np.empty((c_test.shape[0], c_bits * (c_bits + 1) // 2))
    idx = 0
    for i in range(c_bits):
        for j in range(i + 1):
            c_test_quadratic[:, idx] = c_test[:, i] * c_test[:, j]
            idx += 1

    print("Starting OPR LR...")
    reg = LinearRegression().fit(c_train_quadratic, r_train)
    print("Fitting finished!")
    return evaluate_linear_attack(reg, c_test_quadratic, r_test, *args)


def run_opr_ridge(c_bits, c_train, r_train, c_test, r_test, *args, alpha):
    """
    Fits the OPR ridge model using the quadratic challenge transformation and returns the results of the attack.

    :param c_bits: number of bits of a challenge
    :param c_train: training challenges
    :param r_train: training responses
    :param c_test: test challenges
    :param r_test: test responses
    :param args: further arguments for the testing and evaluation of the attack
    :param alpha: penalty term used for ridge
    :return: results of the attack for the test data
    """
    c_train_and = np.empty((c_train.shape[0], c_bits * (c_bits + 1) // 2))
    idx = 0
    for i in range(c_bits):
        for j in range(i + 1):
            c_train_and[:, idx] = c_train[:, i] * c_train[:, j]
            idx += 1
    c_test_and = np.empty((c_test.shape[0], c_bits * (c_bits + 1) // 2))
    idx = 0
    for i in range(c_bits):
        for j in range(i + 1):
            c_test_and[:, idx] = c_test[:, i] * c_test[:, j]
            idx += 1

    print(f"Starting OPR ridge...")
    reg = Ridge(alpha=alpha).fit(c_train_and, r_train)
    print("Fitting finished!")
    return evaluate_linear_attack(reg, c_test_and, r_test, *args)


def evaluate_linear_attack(reg, c_test, r_test, img_size, crop_size, do_crop, custom_gabor):
    """
    Evaluates the fitted linear attack for the test data and returns the absolute difference, pearson correlation 
    coefficient, structural similarity index and fractional hamming distance for each real and predicted response pair.
    
    :param reg: fitted linear model
    :param c_test: test challenges
    :param r_test: test responses
    :param img_size: size of the responses
    :param crop_size: size to which the responses will be cropped if decided to
    :param do_crop: whether to crop the responses
    :param custom_gabor: whether to use the second gabor transformation
    :return: list containing the evaluated metrics for the test data
    """
    r_preds = reg.predict(c_test)
    if do_crop:
        img_size = crop_size
    r_test = r_test.reshape(-1, img_size, img_size)
    r_preds = r_preds.reshape(-1, img_size, img_size)

    abs_diffs = []
    pcs = []
    ssims = []
    fhds = []

    for r_idx, r in enumerate(tqdm(r_test, leave=True, position=0)):
        r_pred = r_preds[r_idx]

        fhd = utils.calc_gabor_fhd(r, r_pred, not do_crop, img_size, crop_size, use_custom=custom_gabor)
        abs_diff = np.mean(np.absolute((r - r_pred))) * 255
        pc, _ = pearsonr(r.flatten(), r_pred.flatten().astype(r.dtype))
        ssim = calc_ssim(r, r_pred.astype(r.dtype))

        fhds.append(fhd)
        pcs.append(pc)
        ssims.append(ssim)
        abs_diffs.append(abs_diff)

    return [abs_diffs, pcs, ssims, fhds]


def run_linear(db_name, img_size, crop_size, do_crop, db_folder, root, log_folder, custom_gabor, only_lr):
    '''
    Runs all linear attacks on the provided dataset.
    
    :param db_name: name of the db file that contains the data
    :param img_size: size of the responses
    :param crop_size: size to which the responses will be cropped if decided to
    :param do_crop: whether to crop the responses
    :param db_folder: name of the folder that contains the db file
    :param root: root folder directory where the folder for the datasets is stored
    :param log_folder: folder where the results of the attacks will be stored
    :param custom_gabor: whether to use the second gabor transformation
    :param only_lr: whether to only run the linear regression attack
    :return: results of all linear attacks
    '''
    data = {}

    tmp_path = f'{log_folder}/tmp/{db_name}_linear_results{"_Crop" if do_crop else ""}.json'
    db_file = f"{root}/{db_folder}/{db_name}.db"
    c_bits = int(db_name.split('b')[0])
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    c_train, r_train = get_training_data(cursor, img_size, crop_size, do_crop)
    c_test, r_test = get_test_data(cursor, img_size, crop_size, do_crop)

    with open("linear/hparams.json", "r") as file:
        hparams = json.load(file)
        try:
            ridge_alpha = hparams[db_folder]["Normal"][db_name]
            ridge_opr_alpha = hparams[db_folder]["OPR"][db_name]
        except Exception:
            print("Missing alpha value on db", db_name, "- falling back to alpha=1")
            ridge_alpha = 1
            ridge_opr_alpha = 1

    if do_crop:
        img_size = img_size // 4

    columns = ['Abs. diff.', 'PC', 'SSIM', 'FHD']

    args = [c_train, r_train, c_test, r_test, img_size, crop_size, do_crop, custom_gabor]

    lr_results = run_lr(*args)
    data["LR"] = dict(zip(columns, lr_results))

    if not only_lr:
        ridge_results = run_ridge(*args, alpha=ridge_alpha)
        data["Ridge"] = dict(zip(columns, ridge_results))

        opr_lr_results = run_opr_lr(c_bits, *args)
        data["OPR"] = dict(zip(columns, opr_lr_results))

        opr_ridge_results = run_opr_ridge(c_bits, *args, alpha=ridge_opr_alpha)
        data["OPR_Ridge"] = dict(zip(columns, opr_ridge_results))

    conn.close()

    with open(tmp_path, 'w') as file:
        json.dump(data, file)

    return data
