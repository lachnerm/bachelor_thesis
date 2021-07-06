import json
import os

from pytorch_lightning import Trainer

from non_linear.Advanced_Generator import AdvancedGeneratorModel
from non_linear.Generator import GeneratorModel
from non_linear.modules.DataModule import AttackDataModule


def run_default_Generator(db_name, db_folder, img_size, crop_size, root, log_folder, custom_gabor, do_crop, crop_type2):
    """
    Runs the default generator DL attack.
    
    :param db_name: name of the db file that contains the data
    :param db_folder: name of the folder that contains the db file
    :param img_size: size of the responses
    :param crop_size: size to which the responses will be cropped if decided to
    :param root: root folder directory where the folder for the datasets is stored
    :param log_folder: folder where the results of the attacks will be stored
    :param custom_gabor: whether to use the second gabor transformation
    :param do_crop: whether to crop the responses
    :param crop_type2: whether to use the cropped generator type 2
    :return: results of the DL attack
    """
    tmp_path = f'{log_folder}/tmp/{db_name}_dl_results{"_Crop" if do_crop else ""}{"_type2" if crop_type2 else ""}.json'
    hparams = {"bs": 32,
               "gen_lr": 0.002,
               "gen_beta1": 0.5,
               "gen_beta2": 0.999,
               "gen_ns": 32,
               }
    epochs = 100

    c_bits = int(db_name.split('b')[0])
    db_file = f"{root}/{db_folder}/{db_name}.db"

    data_module = AttackDataModule(hparams["bs"], db_file, img_size, crop_size, do_crop)
    data_module.setup()
    img_size = crop_size if do_crop else img_size
    model = GeneratorModel(hparams, img_size, crop_size, c_bits, data_module.denormalize, do_crop, crop_type2,
                           custom_gabor)
    trainer = Trainer(gpus=1, max_epochs=epochs, logger=False, checkpoint_callback=False)

    trainer.fit(model, datamodule=data_module)
    data = trainer.test(model, datamodule=data_module)[0]

    with open(tmp_path, 'w') as file:
        json.dump(data, file)

    return data


def run_advanced_generator(db_name, db_folder, img_size, crop_size, root, log_folder, custom_gabor, do_crop):
    """
    Runs the advanced generator DL attack.

    :param db_name: name of the db file that contains the data
    :param db_folder: name of the folder that contains the db file
    :param img_size: size of the responses
    :param crop_size: size to which the responses will be cropped if decided to
    :param root: root folder directory where the folder for the datasets is stored
    :param log_folder: folder where the results of the attacks will be stored
    :param custom_gabor: whether to use the second gabor transformation
    :param do_crop: whether to crop the responses
    :return: results of the DL attack
    """
    tmp_path = f'{log_folder}/tmp/{db_name}_gen_results{"_Crop" if do_crop else ""}.json'
    if os.path.isfile(tmp_path):
        with open(tmp_path, 'r') as tmp:
            results = json.load(tmp)
            print(f"Found tmp file for Generator attack on db {db_name}!")
            return results

    hparams = {"bs": 16,
               "c_weight": 20,
               "gen_lr": 0.01,
               "gen_beta1": 0.8,
               "gen_beta2": 0.9,
               "ns": 64
               }

    challenge_bits = int(db_name.split('b')[0])
    db_file = f"{root}/{db_folder}/{db_name}.db"
    epochs = 300

    data_module = AttackDataModule(hparams["bs"], db_file, img_size, crop_size, do_crop)
    data_module.setup()
    # advanced generator only works with cropped responses
    img_size = crop_size
    model = AdvancedGeneratorModel(hparams, img_size, crop_size, challenge_bits, data_module.denormalize, do_crop,
                                   custom_gabor)
    trainer = Trainer(gpus=1, max_epochs=epochs, logger=False, checkpoint_callback=False)

    trainer.fit(model, datamodule=data_module)
    data = trainer.test(model, datamodule=data_module)[0]

    with open(tmp_path, 'w') as file:
        json.dump(data, file)

    return data


def run_dl(only_default, only_advanced, ct2, do_crop, *args):
    """
    Runs the specified DL attacks.

    :param only_default: whether to use only the default generator
    :param only_advanced: whether to use only the advanced generator
    :param ct2: whether to use the cropped generator type 2
    :param args: further arguments that specify the details and restrictions for the attacks
    :return: results of the DL attacks
    """
    results = {}
    if not only_advanced:
        results["Generator"] = run_default_Generator(*args, do_crop, ct2)
    if not only_default and not ct2 and do_crop:
        results["Advanced Generator"] = run_advanced_generator(*args, do_crop)
    return results
