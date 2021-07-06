import argparse
import copy
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def prepare_attack_results(result_folder):
    """
    Prepares and returns the results of the ML attacks such that it can be visualized using pandas and seaborn.

    :param result_folder: directory where the results of the attacks are stored
    :return: tuple of: (FHD [Gabor 1], FHD [Gabor 2], PC, SSIM)
    """
    sim_attack_folders = ["x1", "x2", "large"]
    fhd_g1_results = {
        "dl": {
            "Generator": {
                "Simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Larger simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Simulation (10k CRPs)": {
                    "7x7": {"Type A": {}},
                    "13x13": {"Type A": {}}
                },
                "Real PUF": {"16x20": {"Type A": {}}}
            },
            "Cropped generator (type 1)": {
                "Simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Larger simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Simulation (10k CRPs)": {
                    "7x7": {"Type A": {}},
                    "13x13": {"Type A": {}}
                },
                "Real PUF": {"16x20": {"Type A": {}}}
            },
            "Cropped generator (type 2)": {
                "Simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Larger simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Simulation (10k CRPs)": {
                    "7x7": {"Type A": {}},
                    "13x13": {"Type A": {}}
                },
                "Real PUF": {"16x20": {"Type A": {}}}
            },
            "Advanced generator": {
                "Simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Larger simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Simulation (10k CRPs)": {
                    "7x7": {"Type A": {}},
                    "13x13": {"Type A": {}}
                },
                "Real PUF": {"16x20": {"Type A": {}}}
            }},
        "linear": {
            "LR": {
                "Simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Larger simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Simulation (10k CRPs)": {
                    "7x7": {"Type A": {}},
                    "13x13": {"Type A": {}}
                },
                "Real PUF": {"16x20": {"Type A": {}}}
            },
            "RR": {
                "Simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Larger simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Simulation (10k CRPs)": {
                    "7x7": {"Type A": {}},
                    "13x13": {"Type A": {}}
                },
                "Real PUF": {"16x20": {"Type A": {}}}
            },
            "QLR": {
                "Simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Larger simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Simulation (10k CRPs)": {
                    "7x7": {"Type A": {}},
                    "13x13": {"Type A": {}}
                },
                "Real PUF": {"16x20": {"Type A": {}}}
            },
            "QRR": {
                "Simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Larger simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Simulation (10k CRPs)": {
                    "7x7": {"Type A": {}},
                    "13x13": {"Type A": {}}
                },
                "Real PUF": {"16x20": {"Type A": {}}}
            }}
    }
    pc_results = copy.deepcopy(fhd_g1_results)
    ssim_results = copy.deepcopy(fhd_g1_results)
    for folder in sim_attack_folders:
        _, _, data_files = next(os.walk(f"{result_folder}/attack_results_{folder}/tmp"))
        for data_file in data_files:
            if "dl_results_Crop_NS" in data_file:
                architecture = "dl"
                model = "Cropped generator (type 2)"
            elif "dl_results_Crop" in data_file:
                architecture = "dl"
                model = "Cropped generator (type 1)"
            elif "dl_results" in data_file:
                architecture = "dl"
                model = "Generator"
            elif "gen_results" in data_file:
                architecture = "dl"
                model = "Advanced generator"
            elif "linear_results_Crop" in data_file:
                architecture = "linear"
                model = False
                is_crop = True
            elif "linear_results" in data_file:
                architecture = "linear"
                model = False
                is_crop = False
            else:
                print("Not fitting data file", data_file)
                continue

            blocks = int(data_file.split("b")[0])
            if blocks == 320:
                block_str = "16x20"
            else:
                blocks_per_row = int(np.sqrt(blocks))
                block_str = f"{blocks_per_row}x{blocks_per_row}"

            with open(f"{result_folder}/attack_results_{folder}/tmp/{data_file}", "r") as file:
                data = json.load(file)
                type = "Type A"
                if "2nd" in data_file:
                    type = "Type B"
                elif "mb" in data_file:
                    max_blocks = int(data_file.split("mb")[1].split("_")[0])
                    if (blocks - 1) // max_blocks == 2:
                        type = "Type C"
                    else:
                        type = "Type D"
                if folder == "x1":
                    dataset = "Simulation"
                elif folder == "x2":
                    dataset = "Larger simulation"
                elif folder == "real_custom_gabor":
                    dataset = "Real PUF"
                elif folder == "large":
                    dataset = "Simulation (10k CRPs)"
                if model:
                    data_obj = {"FHD": data["FHD"], "PC": data["PC"], "SSIM": data["SSIM"]}
                    fhd_g1_results[architecture][model][dataset][block_str][type] = data_obj["FHD"]
                    pc_results[architecture][model][dataset][block_str][type] = data_obj["PC"]
                    ssim_results[architecture][model][dataset][block_str][type] = data_obj["SSIM"]
                else:
                    if is_crop:
                        data_opr_lr = data["OPR"]
                        data_obj_opr_lr = {"FHD": data_opr_lr["FHD"], "PC": data_opr_lr["PC"],
                                           "SSIM": data_opr_lr["SSIM"]}
                        fhd_g1_results[architecture]["QLR"][dataset][block_str][type] = data_obj_opr_lr["FHD"]
                        pc_results[architecture]["QLR"][dataset][block_str][type] = data_obj_opr_lr["PC"]
                        ssim_results[architecture]["QLR"][dataset][block_str][type] = data_obj_opr_lr["SSIM"]

                        data_opr_rr = data["OPR_Ridge"]
                        data_obj_opr_rr = {"FHD": data_opr_rr["FHD"], "PC": data_opr_rr["PC"],
                                           "SSIM": data_opr_rr["SSIM"]}
                        fhd_g1_results[architecture]["QRR"][dataset][block_str][type] = data_obj_opr_rr["FHD"]
                        pc_results[architecture]["QRR"][dataset][block_str][type] = data_obj_opr_rr["PC"]
                        ssim_results[architecture]["QRR"][dataset][block_str][type] = data_obj_opr_rr["SSIM"]
                    else:
                        data_lr = data["LR"]
                        data_obj_lr = {"FHD": data_lr["FHD"], "PC": data_lr["PC"], "SSIM": data_lr["SSIM"]}
                        fhd_g1_results[architecture]["LR"][dataset][block_str][type] = data_obj_lr["FHD"]
                        pc_results[architecture]["LR"][dataset][block_str][type] = data_obj_lr["PC"]
                        ssim_results[architecture]["LR"][dataset][block_str][type] = data_obj_lr["SSIM"]

                        data_rr = data["Ridge"]
                        data_obj_rr = {"FHD": data_rr["FHD"], "PC": data_rr["PC"], "SSIM": data_rr["SSIM"]}
                        fhd_g1_results[architecture]["RR"][dataset][block_str][type] = data_obj_rr["FHD"]
                        pc_results[architecture]["RR"][dataset][block_str][type] = data_obj_rr["PC"]
                        ssim_results[architecture]["RR"][dataset][block_str][type] = data_obj_rr["SSIM"]

    fhd_g2_results = {
        "dl": {
            "Generator": {
                "Simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Larger simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Simulation (10k CRPs)": {
                    "7x7": {"Type A": {}},
                    "13x13": {"Type A": {}}
                },
                "Real PUF": {"16x20": {"Type A": {}}}
            },
            "Cropped generator (type 1)": {
                "Simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Larger simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Simulation (10k CRPs)": {
                    "7x7": {"Type A": {}},
                    "13x13": {"Type A": {}}
                },
                "Real PUF": {"16x20": {"Type A": {}}}
            },
            "Cropped generator (type 2)": {
                "Simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Larger simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Simulation (10k CRPs)": {
                    "7x7": {"Type A": {}},
                    "13x13": {"Type A": {}}
                },
                "Real PUF": {"16x20": {"Type A": {}}}
            },
            "Advanced generator": {
                "Simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Larger simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Simulation (10k CRPs)": {
                    "7x7": {"Type A": {}},
                    "13x13": {"Type A": {}}
                },
                "Real PUF": {"16x20": {"Type A": {}}}
            }},
        "linear": {
            "LR": {
                "Simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Larger simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Simulation (10k CRPs)": {
                    "7x7": {"Type A": {}},
                    "13x13": {"Type A": {}}
                },
                "Real PUF": {"16x20": {"Type A": {}}}
            },
            "RR": {
                "Simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Larger simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Simulation (10k CRPs)": {
                    "7x7": {"Type A": {}},
                    "13x13": {"Type A": {}}
                },
                "Real PUF": {"16x20": {"Type A": {}}}
            },
            "QLR": {
                "Simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Larger simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Simulation (10k CRPs)": {
                    "7x7": {"Type A": {}},
                    "13x13": {"Type A": {}}
                },
                "Real PUF": {"16x20": {"Type A": {}}}
            },
            "QRR": {
                "Simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Larger simulation": {
                    "5x5": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "7x7": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "9x9": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "11x11": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "13x13": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                    "15x15": {
                        "Type A": {},
                        "Type B": {},
                        "Type C": {},
                        "Type D": {},
                    },
                },
                "Simulation (10k CRPs)": {
                    "7x7": {"Type A": {}},
                    "13x13": {"Type A": {}}
                },
                "Real PUF": {"16x20": {"Type A": {}}}
            }}
    }

    sim_attack_folders_g2 = ["x1_custom_gabor", "x2_custom_gabor", "real_custom_gabor"]
    for folder in sim_attack_folders_g2:
        os.walk(f"{result_folder}/attack_results_{folder}/tmp")
        _, _, data_files = next(os.walk(f"{result_folder}/attack_results_{folder}/tmp"))
        for data_file in data_files:
            if "dl_results_Crop_NS" in data_file:
                architecture = "dl"
                model = "Cropped generator (type 2)"
            elif "dl_results_Crop" in data_file:
                architecture = "dl"
                model = "Cropped generator (type 1)"
            elif "dl_results" in data_file:
                architecture = "dl"
                model = "Generator"
            elif "gen_results" in data_file:
                architecture = "dl"
                model = "Advanced generator"
            elif "linear_results_Crop" in data_file:
                architecture = "linear"
                model = False
                is_crop = True
            elif "linear_results" in data_file:
                architecture = "linear"
                model = False
                is_crop = False
            else:
                print("Not fitting data file", data_file)
                continue

            blocks = int(data_file.split("b")[0])
            if blocks == 320:
                block_str = "16x20"
            else:
                blocks_per_row = int(np.sqrt(blocks))
                block_str = f"{blocks_per_row}x{blocks_per_row}"

            with open(f"{result_folder}/attack_results_{folder}/tmp/{data_file}", "r") as file:
                data = json.load(file)
                type = "Type A"
                if "2nd" in data_file:
                    type = "Type B"
                elif "mb" in data_file:
                    max_blocks = int(data_file.split("mb")[1].split("_")[0])
                    if (blocks - 1) // max_blocks == 2:
                        type = "Type C"
                    else:
                        type = "Type D"
                if folder == "x1_custom_gabor":
                    dataset = "Simulation"
                elif folder == "x2_custom_gabor":
                    dataset = "Larger simulation"
                elif folder == "real_custom_gabor":
                    dataset = "Real PUF"
                if model:
                    data_obj = {"FHD": data["FHD"], "PC": data["PC"], "SSIM": data["SSIM"]}
                    fhd_g2_results[architecture][model][dataset][block_str][type] = data_obj["FHD"]
                else:
                    if is_crop:
                        data_opr_lr = data["OPR"]
                        data_obj_opr_lr = {"FHD": data_opr_lr["FHD"], "PC": data_opr_lr["PC"],
                                           "SSIM": data_opr_lr["SSIM"]}
                        fhd_g2_results[architecture]["QLR"][dataset][block_str][type] = data_obj_opr_lr["FHD"]

                        data_opr_rr = data["OPR_Ridge"]
                        data_obj_opr_rr = {"FHD": data_opr_rr["FHD"], "PC": data_opr_rr["PC"],
                                           "SSIM": data_opr_rr["SSIM"]}
                        fhd_g2_results[architecture]["QRR"][dataset][block_str][type] = data_obj_opr_rr["FHD"]
                    else:
                        data_lr = data["LR"]
                        data_obj_lr = {"FHD": data_lr["FHD"], "PC": data_lr["PC"], "SSIM": data_lr["SSIM"]}
                        fhd_g2_results[architecture]["LR"][dataset][block_str][type] = data_obj_lr["FHD"]

                        data_rr = data["Ridge"]
                        data_obj_rr = {"FHD": data_rr["FHD"], "PC": data_rr["PC"], "SSIM": data_rr["SSIM"]}
                        fhd_g2_results[architecture]["RR"][dataset][block_str][type] = data_obj_rr["FHD"]

    return fhd_g1_results, fhd_g2_results, pc_results, ssim_results


def plot_attack_results(fhd_results_g1, fhd_results_g2, pc_results, ssim_results):
    """
    Creates a matrix box plot for each type of simulation and linear/non-linear attacks. Each of these contains all four
    metrics (FHD Gabor 1, FHD Gabor 2, PC, SSIM) and all ML attacks of the corresponding group for all datasets of the
    simulation.

    :param fhd_results_g1: FHD results for the first Gabor transformation
    :param fhd_results_g2: FHD results for the second Gabor transformation
    :param pc_results: PC results
    :param ssim_results: SSIM results
    """
    dl_attack_results = {"FHD (Gabor 1)": fhd_results_g1["dl"], "FHD (Gabor 2)": fhd_results_g2["dl"],
                         "PC": pc_results["dl"], "SSIM": ssim_results["dl"]}
    linear_attack_results = {"FHD (Gabor 1)": fhd_results_g1["linear"], "FHD (Gabor 2)": fhd_results_g2["linear"],
                             "PC": pc_results["linear"], "SSIM": ssim_results["linear"]}
    for ml_type, attack_results in zip(["dl", "linear"], [dl_attack_results, linear_attack_results]):
        for dataset in ["Larger simulation", "Simulation"]:
            fhd_pandas_sim = {(attack, size, datatype, metric): attack_results[metric][attack][dataset][size][datatype]
                              for metric in ["FHD (Gabor 1)", "FHD (Gabor 2)", "PC", "SSIM"]
                              for attack in fhd_results_g1[ml_type].keys()
                              for size in attack_results[metric][attack][dataset].keys()
                              for datatype in attack_results[metric][attack][dataset][size].keys()
                              }

            df = pd.DataFrame.from_dict(fhd_pandas_sim, orient="index")
            df.index = pd.MultiIndex.from_tuples(df.index, names=("Attack", "Challenge size", "Type", "Metric"))
            df = df.stack().reset_index()
            df = df.drop(columns=["level_4"])
            df = df.rename(columns={0: "Value"})

            params = {'legend.fontsize': 24,
                      'figure.figsize': (10, 5),
                      'figure.titlesize': 35,
                      'axes.labelsize': 23,
                      'axes.labelpad': 20,
                      'axes.titlesize': 50,
                      'axes.titlepad': 30,
                      'xtick.labelsize': 15,
                      'ytick.labelsize': 16}
            plt.rcParams.update(params)

            sns.set_context("paper", rc=params)

            # Sim
            g = sns.catplot(x="Challenge size", y="Value", hue="Type", col="Attack",
                            row="Metric", data=df, kind="box", legend=False, sharey="row",
                            palette=["gold", "darkorange", "indianred", "darkred"], margin_titles=True)
            g.set_titles(col_template="{col_name}", row_template="")
            axes = g.axes
            plt.subplots_adjust(hspace=0.1)

            axes[0, 0].set_ylabel("FHD (Gabor 1)")
            axes[1, 0].set_ylabel("FHD (Gabor 2)")
            axes[2, 0].set_ylabel("PC")
            axes[3, 0].set_ylabel("SSIM")

            if ml_type == "dl":
                if dataset == "larger simulation":
                    set_dl_axis_larger_simulation(axes)
                else:
                    set_dl_axis_simulation(axes)
            else:
                if dataset == "larger simulation":
                    set_linear_axis_larger_simulation(axes)
                else:
                    set_linear_axis_simulation(axes)

            plt.legend(loc="upper right", bbox_to_anchor=(1, 4.825), ncol=4, frameon=False)
            dataset = "Larger_simulation" if dataset == "Larger simulation" else dataset

            g.savefig(fname=f"attack_fhds_{ml_type}_{dataset}", bbox_inches='tight', pad_inches=0)


def set_dl_axis_larger_simulation(axes):
    """
    Sets the axes for the metrics for the DL attacks on the larger simulation.

    :param axes: axes object to set the axis parameters for
    """
    for axs in (list(axes[0]) + list(axes[1])):
        steps = np.arange(0, 0.5001, 0.1)
        for step in steps:
            axs.axhline(step, linestyle="dashed", color="gray", linewidth=1)
        axs.set_ylim(0, 0.5)
        axs.set_yticks(steps)

    for axs in axes[2]:
        steps = np.arange(0.6, 1.00001, 0.1)
        for step in steps:
            axs.axhline(step, linestyle="dashed", color="gray", linewidth=1)
        axs.set_ylim(0.6, 1)
        axs.set_yticks(steps)

    for axs in axes[3]:
        steps = np.arange(0.4, 1.00001, 0.1)
        for step in steps:
            axs.axhline(step, linestyle="dashed", color="gray", linewidth=1)
        axs.set_ylim(0.4, 1)
        axs.set_yticks(steps)


def set_linear_axis_larger_simulation(axes):
    """
    Sets the axes for the metrics for the linear attacks on the larger simulation.

    :param axes: axes object to set the axis parameters for
    """
    for axs in (list(axes[0]) + list(axes[1])):
        steps = np.arange(0, 0.5001, 0.1)
        for step in steps:
            axs.axhline(step, linestyle="dashed", color="gray", linewidth=1)
        axs.set_ylim(0, 0.5)
        axs.set_yticks(steps)

    for axs in axes[2]:
        steps = np.arange(0.2, 1.00001, 0.1)
        for step in steps:
            axs.axhline(step, linestyle="dashed", color="gray", linewidth=1)
        axs.set_ylim(0.2, 1)
        axs.set_yticks(steps)

    for axs in axes[3]:
        steps = np.arange(0.2, 1.00001, 0.1)
        for step in steps:
            axs.axhline(step, linestyle="dashed", color="gray", linewidth=1)
        axs.set_ylim(0.2, 1)
        axs.set_yticks(steps)


def set_dl_axis_simulation(axes):
    """
    Sets the axes for the metrics for the DL attacks on the default simulation.

    :param axes: axes object to set the axis parameters for
    """
    for axs in (list(axes[0]) + list(axes[1])):
        steps = np.arange(0, 0.5001, 0.1)
        for step in steps:
            axs.axhline(step, linestyle="dashed", color="gray", linewidth=1)
        axs.set_ylim(0, 0.5)
        axs.set_yticks(steps)

    for axs in axes[2]:
        steps = np.arange(0.7, 1.00001, 0.1)
        for step in steps:
            axs.axhline(step, linestyle="dashed", color="gray", linewidth=1)
        axs.set_ylim(0.6, 1)
        axs.set_yticks(steps)

    for axs in axes[3]:
        steps = np.arange(0.5, 1.00001, 0.1)
        for step in steps:
            axs.axhline(step, linestyle="dashed", color="gray", linewidth=1)
        axs.set_ylim(0.4, 1)
        axs.set_yticks(steps)


def set_linear_axis_simulation(axes):
    """
    Sets the axes for the metrics for the linear attacks on the default simulation.

    :param axes: axes object to set the axis parameters for
    """
    for axs in (list(axes[0]) + list(axes[1])):
        steps = np.arange(0, 0.5001, 0.1)
        for step in steps:
            axs.axhline(step, linestyle="dashed", color="gray", linewidth=1)
        axs.set_ylim(0, 0.5)
        axs.set_yticks(steps)

    for axs in axes[2]:
        steps = np.arange(0.7, 1.00001, 0.1)
        for step in steps:
            axs.axhline(step, linestyle="dashed", color="gray", linewidth=1)
        axs.set_ylim(0.6, 1)
        axs.set_yticks(steps)

    for axs in axes[3]:
        steps = np.arange(0.2, 1.00001, 0.1)
        for step in steps:
            axs.axhline(step, linestyle="dashed", color="gray", linewidth=1)
        axs.set_ylim(0.4, 1)
        axs.set_yticks(steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True, help="folder path where the results are stored.")
    args = parser.parse_args()

    attack_fhd_g1_results, attack_fhd_g2_results, pc_results, ssim_results = prepare_attack_results(args.folder)
    plot_attack_results(attack_fhd_g1_results, attack_fhd_g2_results, pc_results, ssim_results)


if __name__ == "__main__":
    main()
