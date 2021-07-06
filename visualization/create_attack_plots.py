import matplotlib.pyplot as plt
import numpy as np

from utils import utils


def create_attack_plots(results, log_folder, db_folder, do_crop, ct2):
    """
    Plots the results obtained from the machine learning attacks on several datasets. Assumes that the results contain
    the FHD, PC and SSIM. Creates both a scatter plot and a box plot for each metric on each attack.

    :param results: results of the attacks
    :param log_folder: folder where the results of the attacks are stored
    :param db_folder: name of the folder that contains the db files
    :param do_crop: whether the responses were cropped
    :param ct2: whether the cropped generator type 2 was used for the DL attacks
    """

    results = utils.beautify_results_db_names(results)
    plt_results_names = list(map(lambda result: result[0], results))
    scatter_x_tick_labels = list(
        dict.fromkeys((list(map(lambda result_name: result_name.split("|")[0], plt_results_names)))))
    scatter_x_tick_labels = list(
        map(lambda label: f"{int(np.sqrt(int(label)))} x {int(np.sqrt(int(label)))}", scatter_x_tick_labels))

    type_label_and_color = [("Type A", "gold"),
                            ("Type B", "darkorange"),
                            ("Type C", "indianred"),
                            ("Type D", "darkred")]

    steps = lambda lims, steps: np.linspace(lims[0], lims[1], steps)

    scatter_lims = {"FHD": [0.0, 0.4], "PC": [0.8, 1], "SSIM": [0.97, 0.985]}
    scatter_steps = {"FHD": steps(scatter_lims["FHD"], 6), "PC": steps(scatter_lims["PC"], 4),
                     "SSIM": steps(scatter_lims["SSIM"], 16)}
    box_lims = {"FHD": [0, 0.5], "PC": [0.7, 1], "SSIM": [0.9, 1]}
    box_steps = {"FHD": steps(box_lims["FHD"], 6), "PC": steps(box_lims["PC"], 4), "SSIM": steps(box_lims["SSIM"], 4)}

    relevant_attacks = {"Advanced Generator", "Generator", "LR", "OPR", "OPR_Ridge", "Ridge"}

    for metric in ["FHD", "PC", "SSIM"]:
        for attack_type in results[0][1]:
            if attack_type in relevant_attacks:
                plt_results_data = list(map(lambda result: result[1][attack_type][metric], results))

                # --------------------------------------
                # ------------ SCATTER PLOT ------------
                # --------------------------------------

                y_steps = scatter_steps[metric]

                fig = plt.figure(111, figsize=(11, 10))
                ax_scatter = fig.gca()
                ax_scatter.set_ylim(scatter_lims[metric])
                ax_scatter.spines["top"].set_visible(False)
                ax_scatter.spines["right"].set_visible(False)
                for step in y_steps:
                    ax_scatter.axhline(step, linestyle="dashed", color="gray", linewidth=1)

                box_colors = []
                mean_tmp = []

                for idx, result in enumerate(results):
                    x, y = result
                    color, label = get_plt_color_and_label(x, type_label_and_color)
                    box_colors.append(color)
                    # placeholder color for empty spacing entry
                    if (idx + 1) % 4 == 0:
                        box_colors.append("white")
                    x = x.split("|")[0]
                    y = y[attack_type][metric]
                    y = np.mean(y)
                    mean_tmp.append(y)
                    ax_scatter.scatter(x, y, s=100, color=color, label=label)

                ax_handles, ax_labels = ax_scatter.get_legend_handles_labels()
                unique_labels = [(h, l) for i, (h, l) in enumerate(zip(ax_handles, ax_labels)) if
                                 l not in ax_labels[:i]]
                ax_scatter.legend(*zip(*unique_labels), loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=6,
                                  frameon=False)

                ax_scatter.titlesize = 28
                xtickNames = plt.setp(ax_scatter, xticklabels=scatter_x_tick_labels)
                plt.setp(xtickNames, fontsize=18)
                plt.yticks(y_steps, fontsize=18)
                ax_scatter.set_xlabel("Blocks", fontsize=18, labelpad=25)
                ax_scatter.set_ylabel(metric if metric != "FHD" else "Fractional Hamming Distance", fontsize=18,
                                      labelpad=20)

                fig.savefig(
                    fname=f"{log_folder}/{db_folder}_{metric}_{attack_type}_scatter{'_crop' if do_crop else ''}{'_NS' if ct2 else ''}.jpg",
                    bbox_inches='tight',
                    pad_inches=0)
                plt.close("all")

                # --------------------------------------
                # -------------- BOX PLOT --------------
                # --------------------------------------

                y_steps = box_steps[metric]

                fig = plt.figure(111, figsize=(20, 10))
                ax_box = fig.gca()
                ax_box.set_ylim(box_lims[metric])
                ax_box.spines["top"].set_visible(False)
                ax_box.spines["right"].set_visible(False)
                for step in y_steps:
                    ax_box.axhline(step, linestyle="dashed", color="gray", linewidth=1)

                # add empty entries for spacing between same-sized groups
                i = 4
                while i < len(plt_results_data):
                    plt_results_data.insert(i, [])
                    plt_results_names.insert(i, "")
                    i += 5

                box = ax_box.boxplot(plt_results_data, patch_artist=True, medianprops=dict(linewidth=2, color='k'))
                ax_box.legend(*zip(*unique_labels), loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=6, fontsize=20,
                              markerscale=1.5, frameon=False)

                try:
                    plt.setp(ax_box, xticks=list(np.arange(2.5, 30, 5)), xticklabels=scatter_x_tick_labels)
                except Exception:
                    plt.setp(ax_box, xticks=list(range(len(scatter_x_tick_labels))), xticklabels=scatter_x_tick_labels)
                    pass
                plt.xticks(fontsize=25)
                plt.yticks(fontsize=25)
                ax_box.set_xlabel("Challenge size", fontsize=25, labelpad=25)
                if metric == "FHD":
                    metric_title = "Fractional Hamming distance"
                elif metric == "PC":
                    metric_title = "Pearson correlation"
                else:
                    metric_title = "SSIM"
                ax_box.set_ylabel(metric_title, fontsize=25, labelpad=35)

                for patch, color in zip(box["boxes"], box_colors):
                    patch.set_facecolor(color)
                fig.savefig(
                    fname=f"{log_folder}/{db_folder}_{metric}_{attack_type}_box{'_crop' if do_crop else ''}{'_NS' if ct2 else ''}.jpg",
                    bbox_inches='tight',
                    pad_inches=0)
                plt.close("all")


def get_plt_color_and_label(name, type_label_and_color):
    """
    Helper function that returns the corresponding color and label that will be used for the type of the dataset.

    :param name: name of the dataset
    :param type_label_and_color: all available labels and colors
    :return: tuple of label and color for the type of the dataset
    """
    if "2nd" in name:
        label, color = type_label_and_color[1]
    elif "m" in name:
        max_blocks = name.split("|")[1].split("m")[1]
        if max_blocks == "1/2":
            label, color = type_label_and_color[2]
        else:
            label, color = type_label_and_color[3]
    else:
        label, color = type_label_and_color[0]
    return color, label
