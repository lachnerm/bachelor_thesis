import matplotlib.pyplot as plt
import numpy as np

from utils import utils
from visualization.create_attack_plots import get_plt_color_and_label


def create_metric_plots(results, log_folder, db_folder, do_rename):
    """
    Plots the results obtained from the dataset evaluations. Assumes that the results contain the FHD, shannon
    entropy and cropped shannon entropy. Creates both a scatter plot and a box plot for each metric.

    :param results: results of the attacks
    :param log_folder: folder where the results of the attacks are stored
    :param db_folder: name of the folder that contains the db files
    :param do_rename: whether the database names of the results should be renamed
    """
    plt_results = utils.beautify_results_db_names(results) if do_rename else results
    plt_results_names = list(map(lambda result: result[0], plt_results))
    scatter_x_tick_labels = list(
        dict.fromkeys((list(map(lambda result_name: result_name.split("|")[0], plt_results_names)))))
    scatter_x_tick_labels = list(
        map(lambda label: f"{int(np.sqrt(int(label)))} x {int(np.sqrt(int(label)))}", scatter_x_tick_labels))
    is_grouped_data = len(plt_results_names) > len(scatter_x_tick_labels)

    labels = [("Type A", "mediumslateblue"),
              ("Type B", "teal"),
              ("Type C", "dodgerblue"),
              ("Type D", "mediumblue")]

    steps = lambda lims, steps: np.linspace(lims[0], lims[1], steps)

    scatter_lims = {"FHD": [0.3, 0.45], "Entropy": [2.5, 3], "Crop Entropy": [7.6, 7.7]}
    box_lims = {"FHD": [0, 0.65], "Entropy": [1.2, 2.6], "Crop Entropy": [6.5, 8]}
    box_steps = {"FHD": steps([0, 0.6], 7), "Entropy": steps(box_lims["Entropy"], 8),
                 "Crop Entropy": steps(box_lims["Crop Entropy"], 4)}

    for metric in plt_results[0][1]:
        plt_results_data = list(map(lambda result: result[1][metric], plt_results))

        fig = plt.figure(111, figsize=(11, 10))

        if len(plt_results_data) > 1:
            ax_scatter = fig.gca()
            ax_scatter.set_ylim(scatter_lims[metric])
            ax_scatter.spines["top"].set_visible(False)
            ax_scatter.spines["right"].set_visible(False)

            box_colors = []
            mean_tmp = []
            for idx, result in enumerate(plt_results):
                x, y = result
                color, label = get_plt_color_and_label(x, labels) if do_rename else ("black", x)
                box_colors.append(color)
                # placeholder color for empty spacing entry
                if (idx + 1) % 4 == 0:
                    box_colors.append("white")
                blocks = x.split("|")[0]
                blocks_per_row = int(np.sqrt(int(blocks)))
                x = f"{blocks_per_row} x {blocks_per_row}"
                y = np.mean(y[metric])
                mean_tmp.append(y)
                ax_scatter.scatter(x, y, s=100, color=color, label=label)

            if is_grouped_data:
                ax_handles, ax_labels = ax_scatter.get_legend_handles_labels()
                unique_labels = [(h, l) for i, (h, l) in enumerate(zip(ax_handles, ax_labels)) if
                                 l not in ax_labels[:i]]
                ax_scatter.legend(*zip(*unique_labels), loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=6,
                                  frameon=False)

            x_tick_names = plt.setp(ax_scatter, xticklabels=scatter_x_tick_labels)
            plt.setp(x_tick_names, fontsize=18)
            plt.yticks(fontsize=18)
            ax_scatter.set_xlabel("Blocks", fontsize=18, labelpad=20)
            if metric == "FHD":
                metric_title = "Fractional Hamming distance"
            else:
                metric_title = "Shannon entropy"
            ax_scatter.set_ylabel(metric_title, fontsize=18, labelpad=20)
        else:
            box_colors = ["black"]
            x = plt_results[0][0]
            ys = plt_results[0][1][metric]
            blocks = x.split("|")[0]
            blocks_per_row = int(np.sqrt(int(blocks)))
            x = f"{blocks_per_row} x {blocks_per_row}"
            plt.plot(x, np.mean(ys), "ko")

        fig.savefig(fname=f"{log_folder}/{db_folder}_{metric}_scatter.jpg")
        plt.close("all")

        # --------------------------------------
        # BOXPLOT
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
        if is_grouped_data:
            ax_box.legend(*zip(*unique_labels), loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=6, fontsize=20,
                          markerscale=1.5, frameon=False)

        if is_grouped_data:
            xticks = list(np.arange(2.5, len(scatter_x_tick_labels) * 5, 5))
            plt.setp(ax_box, xticks=xticks, xticklabels=scatter_x_tick_labels)
        else:
            plt.setp(ax_box, xticklabels=scatter_x_tick_labels)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        ax_box.set_xlabel("Challenge size", fontsize=25, labelpad=25)
        if metric == "FHD":
            metric_title = "Fractional Hamming distance"
        else:
            metric_title = "Shannon entropy"
        ax_box.set_ylabel(metric_title, fontsize=25, labelpad=35)

        for patch, color in zip(box["boxes"], box_colors):
            patch.set_facecolor(color)
        fig.savefig(fname=f"{log_folder}/{db_folder}_{metric}_box.jpg", bbox_inches='tight', pad_inches=0)
        plt.close("all")
