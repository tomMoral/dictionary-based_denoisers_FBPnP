# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import matplotlib.colors as mcolors

# %%
data = pd.read_csv("results/results_pnp.csv")
# %%
data.head()

# %%



noise_value = data["std_noise"].unique()[0]

combinations_params = [(True, False), (True, True), (False, True), (False, False)]


# %%

fig, axs = plt.subplots(len(combinations_params), 3, figsize=(15, 12))

dico_legend = {}
for model in data["model"].unique():
    dico_legend[model] = None

for j, current_tuple in enumerate(combinations_params):
    update_dual, shared = current_tuple
    for i, metric in enumerate(["psnr", "ssim", "conv"]):
        for l, model in enumerate(data["model"].unique()):
            for image_num in data["images"].unique():
                current_data = data[
                    (data["model"] == model)
                    & (data["std_noise"] == noise_value)
                    & (data["images"] == image_num)
                    & (data["shared"] == shared)
                    & (data["update_dual"] == update_dual)
                ]

                line, = axs[j, i].plot(
                    current_data["reg_par"],
                    current_data[metric],
                    color=list(mcolors.TABLEAU_COLORS.keys())[l],
                    alpha=0.5
                )
                dico_legend[model] = line

        if metric == "conv":
            metric = r"$\frac{\|x_n - x_{n+1} \|}{\|x_n \|}$"
            axs[j, i].set_yscale("log")
            axs[j, i].set_ylim([1e-5, 1.])

        if metric == "psnr":
            axs[j, i].set_ylim([0, 35])

        if metric == "ssim":
            axs[j, i].set_ylim([0, 1.])

        axs[j, i].set_xlabel("Reg value")
        axs[j, i].set_ylabel(metric)
        axs[j, i].set_xscale("log")
        axs[j, i].set_title(f"Warm restart: {update_dual}, Shared params {shared}")

axs[j, i].legend(
    list(dico_legend.values()),
    list(dico_legend.keys())
)
plt.tight_layout()
plt.savefig(f"figures/pnp_benchmark_unrolled_general_{noise_value}.pdf")
# %%
