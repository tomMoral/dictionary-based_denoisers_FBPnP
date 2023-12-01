# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import matplotlib.colors as mcolors

# %%
data = pd.read_csv("results/results_pnp_study_lambda.csv")
# %%
data.head()

# %%

lmbd = data["lmbd"].unique()[2]
std_noise = data["std_noise"].unique()[1]
avg = False

# %%

update_dual = True
D_shared = True


# %%

fig, axs = plt.subplots(1, 3, figsize=(8, 4))

dico_legend = {}
for model in data["model"].unique():
    dico_legend[model] = None

for i, metric in enumerate(["psnr", "ssim", "conv"]):
    for l, model in enumerate(data["model"].unique()):
        for image_num in data["images"].unique():
            current_data = data[
                (data["model"] == model)
                & (data["lmbd"] == lmbd)
                & (data["images"] == image_num)
                & (data["D_shared"] == D_shared)
                & (data["update_dual"] == update_dual)
                & (data["avg"] == avg)
                & (data["std_noise"] == std_noise)
            ]

            line, = axs[i].plot(
                current_data["reg_par"],
                current_data[metric],
                color=list(mcolors.TABLEAU_COLORS.keys())[l],
                alpha=0.5
            )
            dico_legend[model] = line

    if metric == "conv":
        metric = r"$\frac{\|x_n - x_{n+1} \|}{\|x_n \|}$"
        axs[i].set_yscale("log")
        axs[i].set_ylim([1e-15, 1e5])

    if metric == "psnr":
        axs[i].set_ylim([0, 35])

    if metric == "ssim":
        axs[i].set_ylim([0, 1.])

    axs[i].set_xlabel("Reg value")
    axs[i].set_ylabel(metric)
    axs[i].set_xscale("log")
    axs[i].set_title(
        f"Warm restart: {update_dual}\n"
        f"Shared params: {D_shared}"
    )

axs[i].legend(
    list(dico_legend.values()),
    list(dico_legend.keys())
)
plt.tight_layout()
plt.savefig(f"figures/pnp_benchmark_imagenet_unrolled_std_noise_{std_noise}_lmbd_{lmbd}.pdf")
print(f"figures/pnp_benchmark_imagenet_unrolled_std_noise_{std_noise}_lmbd_{lmbd}.pdf")

# %%