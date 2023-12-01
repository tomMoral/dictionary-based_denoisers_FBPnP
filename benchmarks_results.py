# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import matplotlib.colors as mcolors
# %%
data = pd.read_csv("results/results_pnp_num_layers.csv")
# %%
data.head()

# %%

avg = False
std_noise = data["std_noise"].unique()[0]
lmbd = data["lmbd"].unique()[4]

combinations_params = [
    (True, True),
    (True, False),
    (False, True),
    (False, False),
]

# %%

fig, axs = plt.subplots(len(combinations_params), 3, figsize=(12, 16))

dico_legend = {}
for model in data["model"].unique():
    dico_legend[model] = None

for j, current_tuple in enumerate(combinations_params):
    update_dual, D_shared = current_tuple
    for i, metric in enumerate(["psnr", "ssim", "conv"]):
        for l, model in enumerate(data["model"].unique()):
            for image_num in data["images"].unique():
                for n_layers in data["n_layers"].unique():
                    current_data = data[
                        (data["model"] == model)
                        & (data["lmbd"] == lmbd)
                        & (data["images"] == image_num)
                        & (data["D_shared"] == D_shared)
                        & (data["update_dual"] == update_dual)
                        & (data["avg"] == avg)
                        & (data["std_noise"] == std_noise)
                        & (data["n_layers"] == n_layers)
                    ]

                    line, = axs[j, i].plot(
                        current_data["reg_par"],
                        current_data[metric],
                        color=list(mcolors.TABLEAU_COLORS.keys())[l],
                        alpha=0.2,
                        linewidth=1 + n_layers / 5
                    )
                dico_legend[model] = line

        if metric == "conv":
            metric = r"$\frac{\|x_n - x_{n+1} \|}{\|x_n \|}$"
            axs[j, i].set_yscale("log")
            axs[j, i].set_ylim([1e-15, 1e2])

        if metric == "psnr":
            axs[j, i].set_ylim([0, 35])

        if metric == "ssim":
            axs[j, i].set_ylim([0, 1.])

        axs[j, i].set_xlabel("Reg value")
        axs[j, i].set_ylabel(metric)
        axs[j, i].set_xscale("log")
        axs[j, i].set_title(
            f"Warm restart: {update_dual}\n"
            f"Shared params: {D_shared}\n"
            f"Rescaling: {avg}"
        )

axs[j, i].legend(
    list(dico_legend.values()),
    list(dico_legend.keys())
)
plt.tight_layout()
plt.savefig(f"figures/pnp_benchmark_unrolled_num_layers_lmbd_{lmbd}_std_noise_{std_noise}_rescaling_{avg}.pdf")
print(f"figures/pnp_benchmark_unrolled_num_layers_lmbd_{lmbd}_std_noise_{std_noise}_rescaling_{avg}.pdf")
# %%


