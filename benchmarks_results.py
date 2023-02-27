# %%
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.colors as mcolors
# %%
data = pd.read_csv("results/results_pnp_fixed_noise.csv")
# %%
data.head()

# %%

lmbd = data["lmbd"].unique()[1]
std_noise = data["std_noise"].unique()[0]

combinations_params = [
    (True, True, True),
    (True, True, False),
    (True, False, True),
    (True, False, False),
    (False, True, True),
    (False, True, False),
    (False, False, True),
    (False, False, False),
]

# %%

fig, axs = plt.subplots(len(combinations_params), 3, figsize=(12, 16))

dico_legend = {}
for model in data["model"].unique():
    dico_legend[model] = None

for j, current_tuple in enumerate(combinations_params):
    update_dual, D_shared, avg = current_tuple
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
plt.savefig(f"figures/pnp_benchmark_unrolled_std_noise_{std_noise}_lmbd_{lmbd}.pdf")
print(f"figures/pnp_benchmark_unrolled_std_noise_{std_noise}_lmbd_{lmbd}.pdf")
# %%
