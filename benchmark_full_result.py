# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%

data = pd.read_csv("results/results_benchmark_full.csv")
# %%
data.shape
# %%
data.head()
# %%


fig, axs = plt.subplots(3, 5, figsize=(10, 8), sharey=True, sharex=True)

for i, model in enumerate(data["model"].unique()):
    for j, reg_par in enumerate(data["reg_par"].unique()):

        current_data = data.query(f"model == '{model}' & reg_par == {reg_par}")

        for image_num in current_data["image_num"].unique():

            array_string = current_data.query(f"image_num == {image_num}")["convergence"].values[0]

            array_string = array_string.replace("[", "")
            array_string = array_string.replace("]", "")
            array_string = array_string.replace("\n", "")
            array_string = array_string.replace(",", "")
            new_array = []
            for elt in array_string.split(" "):
                try:
                    new_array.append(float(elt))
                except ValueError:
                    pass

            axs[i, j].plot(new_array, color="g", alpha=0.7)

        axs[i, j].set_title(f"{model} {reg_par}")
        if i == 2:
            axs[i, j].set_xlabel("N iterations")
        if j == 0:
            axs[i, j].set_ylabel(r"$||X_N - X_{N-1}*||$")
        axs[i, j].set_yscale("log")

plt.tight_layout()
plt.savefig("benchmark_full_convergence.pdf")

# %%
