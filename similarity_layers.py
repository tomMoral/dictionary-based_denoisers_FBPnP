# %%
import matplotlib.pyplot as plt
import torch
import numpy as np

from scipy.signal import correlate2d
from scipy.optimize import linear_sum_assignment

from pnp_unrolling.unrolled_cdl import UnrolledCDL
from tqdm import tqdm

PATH_DATA = "/storage/store2/work/bmalezie/imagewoof/"
COLOR = False
DEVICE = "cuda:1"
reg = 0.5
STD_NOISE = 0.05

# %%

params_model = {
    "n_layers": 20,
    "n_components": 50,
    "kernel_size": 5,
    "color": COLOR,
    "device": DEVICE,
    "dtype": torch.float,
    "optimizer": "adam",
    "path_data": PATH_DATA,
    "max_sigma_noise": STD_NOISE,
    "min_sigma_noise": STD_NOISE,
    "mini_batch_size": 1,
    "max_batch": 10,
    "epochs": 30,
    "avg": False,
    "rescale": False,
    "fixed_noise": True,
    "D_shared": False,
    "lr": 1e-1,
    "verbose": False
}

list_params = {
    "analysis": [],
    "synthesis": []
}

lmbd = 1e-1
n_exp = 5

for type_unrolling in ["analysis", "synthesis"]:
    unrolled_cdl = UnrolledCDL(
        **params_model,
        lmbd=lmbd,
        type_unrolling=type_unrolling
    )
    unrolled_net, train_loss, test_loss = unrolled_cdl.fit()
    for layer in unrolled_net.model:
        list_params[type_unrolling].append(layer.parameter.detach().cpu().numpy())

# %%
def cost_matrix(D, Dref):
    C = np.zeros((D.shape[0], Dref.shape[0]))
    for i in range(D.shape[0]):
        for j in range(Dref.shape[0]):
            C[i, j] = np.abs(correlate2d(
                D[i, 0] / np.linalg.norm(D[i, 0]),
                Dref[j, 0] / np.linalg.norm(Dref[j, 0])
            )).max()
    return C


def recovery_score(D, Dref):
    """
    Comparison between a learnt prior and the truth
    """
    try:
        C = cost_matrix(D, Dref)
        # row_ind, col_ind = linear_sum_assignment(C, maximize=True)
        row_ind, col_ind = np.arange(D.shape[0]), np.arange(D.shape[1])
        score = C[row_ind, col_ind].mean()
    except:
        score = 0
    return score

# %%


fig, ax = plt.subplots(1, 2, figsize=(15, 15))

for t, type_unrolling in enumerate(["analysis", "synthesis"]):

    len_params = len(list_params[type_unrolling])
    C = np.zeros((len_params, len_params))

    for i in range(len_params):
        for j in range(len_params):
            D0 = list_params[type_unrolling][i]
            D1 = list_params[type_unrolling][j]
            C[i, j] = recovery_score(D0, D1)

    ax[t].matshow(C, vmin=0.4, vmax=1)
    ax[t].set_xlabel("Layer")
    ax[t].set_ylabel("Layer")

    for i in range(len_params):
        for j in range(len_params):
            ax[t].text(i, j, f"{C[i, j]:.2f}", ha="center", va="center", color="w")

    ax[t].set_title(f"Similarity {type_unrolling}")

plt.tight_layout()
plt.savefig(f"figures/similarity_layers_lmbd_{lmbd}.pdf")

# %%
