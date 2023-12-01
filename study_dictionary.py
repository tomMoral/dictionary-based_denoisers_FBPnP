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
    "epochs": 10,
    "avg": False,
    "rescale": False,
    "fixed_noise": True,
    "D_shared": True,
    "lr": 1e-1,
    "verbose": False
}

list_params = {
    "analysis": [],
    "synthesis": []
}

lambdas = np.logspace(-4, 0, num=10)
n_exp = 5

for lmbd in tqdm(lambdas):
    for type_unrolling in ["analysis", "synthesis"]:
        unrolled_cdl = UnrolledCDL(
            **params_model,
            lmbd=lmbd,
            type_unrolling=type_unrolling
        )
        unrolled_net, train_loss, test_loss = unrolled_cdl.fit()
        list_params[type_unrolling].append(unrolled_net.parameter.detach().cpu().numpy())

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
        row_ind, col_ind = linear_sum_assignment(C, maximize=True)
        score = C[row_ind, col_ind].mean()
    except:
        score = 0
    return score

# %%


fig, ax = plt.subplots(1, 2, figsize=(10, 10))

for t, type_unrolling in enumerate(["analysis", "synthesis"]):

    len_params = len(list_params[type_unrolling])
    C = np.zeros((len_params, len_params))

    for i in range(len_params):
        for j in range(len_params):
            D0 = list_params[type_unrolling][i]
            D1 = list_params[type_unrolling][j]
            C[i, j] = recovery_score(D0, D1)

    ax[t].matshow(C, vmin=0.3, vmax=1.)
    ax[t].set_xticks(np.arange(len(lambdas)), labels=[f"{lmbd:.0e}" for lmbd in lambdas])
    ax[t].set_yticks(np.arange(len(lambdas)), labels=[f"{lmbd:.0e}" for lmbd in lambdas])
    ax[t].set_xlabel(r"$\lambda$ training")
    ax[t].set_ylabel(r"$\lambda$ training")

    for i in range(len(lambdas)):
        for j in range(len(lambdas)):
            ax[t].text(i, j, f"{C[i, j]:.2f}", ha="center", va="center", color="w")

    ax[t].set_title(f"Similarity {type_unrolling}")

plt.tight_layout()
plt.savefig("figures/similarity_dictionaries.pdf")

# %%
