# %%
import time
import pickle

import scipy
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

import torch
import torch.nn as nn

import deepinv

from pnp_unrolling.unrolled_cdl import UnrolledCDL
from utils.measurement_tools import get_operators
from utils.tools import op_norm2
from pnp_unrolling.datasets import (
    create_imagenet_dataloader,
)


def plot_img(img, ax, title=None):
    img = img.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    ax.imshow(img)
    ax.set_axis_off()
    if title:
        ax.set_title(title)


DATASET = "bsd"
COLOR = True
DEVICE = "cuda:3"
STD_NOISE = 0.05

# Here the dataset is "BSD" but we use the same create_imagenet_dataloader
# function which need to set dataset="imagenet"
create_dataloader = create_imagenet_dataloader
DATA_PATH = "./BSDS500/BSDS500/data/images"
DATASET = "imagenet"


# %%

params_model_1 = {
    "n_layers": 1,
    "n_components": 50,
    "kernel_size": 5,
    "lmbd": 1e-4,
    "color": COLOR,
    "device": DEVICE,
    "dtype": torch.float,
    "optimizer": "adam",
    "path_data": DATA_PATH,
    "max_sigma_noise": STD_NOISE,
    "min_sigma_noise": STD_NOISE,
    "mini_batch_size": 1,
    "max_batch": 10,
    "epochs": 50,
    "avg": False,
    "rescale": False,
    "fixed_noise": True,
    "D_shared": True,
    "step_size_scaling": 1.8,
    "lr": 1e-3,
    "dataset": DATASET,
}
params_model_20 = {k: v for k, v in params_model_1.items()}
params_model_20["n_layers"] = 20


def get_denoiser(model, **kwargs):

    if model == "drunet":
        nc = 3 if COLOR else 1
        net = deepinv.models.DRUNet(
            in_channels=nc,
            out_channels=nc,
            nc=[64, 128, 256, 512],
            nb=4,
            act_mode="R",
            downsample_mode="strideconv",
            upsample_mode="convtranspose",
            pretrained="download",
        )
        net = nn.DataParallel(net, device_ids=[int(DEVICE[-1])])
    elif model in ["analysis", "synthesis"]:
        unrolled_cdl = UnrolledCDL(type_unrolling=model, **kwargs)
        # Training unrolled networks
        net, *_ = unrolled_cdl.fit()
    else:
        raise ValueError(
            f"Requested denoiser {model} which is not available."
        )
    return net


DENOISERS = {
    'SD': dict(model="synthesis", **params_model_20),
    'AD': dict(model="analysis", **params_model_20),
    'SD1': dict(model="synthesis", **params_model_1),
    'AD1': dict(model="analysis", **params_model_1),
    "DRUNet": dict(model="drunet"),
}

for n in DENOISERS:
    print(f"Loading {n}")
    DENOISERS[n]["net"] = get_denoiser(**DENOISERS[n])

# Add a denoiser composed of multiple iteration of a denoiser trained
# with one layer
for n in ["SD", "AD"]:
    denoiser = DENOISERS[f"{n}1"]
    for n_rep in [20, 50]:
        if n_rep == 50 and n == "AD":
            continue
        old_net = denoiser["net"]
        net = UnrolledCDL(
            type_unrolling=denoiser["model"],
            **{k: v for k, v in denoiser.items() if k not in ["model", "net"]}
        ).unrolled_net
        assert len(net.model) == 1
        net.parameter = old_net.parameter
        net.model = torch.nn.ModuleList([old_net.model[0]] * n_rep)
        DENOISERS[f"{n}_repeat_{n_rep}"] = dict(
            net=net, model=denoiser["model"]
        )
    # Restrict the model trained with 20 layers to only one layer
    denoiser = DENOISERS[f"{n}"]
    old_net = denoiser["net"]
    net = UnrolledCDL(
        type_unrolling=denoiser["model"],
        **{k: v for k, v in denoiser.items() if k not in ["model", "net"]}
    ).unrolled_net
    assert len(net.model) == 20
    # Replace the model with only the first layer of the trained model
    net.parameter = old_net.parameter
    net.model = torch.nn.ModuleList([old_net.model[0]])
    DENOISERS[f"{n}_only1"] = dict(net=net, model=denoiser["model"])
print("All denoisers loaded")


# %%
print("Evaluating all denoisers...")
dataloader = create_dataloader(
    DATA_PATH, STD_NOISE, STD_NOISE, device=DEVICE, dtype=torch.float,
    mini_batch_size=1, train=True, random_state=42, color=COLOR,
    download=True, fixed_noise=True
)
img_noise, img = next(iter(dataloader))

n_d = len(DENOISERS)
fig, axes = plt.subplots(3, 4)
axes = axes.flatten()
plot_img(img[0], axes[0], title="Original")
plot_img(img_noise[0], axes[1], title="Noisy")
for i, n in enumerate(DENOISERS):
    with torch.no_grad():
        if n != "DRUNet":
            img_clean = DENOISERS[n]["net"](img_noise)[0]
        else:
            img_clean = DENOISERS[n]["net"](img_noise, STD_NOISE)
    plot_img(img_clean[0], axes[i+2], title=n)
    if i == 11:
        break

plt.tight_layout()
plt.savefig("denoisers.pdf")
plt.show()

# import IPython; IPython.embed(colors='neutral')

n_imgs = 100
inference_runtime = []
for k, (img_noise, _) in enumerate(dataloader):
    if k == n_imgs:
        print("\rDone computing runtime")
        break
    print(f"\rImage {k}/{n_imgs}", end="", flush=True)
    for i, n in enumerate(DENOISERS):
        with torch.no_grad():
            t_start = time.perf_counter()
            if n != "DRUNet":
                img_clean = DENOISERS[n]["net"](img_noise)[0]
            else:
                img_clean = DENOISERS[n]["net"](img_noise, STD_NOISE)
            t_end = time.perf_counter()
            inference_runtime.append({
                'id_img': k,
                'denoiser': n,
                'runtime': t_end - t_start
            })
inference = pd.DataFrame(inference_runtime)
inference.to_csv("inference_runtime.csv")
print(inference.groupby("denoiser")["runtime"].agg(['mean', 'std']).to_latex())


# %%


def apply_model(model, x, dual, reg_par, net=None, update_dual=False):

    if model == "unrolled":
        net.set_lmbd(reg_par)
        x_torch = torch.tensor(x, device=DEVICE, dtype=torch.float)[None, :]
        if dual is not None:
            dual = torch.tensor(dual, device=DEVICE, dtype=torch.float)
        with torch.no_grad():
            xnet, new_dual = net(x_torch, dual)
        if not update_dual:
            return xnet.detach().cpu().numpy()[0], None
        else:
            return (
                xnet.detach().cpu().numpy()[0],
                new_dual.detach().cpu().numpy()
            )
    elif model == "identity":
        return x, None
    elif model == "drunet":
        x_torch = torch.tensor(x, device=DEVICE, dtype=torch.float)[None, :]
        with torch.no_grad():
            xnet = net(x_torch, reg_par)
        return np.clip(xnet.detach().cpu().numpy()[0], 0, 1), None


def Phi_channels(x, Phi):

    new_x = np.concatenate(
        [Phi(x[i])[None, :] for i in range(x.shape[0])],
        axis=0
    )

    return new_x


def pnp_deblurring(
    model,
    pth_kernel,
    x_observed,
    reg_par=0.5 * STD_NOISE,
    n_iter=50,
    net=None,
    update_dual=False,
    x_truth=None,
):

    if model in ["analysis", "synthesis"]:
        model = "unrolled"

    # define operators
    Phi, Phit = get_operators(type_op="deconvolution", pth_kernel=pth_kernel)

    normPhi2 = op_norm2(Phi, Phit, x_observed.shape)
    # gamma = 1.99 / normPhi2
    gamma = 1.0 / normPhi2

    x_n = Phi_channels(x_observed, Phit)

    cvg, psnr, runtime = [1e10] * n_iter, [0] * n_iter, [0] * n_iter

    current_dual = None
    t_iter = 0
    for k in tqdm(range(0, n_iter)):
        t_start = time.perf_counter()
        g_n = Phi_channels((Phi_channels(x_n, Phi) - x_observed), Phit)
        tmp = x_n - gamma * g_n
        x_old = x_n.copy()
        x_n, current_dual = apply_model(
            model, tmp, current_dual, reg_par, net, update_dual
        )
        t_iter += time.perf_counter() - t_start
        cvg[k] = np.sum((x_n - x_old) ** 2)
        runtime[k] = t_iter
        if x_truth is not None:
            psnr[k] = peak_signal_noise_ratio(x_n, x_truth)

    return dict(img=np.clip(x_n, 0, 1), cvg=cvg, psnr=psnr, time=runtime)


# %%

dataloader = create_dataloader(
    DATA_PATH,
    min_sigma_noise=STD_NOISE,
    max_sigma_noise=STD_NOISE,
    device=DEVICE,
    dtype=torch.float,
    mini_batch_size=1,
    train=False,
    color=COLOR,
    fixed_noise=True,
    crop=False,
)

# %%

pth_kernel = "blur_models/blur_3.mat"


def generate_results_pnp(pth_kernel, img, n_iter=1000, reg_par=0.1):

    h = scipy.io.loadmat(pth_kernel)
    h = np.array(h["blur"])

    Phi, Phit = get_operators(type_op="deconvolution", pth_kernel=pth_kernel)
    x_blurred = Phi_channels(img, Phi)
    nc, nxb, nyb = x_blurred.shape
    x_observed = x_blurred + STD_NOISE * np.random.rand(nc, nxb, nyb)

    results = {
        "observation": x_observed,
        "truth": img,
        "reg_par": reg_par,
    }
    for name, denoiser in DENOISERS.items():

        results[name] = pnp_deblurring(
            denoiser["model"],
            pth_kernel,
            x_observed,
            n_iter=n_iter,
            reg_par=reg_par,
            update_dual=True,
            net=denoiser["net"],
            x_truth=img,
        )

    return results


# %%
print("Running experiments...")
N_EXP = 4
list_results = []
reg_pars = [1e-5, 1e-3, 1e-2, 1e-1]
for _ in range(N_EXP):
    img_noise, img = next(iter(dataloader))
    img_noise, img = img_noise.cpu().numpy()[0], img.cpu().numpy()[0]
    for reg_par in reg_pars:
        results = generate_results_pnp(pth_kernel, img, reg_par=reg_par)
        list_results.append(results)

# %%
# Save the results early to avoid loosing them.
with open("results.pkl", "wb") as f:
    pickle.dump(list_results, f)

# %%
n_reg = len(reg_pars)
n_denoisers = len(DENOISERS)
n_rows = len(list_results)
n_cols = n_denoisers + 2
fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4*n_rows))

for i, results in enumerate(list_results):

    x_observed = results["observation"].transpose(1, 2, 0).clip(0, 1)
    img = results["truth"].transpose(1, 2, 0).clip(0, 1)

    exponent = int(np.log10(results["reg_par"]))

    cmap = None

    axs[i, 0].set_axis_off()
    axs[i, 0].imshow(x_observed, cmap=cmap)
    axs[i, 0].set_title(r"Observation $y$")
    axs[i, 1].imshow(img, cmap=cmap)
    axs[i, 1].set_title(r"Ground truth $\overline{x}$")
    axs[i, 1].set_axis_off()
    for j, name in enumerate(DENOISERS):
        res = results[name]
        img_result = res['img'].transpose(1, 2, 0).clip(0, 1)
        axs[i, j+2].imshow(img_result, cmap=cmap)
        axs[i, j+2].set_title(
            f"PnP-{name}$\\lambda = 10^{{{exponent}}}$\n"
            f"PSNR = {results[name]['psnr'][-1]:0.2f} dB"
        )
        axs[i, j+2].set_axis_off()

plt.tight_layout()
plt.savefig("example_images.pdf", dpi=150)
plt.clf()

# %%

fig, axs = plt.subplots(n_d, n_reg, sharey=True, figsize=(4*n_reg, 2.2*n_d))
for j in range(n_reg):
    # Calculate the exponent for 10^exponent
    exponent = int(np.log10(reg_pars[j]))

    for i, results in enumerate(list_results):
        if results["reg_par"] == reg_pars[j]:
            for k, name in enumerate(DENOISERS):
                axs[k, j].loglog(results[name]['cvg'], c="green")
                axs[k, j].set_title(
                    rf"PnP-{name} $\lambda = 10^{{{exponent}}}$"
                )  # Scientific notation for reg_pars[j]
                axs[k, j].grid(True)

# Set the ylable on the first column of each row
for axes in axs:
    axes[0].set_ylabel(r"$||x_{k-1} - x_{k}||$")

plt.tight_layout()
plt.savefig("convergence.pdf")


# %%

fig, axs = plt.subplots(
    n_d, n_reg, sharey=True, figsize=(4*n_reg, 2.2*n_d)
)
fig_time, axs_time = plt.subplots(
    n_d, n_reg, sharey=True, figsize=(4*n_reg, 2.2*n_d)
)

for j, reg in enumerate(reg_pars):
    exponent = int(np.log10(reg))  # Calculate the exponent for 10^exponent

    for i, results in enumerate(list_results):
        if results["reg_par"] == reg:
            for k, name in enumerate(DENOISERS):
                res = results[name]
                axs[k, j].semilogx(res['psnr'], c="green")
                axs_time[k, j].semilogx(res['time'], res['psnr'], c="green")
                axs[k, j].set_title(
                    rf"{name} $\lambda = 10^{{{exponent}}}$"
                )  # Scientific notation for reg
                axs[k, j].grid(True)

# Set the ylable on the first column of each row
for axes in axs:
    axes[0].set_ylabel("PSNR")

fig.tight_layout()
fig.savefig("psnr.pdf")
fig_time.savefig("psnr_time.pdf")


# %%
