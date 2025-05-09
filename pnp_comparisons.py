# %%
import time
import pickle

import scipy
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from skimage.metrics import peak_signal_noise_ratio

import torch
import torch.nn as nn

import deepinv

from pnp_unrolling.unrolled_cdl import UnrolledCDL
from pnp_unrolling.datasets import (
    create_imagenet_dataloader,
)


def plot_img(img, ax, title=None, ref=None):
    img = img.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    ax.imshow(img)
    ax.set_axis_off()
    if title:
        if ref is not None:
            ref = ref.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
            psnr = peak_signal_noise_ratio(img, ref)
            title = f"{title}\n({psnr:.1f})"
        ax.set_title(title)


def get_full_model(denoiser, max_iter=1000):
    old_net = denoiser["net"]
    net = UnrolledCDL(
        type_unrolling=denoiser["model"],
        **{k: v for k, v in denoiser.items() if k not in ["model", "net"]}
    ).unrolled_net
    net.parameter = old_net.parameter
    net.model = torch.nn.ModuleList([old_net.model[0]] * max_iter)
    return net


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
    "lmbd": 1e-2,
    "color": COLOR,
    "device": DEVICE,
    "dtype": torch.float,
    "optimizer": "adam",
    "path_data": DATA_PATH,
    "max_sigma_noise": STD_NOISE,
    "min_sigma_noise": STD_NOISE,
    "mini_batch_size": 20,
    "max_batch": 50,
    "epochs": 100,
    "avg": False,
    "rescale": False,
    "fixed_noise": True,
    "D_shared": True,
    "step_size_scaling": 1.8,
    "lr": 2e-4,
    "dataset": DATASET,
    "init_dual": False
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
    "DRUNet": dict(model="drunet"),
    'SD': dict(model="synthesis", **params_model_20),
    'AD': dict(model="analysis", **params_model_20),
    'SD1': dict(model="synthesis", **params_model_1),
    'AD1': dict(model="analysis", **params_model_1),
}

for n in DENOISERS:
    print(f"Loading {n}")
    DENOISERS[n]["net"] = get_denoiser(**DENOISERS[n])
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
fig, axes = plt.subplots(2, 4)
axes = axes.flatten()
plot_img(img[0], axes[0], title="Original")
plot_img(img_noise[0], axes[1], title="Noisy", ref=img[0])
for i, n in enumerate(DENOISERS):
    with torch.no_grad():
        if n != "DRUNet":
            # Get a model with the same parameters, but running to convergence
            # (i.e. 1000 iterations)
            net = get_full_model(DENOISERS[n])
            img_clean = net(img_noise)[0]
        else:
            img_clean = DENOISERS[n]["net"](img_noise, STD_NOISE)
    plot_img(img_clean[0], axes[i+2], title=n, ref=img[0])
    if i == 4:
        break

plt.tight_layout()
plt.savefig(f"denoisers_{STD_NOISE}.pdf")
plt.show()

# %%
# Add a denoiser composed of multiple iteration of a denoiser trained
# with one layer
for n in ["SD", "AD"]:
    n_rep = 20
    denoiser = DENOISERS[f"{n}1"]
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
    denoiser = DENOISERS[n]
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
# Evaluate the runtime of all denoisers
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
def pnp_deblurring(
    model,
    phy,
    x_observed,
    reg_par=0.5 * STD_NOISE,
    n_iter=50,
    net=None,
    x_truth=None,
):

    if model in ["analysis", "synthesis"]:
        model = "unrolled"
        net.set_lmbd(reg_par)

    with torch.no_grad():
        x_n = phy.A_adjoint(torch.tensor(x_observed, device=DEVICE))
        normPhi2 = phy.compute_norm(
            torch.rand_like(x_observed[0]), verbose=False
        )
    gamma = 1.0 / normPhi2
    F_psnr = deepinv.metric.PSNR()

    cvg, psnr, runtime = [1e10] * n_iter, [0] * n_iter, [0] * n_iter

    current_dual = None
    t_iter = 0
    with torch.no_grad():
        for k in tqdm(range(0, n_iter), leave=False):
            t_start = time.perf_counter()
            g_n = phy.A_adjoint((phy.A(x_n) - x_observed))
            tmp = x_n - gamma * g_n
            x_old = x_n
            if model == "unrolled":
                x_n, current_dual = net(tmp, current_dual)
            elif model == "drunet":
                x_n = net(tmp, reg_par).clip(0, 1)
            t_iter += time.perf_counter() - t_start
            cvg[k] = ((x_n - x_old) ** 2).sum().item()
            runtime[k] = t_iter
            if x_truth is not None:
                psnr[k] = F_psnr(x_n, x_truth).item()

    return dict(
        img=torch.clip(x_n, 0, 1).detach().cpu().numpy()[0],
        cvg=cvg, psnr=psnr, time=runtime
    )


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


def generate_results_pnp(pth_kernel, img, n_iter=1000, reg_par=0.1, seed=427):

    h = scipy.io.loadmat(pth_kernel)
    h = np.array(h["blur"])

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)

    phy = deepinv.physics.BlurFFT(
        filter=torch.tensor(h[None, None]), img_size=img.shape[1:],
        noise_model=deepinv.physics.GaussianNoise(
            sigma=STD_NOISE, rng=generator
        ),
        # noise_model=deepinv.physics.SaltPepperNoise(
        #     s=0.025, p=0.025, rng=generator
        # ),
        device=DEVICE
    )
    x_observed = phy(img).clip(0, 1)

    results = {
        "observation": x_observed.detach().cpu().numpy()[0],
        "truth": img.detach().cpu().numpy()[0],
        "reg_par": reg_par,
    }
    for name, denoiser in DENOISERS.items():

        results[name] = pnp_deblurring(
            denoiser["model"],
            phy,
            x_observed,
            n_iter=n_iter,
            reg_par=reg_par,
            net=denoiser["net"],
            x_truth=img,
        )
        psnr = results[name]['psnr'][-1]
        print(f"PSNR {name}: {psnr:.2f}")

    return results


# %%
print("Running experiments...")
N_EXP = 4
list_results = []
reg_pars = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
for i in range(N_EXP):
    img_noise, img = next(iter(dataloader))
    for reg_par in reg_pars:
        print("\n\n" + "-" * 80)
        print(f"Running experiment with img {i} and reg {reg_par}")
        print("-" * 80)
        results = generate_results_pnp(
            pth_kernel, img, reg_par=reg_par, seed=i
        )
        list_results.append(results)

# %%
# Save the results early to avoid loosing them.
with open("results.pkl", "wb") as f:
    pickle.dump(list_results, f)


# %%

LABELS = {
    "SD": "PNP-SD ($L_{train} = 20$)",
    "SD_only1": "PNP-SD ($L_{train/test} = 20/1$)",
    "SD1": "PNP-SD ($L_{train} = 1$)",
    "SD_repeat_20": "PNP-SD ($L_{train/test} = 1/20$)",
    "AD": "PNP-AD ($L_{train} = 20$)",
    "AD_only1": "PNP-AD ($L_{train/test} = 20/1$)",
    "AD1": "PNP-AD ($L_{train} = 1$)",
    "AD_repeat_20": "PNP-AD ($L_{train/test} = 1/20$)",
    "DRUNet": "PNP-DRUNet",
}

YLABELS = {
    "cvg": r"$||x_{k-1} - x_{k}||$",
    "psnr": "PSNR",
}

n_d = len(DENOISERS)
n_cols = 4
n_rows = n_d // n_cols + ((n_d % n_cols) != 0)

cmap = plt.get_cmap('viridis', len(reg_pars))
norm = LogNorm(min(*reg_pars), max(*reg_pars)*2)

for col in ["cvg", "psnr"]:
    fig = plt.figure(figsize=(4*n_cols, 2.8*(n_rows + 0.1)))

    gs = plt.GridSpec(
        n_rows + 1, n_cols, height_ratios=[0.1]+[1]*n_rows,
        left=0.05, right=0.99, top=0.99, bottom=0.05, hspace=0.5
    )
    ax = fig.add_subplot(gs[1, 0])
    axes = [ax] + [
        fig.add_subplot(gs[1+k // n_cols, k % n_cols], sharex=ax, sharey=ax)
        for k, name in enumerate(LABELS) if k > 0
    ]
    axes = {n: ax for n, ax in zip(LABELS.keys(), axes)}
    for k, ax in enumerate(axes.values()):
        if k >= n_rows * n_cols - n_cols:
            ax.set_xlabel("Iterations")
        if k % n_cols == 0:
            ax.set_ylabel(YLABELS[col])

    for j, reg in enumerate(reg_pars):
        med_curves = {}
        for i, results in enumerate(list_results):
            if results["reg_par"] == reg:
                for name in DENOISERS:
                    if name not in axes:
                        continue
                    t, curve = results[name]['time'], results[name][col]
                    axes[name].semilogx(curve, c=cmap(norm(reg)), alpha=0.3)
                    all_curves = med_curves.get(name, [])
                    all_curves.append([t, curve])
                    med_curves[name] = all_curves
        for k, (name, all_curves) in enumerate(med_curves.items()):
            if name not in axes:
                continue
            ax = axes[name]
            t, curve = np.median(all_curves, axis=0)
            ax.semilogx(curve, c=cmap(norm(reg)))
            if col == "cvg":
                ax.set_yscale("log")
            ax.grid(True)
            label = LABELS.get(name, name)
            ax.set_title(f"{label}\n({t[-1]:.1f}s)")

    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.set_axis_off()
    ax_legend.legend(
        [plt.Line2D([], [], c=cmap(norm(reg)), lw=2) for reg in reg_pars],
        [f"$\\lambda = {reg:.0e}$" for reg in reg_pars],
        loc='upper center', ncols=4
    )

    plt.savefig(f"final_{col}_{STD_NOISE}.pdf")

# %%
n_denoisers = len(DENOISERS)
n_rows = len(list_results)
n_cols = n_denoisers + 2
fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))

for i, results in enumerate(list_results):

    x_observed = results["observation"].transpose(1, 2, 0).clip(0, 1)
    img = results["truth"].transpose(1, 2, 0).clip(0, 1)

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
            f"PnP-{name}$\\lambda = {results["reg_par"]:.0e}$\n"
            f"PSNR = {results[name]['psnr'][-1]:0.2f} dB"
        )
        axs[i, j+2].set_axis_off()

plt.tight_layout()
plt.savefig(f"example_images_{STD_NOISE}.pdf", dpi=150)
plt.clf()
