# %%
import matplotlib.pyplot as plt
import torch
import numpy as np
import bm3d
import scipy
import torch.nn as nn

from pnp_unrolling.unrolled_cdl import UnrolledCDL
from external.network_unet import UNetRes
from external.utils_dpir import test_mode as test_mode_dpir
from pnp_unrolling.datasets import create_imagewoof_dataloader
from utils.wavelet_utils import wavelet_op
from utils.measurement_tools import get_operators
from utils.tools import op_norm2
from tqdm import tqdm


PATH_DATA = "/storage/store2/work/bmalezie/imagewoof/"
COLOR = True
DEVICE = "cuda:3"
STD_NOISE = 0.05
reg = 0.3


def load_nets(reg=0.1):
    print("Loading drunet...")
    net = UNetRes(
        in_nc=1 + 1,
        out_nc=1,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode='R',
        downsample_mode="strideconv",
        upsample_mode="convtranspose"
    )
    net = nn.DataParallel(net, device_ids=[int(DEVICE[-1])])

    filename = 'checkpoint/drunet_gray.pth'
    checkpoint = torch.load(filename,
                            map_location=lambda storage,
                            loc: storage)
    try:
        net.module.load_state_dict(checkpoint, strict=True)
    except:
        net.module.load_state_dict(checkpoint.module.state_dict(), strict=True)

    params_model = {
        "n_layers": 20,
        "n_components": 50,
        "kernel_size": 5,
        "lmbd": reg * STD_NOISE,
        "color": COLOR,
        "device": DEVICE,
        "dtype": torch.float,
        "D_shared": False,
        "optimizer": "adam",
        "path_data": PATH_DATA,
        "max_sigma_noise": 0.1,
        "min_sigma_noise": 0,
        "mini_batch_size": 1,
        "max_batch": 10,
        "epochs": 50,
        "avg": True,
        "rescale": False,
        "pseudo_gd": False,
        "fixed_noise": False
    }

    print("Loading unrolled analysis")
    unrolled_cdl_analysis = UnrolledCDL(**params_model,
                                        type_unrolling="analysis")
    net_analysis, _, _ = unrolled_cdl_analysis.fit()

    print("Loading unrolled synthesis")
    unrolled_cdl_synthesis = UnrolledCDL(**params_model,
                                         type_unrolling="synthesis")
    net_synthesis, _, _ = unrolled_cdl_synthesis.fit()

    return net, net_analysis, net_synthesis



# %%

DRUNET, NET_ANALYSIS, NET_SYNTHESIS = load_nets()

# %%

def prox_L1(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)


def apply_model(model, x, reg_par, net=None):

    if model == "bm3d":
        return bm3d.bm3d(x, reg_par)
    elif model == "wavelet":
        wave_choice = 'db8'
        Psi, Psit = wavelet_op(x, wav=wave_choice, level=4)
        return Psit(prox_L1(Psi(x), reg_par))
    elif model == "drunet":
        x_torch = torch.tensor(
            x,
            device=DEVICE,
            dtype=torch.float
        )[None, None, :]
        ths_map = torch.tensor(
            [reg_par],
            device=DEVICE,
            dtype=torch.float
        ).repeat(1, 1, x_torch.shape[2], x_torch.shape[3])
        img_in = torch.cat((x_torch, ths_map), dim=1)
        xnet = test_mode_dpir(net, img_in, refield=64, mode=5)
        return xnet.detach().cpu().numpy()[0, 0]
    elif model == "unrolled":
        net.set_lmbd(reg_par)
        x_torch = torch.tensor(
            x,
            device=DEVICE,
            dtype=torch.float
        )[None, None, :]
        xnet = net(x_torch)
        return xnet.detach().cpu().numpy()[0, 0]
    elif model == "identity":
        return x


def pnp_deblurring(model, pth_kernel, x_observed, reg_par=0.5 * STD_NOISE,
                   n_iter=50, net=None):

    if model in ["analysis", "synthesis"]:
        model = "unrolled"

    # define operators
    Phi, Phit = get_operators(type_op='deconvolution', pth_kernel=pth_kernel)

    normPhi2 = op_norm2(Phi, Phit, x_observed.shape)
    gamma = 1.99 / normPhi2

    x_n = Phit(x_observed)
    table_energy = 1e10 * np.ones(n_iter)

    for k in range(0, n_iter):
        g_n = Phit((Phi(x_n) - x_observed))
        tmp = x_n - gamma * g_n
        x_old = x_n.copy()
        x_n = apply_model(model, tmp, reg_par, net)
        table_energy[k] = np.sum((x_n - x_old)**2) / np.sum(x_old**2)

    return np.clip(x_n, 0, 1), table_energy


# %%

dataloader = create_imagewoof_dataloader(
    PATH_DATA,
    min_sigma_noise=STD_NOISE,
    max_sigma_noise=STD_NOISE,
    device=DEVICE,
    dtype=torch.float,
    mini_batch_size=1,
    train=False,
    color=False,
    fixed_noise=True
)

img_noise, img = next(iter(dataloader))
img_noise, img = img_noise.cpu().numpy()[0, 0], img.cpu().numpy()[0, 0]

pth_kernel = 'blur_models/blur_3.mat'
h = scipy.io.loadmat(pth_kernel)
h = np.array(h['blur'])

Phi, Phit = get_operators(type_op='deconvolution', pth_kernel=pth_kernel)
x_blurred = Phi(img)
nxb, nyb = x_blurred.shape
x_observed = x_blurred + STD_NOISE * np.random.rand(nxb, nyb)

_, iterates_synthesis = pnp_deblurring("synthesis", pth_kernel, x_observed, reg_par=0.3, n_iter=1000, net=NET_SYNTHESIS)
_, iterates_analysis = pnp_deblurring("analysis", pth_kernel, x_observed, reg_par=0.0005, n_iter=1000, net=NET_ANALYSIS)
_, iterates_drunet = pnp_deblurring("drunet", pth_kernel, x_observed, reg_par=0.02, n_iter=1000, net=DRUNET)
_, iterates_wavelet = pnp_deblurring("wavelet", pth_kernel, x_observed, reg_par=0.01, n_iter=1000)


# %%
fig = plt.figure(figsize=(4, 3))

plt.plot(iterates_synthesis, label="Synthesis")
plt.plot(iterates_analysis, label="Analysis")
plt.plot(iterates_drunet, label="Drunet")
plt.plot(iterates_wavelet, label="Wavelet")
plt.xlabel("PnP iterations")
plt.ylabel(r"$\frac{\|x_n - x_{n+1} \|}{\|x_n \|}$")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("comparison_convergence_pnp.png")


# %%
