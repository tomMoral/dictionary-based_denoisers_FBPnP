# %%
import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy
import torch.nn as nn
import itertools
import pandas as pd

from tqdm import tqdm
from pnp_unrolling.unrolled_cdl import UnrolledCDL
from utils.measurement_tools import get_operators
from utils.tools import op_norm2
from external.utils_dpir import test_mode as test_mode_dpir
from external.network_unet import UNetRes
from pnp_unrolling.datasets import (create_imagewoof_dataloader,
                                    create_imagenet_dataloader)
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

DATASET = "bsd"
COLOR = True
DEVICE = "cuda:0"
STD_NOISE = 0.05
N_EXP = 5

if DATASET == "imagenet":
    create_dataloader = create_imagenet_dataloader
    PATH_DATA = "/data/parietal/store2/data/ImageNet"
elif DATASET == "bsd":
    create_dataloader = create_imagenet_dataloader
    PATH_DATA = "/storage/store2/work/bmalezie/BSR/BSDS500/data/images"
    DATASET = "imagenet"
elif DATASET == "imagewoof":
    create_dataloader = create_imagewoof_dataloader
    PATH_DATA = "/storage/store2/work/bmalezie/imagewoof/"

# %%

params_model = {
    "n_layers": 20,
    "n_components": 50,
    "kernel_size": 5,
    "lmbd": 1e-4,
    "color": COLOR,
    "device": DEVICE,
    "dtype": torch.float,
    "optimizer": "adam",
    "path_data": PATH_DATA,
    "max_sigma_noise": STD_NOISE,
    "min_sigma_noise": STD_NOISE,
    "mini_batch_size": 1,
    "max_batch": 10,
    "epochs": 50,
    "avg": False,
    "rescale": False,
    "fixed_noise": True,
    "D_shared": True,
    # "type_unrolling": "analysis",
    "lr": 1e-3,
    "dataset": DATASET
}


def load_nets(params_unrolled, color):

    if color:
        nc = 3
    else:
        nc = 1

    print("Loading drunet...")
    net = UNetRes(
        in_nc=nc + 1,
        out_nc=nc,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode='R',
        downsample_mode="strideconv",
        upsample_mode="convtranspose"
    )
    net = nn.DataParallel(net, device_ids=[int(DEVICE[-1])])

    filename = f'checkpoint/drunet_{"color" if color else "gray"}.pth'
    checkpoint = torch.load(filename,
                            map_location=lambda storage,
                            loc: storage)
    try:
        net.module.load_state_dict(checkpoint, strict=True)
    except:
        net.module.load_state_dict(checkpoint.module.state_dict(),
                                   strict=True)

    print("Loading unrolled")

    unrolled_cdl_analysis = UnrolledCDL(
        **params_unrolled,
        type_unrolling="analysis"
    )

    unrolled_cdl_synthesis = UnrolledCDL(
        **params_unrolled,
        type_unrolling="synthesis"
    )

    print("Done")

    return net, unrolled_cdl_analysis, unrolled_cdl_synthesis


drunet, unrolled_analysis, unrolled_synthesis = load_nets(params_model, COLOR)

unrolled_net_synthesis, _, _ = unrolled_synthesis.fit()
unrolled_net_analysis, _, _ = unrolled_analysis.fit()

def apply_model(model, x, dual, reg_par, net=None, update_dual=False):

    if model == "unrolled":
        net.set_lmbd(reg_par)
        x_torch = torch.tensor(
            x,
            device=DEVICE,
            dtype=torch.float
        )[None, :]
        if dual is not None:
            dual = torch.tensor(
                dual,
                device=DEVICE,
                dtype=torch.float
            )
        xnet, new_dual = net(x_torch, dual)
        if not update_dual:
            return xnet.detach().cpu().numpy()[0], None
        else:
            return xnet.detach().cpu().numpy()[0], new_dual.detach().cpu().numpy()
    elif model == "identity":
        return x, None
    elif model == "drunet":
        x_torch = torch.tensor(
            x,
            device=DEVICE,
            dtype=torch.float
        )[None, :]
        ths_map = torch.tensor(
            [reg_par],
            device=DEVICE,
            dtype=torch.float
        ).repeat(1, 1, x_torch.shape[2], x_torch.shape[3])
        img_in = torch.cat((x_torch, ths_map), dim=1)
        xnet = test_mode_dpir(net, img_in, refield=64, mode=5)
        return np.clip(xnet.detach().cpu().numpy()[0], 0, 1), None


def Phi_channels(x, Phi):

    new_x = np.concatenate(
        [Phi(x[i])[None, :] for i in range(x.shape[0])],
        axis=0
    )

    return new_x


def pnp_deblurring(model, pth_kernel, x_observed, reg_par=0.5 * STD_NOISE,
                   n_iter=50, net=None, update_dual=False, x_truth=None):

    if model in ["analysis", "synthesis"]:
        model = "unrolled"

    # define operators
    Phi, Phit = get_operators(type_op='deconvolution', pth_kernel=pth_kernel)

    normPhi2 = op_norm2(Phi, Phit, x_observed.shape)
    # gamma = 1.99 / normPhi2
    gamma = 1. / normPhi2

    x_n = Phi_channels(x_observed, Phit)

    table_energy = 1e10 * np.ones(n_iter)
    table_psnr = np.zeros(n_iter)
    current_dual = None

    for k in tqdm(range(0, n_iter)):
        g_n = Phi_channels((Phi_channels(x_n, Phi) - x_observed), Phit)
        tmp = x_n - gamma * g_n
        x_old = x_n.copy()
        x_n, current_dual = apply_model(model, tmp, current_dual,
                                        reg_par, net, update_dual)
        table_energy[k] = np.sum((x_n - x_old)**2)
        if x_truth is not None:
            table_psnr[k] = peak_signal_noise_ratio(x_n, x_truth)

    return np.clip(x_n, 0, 1), table_energy, table_psnr

# %%

dataloader = create_dataloader(
    PATH_DATA,
    min_sigma_noise=STD_NOISE,
    max_sigma_noise=STD_NOISE,
    device=DEVICE,
    dtype=torch.float,
    mini_batch_size=1,
    train=False,
    color=COLOR,
    fixed_noise=True,
    crop=False
)


# %%

def generate_results_pnp(model, pth_kernel, image_num, n_iter=1000, reg_par=1e-2):

    img = IMAGES[image_num]

    h = scipy.io.loadmat(pth_kernel)
    h = np.array(h['blur'])

    Phi, Phit = get_operators(
        type_op='deconvolution',
        pth_kernel=pth_kernel
    )
    x_blurred = Phi_channels(img, Phi)
    nc, nxb, nyb = x_blurred.shape
    x_observed = x_blurred + STD_NOISE * np.random.rand(nc, nxb, nyb)

    if model == "synthesis":
        net = unrolled_net_synthesis
    elif model == "analysis":
        net = unrolled_net_analysis
    else:
        net = drunet

    _, convergence, psnr = pnp_deblurring(
        model,
        pth_kernel,
        x_observed,
        n_iter=n_iter,
        reg_par=reg_par,
        update_dual=True,
        net=net,
        x_truth=img
    )

    results = {
        "convergence": convergence,
        "psnr": psnr
    }

    return results


# %%

IMAGES = []
for _ in range(N_EXP):
    img_noise, img = next(iter(dataloader))
    img_noise, img = img_noise.cpu().numpy()[0], img.cpu().numpy()[0]
    IMAGES.append(img)

hyperparams = {
    "image_num": np.arange(0, N_EXP, 1, dtype=int),
    "pth_kernel": ['blur_models/blur_3.mat'],
    "reg_par": np.logspace(-5, -1, num=5),
    "model": ["drunet", "analysis", "synthesis"],
    "n_iter": [1000],
}

keys, values = zip(*hyperparams.items())
permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]


dico_results = {}

for params in tqdm(permuts_params):
    try:

        # Run test
        results = generate_results_pnp(**params)

        # Storing results
        for key in params.keys():
            if key not in dico_results:
                dico_results[key] = [params[key]]
            else:
                dico_results[key].append(params[key])

        for key in results.keys():
            if key not in dico_results:
                dico_results[key] = [results[key]]
            else:
                dico_results[key].append(results[key])

    except (KeyboardInterrupt, SystemExit):
        raise

results = pd.DataFrame(dico_results)
results.to_csv("results/results_benchmark_full.csv")
