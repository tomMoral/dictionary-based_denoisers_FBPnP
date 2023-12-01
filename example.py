# %%
import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy
import torch.nn as nn

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
DEVICE = "cuda:1"
STD_NOISE = 0.05

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

# %%
img_noise, img = next(iter(unrolled_synthesis.train_dataloader))
img_result_synthesis, _ = unrolled_synthesis.unrolled_net(img_noise)
img_result_analysis, _ = unrolled_analysis.unrolled_net(img_noise)
img_result_synthesis = img_result_synthesis.to("cpu").detach().numpy()
img_result_analysis = img_result_analysis.to("cpu").detach().numpy()

fig, axs = plt.subplots(1, 4)

if img_noise.shape[1] == 1:
    cmap = "gray"
else:
    cmap = None

axs[0].imshow(img_noise.to("cpu").numpy()[0].transpose(1, 2, 0), cmap=cmap)
axs[0].set_axis_off()
axs[1].imshow(img.to("cpu").numpy()[0].transpose(1, 2, 0), cmap=cmap)
axs[1].set_axis_off()
axs[2].imshow(img_result_synthesis[0].transpose(1, 2, 0), cmap=cmap)
axs[2].set_axis_off()
axs[3].imshow(img_result_analysis[0].transpose(1, 2, 0), cmap=cmap)
axs[3].set_axis_off()
plt.tight_layout()
plt.show()

# %%
unrolled_net_synthesis, _, _ = unrolled_synthesis.fit()
unrolled_net_analysis, _, _ = unrolled_analysis.fit()


# %%


img_result_synthesis, _ = unrolled_net_synthesis(img_noise)
img_result_analysis, _ = unrolled_net_analysis(img_noise)
img_result_synthesis = img_result_synthesis.to("cpu").detach().numpy()
img_result_analysis = img_result_analysis.to("cpu").detach().numpy()

fig, axs = plt.subplots(1, 4)

if img_noise.shape[1] == 1:
    cmap = "gray"
else:
    cmap = None

axs[0].imshow(img_noise.to("cpu").numpy()[0].transpose(1, 2, 0), cmap=cmap)
axs[0].set_axis_off()
axs[1].imshow(img.to("cpu").numpy()[0].transpose(1, 2, 0), cmap=cmap)
axs[1].set_axis_off()
axs[2].imshow(img_result_synthesis[0].transpose(1, 2, 0), cmap=cmap)
axs[2].set_axis_off()
axs[3].imshow(img_result_analysis[0].transpose(1, 2, 0), cmap=cmap)
axs[3].set_axis_off()
plt.tight_layout()
plt.show()
# %%


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

pth_kernel = 'blur_models/blur_3.mat'


def generate_results_pnp(pth_kernel, n_iter=1000):

    img_noise, img = next(iter(dataloader))
    img_noise, img = img_noise.cpu().numpy()[0], img.cpu().numpy()[0]

    h = scipy.io.loadmat(pth_kernel)
    h = np.array(h['blur'])

    Phi, Phit = get_operators(
        type_op='deconvolution',
        pth_kernel=pth_kernel
    )
    x_blurred = Phi_channels(img, Phi)
    nc, nxb, nyb = x_blurred.shape
    x_observed = x_blurred + STD_NOISE * np.random.rand(nc, nxb, nyb)

    x_result_synthesis, convergence_synthesis, psnr_synthesis = pnp_deblurring(
        "synthesis",
        pth_kernel,
        x_observed,
        n_iter=n_iter,
        reg_par=0.1,
        update_dual=True,
        net=unrolled_net_synthesis,
        x_truth=img
    )

    x_result_analysis, convergence_analysis, psnr_analysis = pnp_deblurring(
        "analysis",
        pth_kernel,
        x_observed,
        n_iter=n_iter,
        reg_par=0.001,
        update_dual=True,
        net=unrolled_net_analysis,
        x_truth=img
    )

    x_result_drunet, convergence_drunet, psnr_drunet = pnp_deblurring(
        "drunet",
        pth_kernel,
        x_observed,
        n_iter=n_iter,
        reg_par=0.1,
        update_dual=True,
        net=drunet,
        x_truth=img
    )

    results = {
        "analysis": [x_result_analysis, convergence_analysis, psnr_analysis],
        "synthesis": [x_result_synthesis, convergence_synthesis, psnr_synthesis],
        "drunet": [x_result_drunet, convergence_drunet, psnr_drunet],
        "observation": x_observed,
        "truth": img
    }

    return results


# %%
N_EXP = 5
list_results = []
for _ in range(N_EXP):
    results = generate_results_pnp(pth_kernel)
    list_results.append(results)

# %%

fig, axs = plt.subplots(1, 3)

axs[0].plot(list_results[0]["synthesis"][1])
axs[0].set_yscale("log")
axs[0].set_title("Synthesis")
axs[0].set_ylabel(r"$|x_{n+1} - x_{n}|$")
axs[0].set_xlabel("N iterations")

axs[1].plot(list_results[0]["analysis"][1])
axs[1].set_yscale("log")
axs[1].set_title("Analysis")

axs[2].plot(list_results[0]["drunet"][1])
axs[2].set_yscale("log")
axs[2].set_title("Drunet")

plt.tight_layout()
plt.savefig("convergence.pdf")

# %%

for i, results in enumerate(list_results):

    fig, axs = plt.subplots(1, 5, figsize=(10, 5))

    x_observed = results["observation"]
    img = results["truth"]
    x_result_synthesis = results["synthesis"][0]
    x_result_analysis = results["analysis"][0]
    x_result_drunet = results["drunet"][0]

    psnr_synthesis = results["synthesis"][2][-1]
    psnr_analysis = results["analysis"][2][-1]
    psnr_drunet = results["drunet"][2][-1]

    cmap = None

    axs[0].set_axis_off()
    axs[0].imshow(x_observed.transpose(1, 2, 0), cmap=cmap)
    axs[0].set_title("Observation")
    axs[1].imshow(img.transpose(1, 2, 0), cmap=cmap)
    axs[1].set_title("Ground truth")
    axs[1].set_axis_off()
    axs[2].imshow(x_result_synthesis.transpose(1, 2, 0), cmap=cmap)
    axs[2].set_title(f"Synthesis\n{psnr_synthesis:0.2f}dB")
    axs[2].set_axis_off()
    axs[3].imshow(x_result_analysis.transpose(1, 2, 0), cmap=cmap)
    axs[3].set_title(f"Analysis\n{psnr_analysis:0.2f}dB")
    axs[3].set_axis_off()
    axs[4].imshow(x_result_drunet.transpose(1, 2, 0), cmap=cmap)
    axs[4].set_title(f"Drunet\n{psnr_drunet:0.2f}dB")
    axs[4].set_axis_off()

    plt.tight_layout()
    plt.savefig(f"example_image_{i}.pdf")
    plt.clf()
# %%
