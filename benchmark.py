import numpy as np
import torch
import torch.nn as nn
import scipy
import bm3d
import itertools
import pandas as pd

from pnp_unrolling.unrolled_cdl import UnrolledCDL
from external.network_unet import UNetRes
from external.utils_dpir import test_mode as test_mode_dpir
from pnp_unrolling.datasets import create_imagewoof_dataloader
from utils.wavelet_utils import wavelet_op
from utils.measurement_tools import get_operators
from utils.tools import op_norm2
from tqdm import tqdm
from joblib import Memory

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


PATH_DATA = "/storage/store2/work/bmalezie/imagewoof/"
COLOR = False
DEVICE = "cuda:3"
mem = Memory(location='./tmp_benchmark_pnp/', verbose=0)
N_EXP = 1


PARAMS_UNROLLED = {
    "n_layers": [20],
    "n_components": [50],
    "kernel_size": [5],
    "avg": [True, False],
    "D_shared": [True, False],
    "lmbd": np.logspace(-5, -2, num=4),
    "std_noise": [0.02, 0.05, 0.1]
}


def load_nets(params_unrolled):
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
        net.module.load_state_dict(checkpoint.module.state_dict(),
                                   strict=True)

    keys, values = zip(*params_unrolled.items())
    permuts_params_unrolled = [
        dict(zip(keys, v)) for v in itertools.product(*values)
    ]

    params_model = {
        "color": COLOR,
        "device": DEVICE,
        "dtype": torch.float,
        "optimizer": "adam",
        "path_data": PATH_DATA,
        "mini_batch_size": 1,
        "max_batch": 10,
        "epochs": 50,
        "rescale": False,
        "fixed_noise": True,
    }

    dico_nets = {}

    print("Loading unrolled")

    for param_unrolled in permuts_params_unrolled:
        print(param_unrolled)

        signature = ""
        for key in param_unrolled:
            signature += f"{key}_{param_unrolled[key]}_"

        unrolled_cdl_analysis = UnrolledCDL(
            **params_model,
            **param_unrolled,
            type_unrolling="analysis"
        )

        dico_nets[signature + "analysis"], _, _ = unrolled_cdl_analysis.fit()

        unrolled_cdl_synthesis = UnrolledCDL(
            **params_model,
            **param_unrolled,
            type_unrolling="synthesis"
        )

        dico_nets[signature + "synthesis"], _, _ = unrolled_cdl_synthesis.fit()

    print("Done")

    return net, dico_nets


DRUNET, DICO_NETS = load_nets(PARAMS_UNROLLED)


def prox_L1(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)


def apply_model(model, x, dual, reg_par, net=None, update_dual=False):

    if model == "bm3d":
        return bm3d.bm3d(x, reg_par), None
    elif model == "wavelet":
        wave_choice = 'db8'
        Psi, Psit = wavelet_op(x, wav=wave_choice, level=4)
        return Psit(prox_L1(Psi(x), reg_par)), None
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
        return xnet.detach().cpu().numpy()[0, 0], None
    elif model == "unrolled":
        net.set_lmbd(reg_par)
        x_torch = torch.tensor(
            x,
            device=DEVICE,
            dtype=torch.float
        )[None, None, :]
        if dual is not None:
            dual = torch.tensor(
                dual,
                device=DEVICE,
                dtype=torch.float
            )
        xnet, new_dual = net(x_torch, dual)
        if not update_dual:
            return xnet.detach().cpu().numpy()[0, 0], None
        else:
            return xnet.detach().cpu().numpy()[0, 0], new_dual.detach().cpu().numpy()
    elif model == "identity":
        return x, None


def pnp_deblurring(model, pth_kernel, x_observed, reg_par=1e-2,
                   n_iter=50, net=None, update_dual=False):

    if model in ["analysis", "synthesis"]:
        model = "unrolled"

    # define operators
    Phi, Phit = get_operators(type_op='deconvolution', pth_kernel=pth_kernel)

    normPhi2 = op_norm2(Phi, Phit, x_observed.shape)
    gamma = 1.99 / normPhi2

    x_n = Phit(x_observed)
    table_energy = 1e10 * np.ones(n_iter)
    current_dual = None

    for k in range(0, n_iter):
        g_n = Phit((Phi(x_n) - x_observed))
        tmp = x_n - gamma * g_n
        x_old = x_n.copy()
        x_n, current_dual = apply_model(model, tmp, current_dual,
                                        reg_par, net, update_dual)
        table_energy[k] = np.sum((x_n - x_old)**2) / np.sum(x_old**2)

    return np.clip(x_n, 0, 1), table_energy


@mem.cache
def run_test(params):

    pth_kernel = params["pth_kernel"]
    model = params["model"]
    reg_par = params["reg_par"]
    img_num = params["images"]
    std_noise = params["std_noise"]
    n_iter = params["n_iter"]
    update_dual = params["update_dual"]

    img = IMAGES[img_num]

    h = scipy.io.loadmat(pth_kernel)
    h = np.array(h['blur'])

    Phi, Phit = get_operators(type_op='deconvolution',
                              pth_kernel=pth_kernel)
    x_blurred = Phi(img)
    nxb, nyb = x_blurred.shape
    x_observed = x_blurred + std_noise * np.random.rand(nxb, nyb)

    if model in ["analysis", "synthesis"]:
        signature = ""
        for key in PARAMS_UNROLLED:
            signature += f"{key}_{params[key]}_"
        signature += model
        net = DICO_NETS[signature]
    elif model == "drunet":
        net = DRUNET
    else:
        net = None

    rec, energy = pnp_deblurring(model, pth_kernel, x_observed,
                                 reg_par, net=net, n_iter=n_iter,
                                 update_dual=update_dual)

    psnr = peak_signal_noise_ratio(rec, img)
    ssim = structural_similarity(rec, img)
    conv = energy[:-10].mean()

    results = {
        "psnr": psnr,
        "ssim": ssim,
        "conv": conv
    }

    return results


if __name__ == "__main__":

    dataloader = create_imagewoof_dataloader(
        PATH_DATA,
        min_sigma_noise=0.1,
        max_sigma_noise=0.1,
        device=DEVICE,
        dtype=torch.float,
        mini_batch_size=1,
        train=False,
        color=False,
        fixed_noise=True
    )

    IMAGES = []

    for i in range(N_EXP):

        img_noise, img = next(iter(dataloader))
        img_noise, img = img_noise.cpu().numpy()[0, 0], img.cpu().numpy()[0, 0]

        IMAGES.append(img.copy())

    hyperparams = {
        "images": np.arange(0, N_EXP, 1, dtype=int),
        "pth_kernel": ['blur_models/blur_3.mat'],
        "reg_par": np.logspace(-5, 1, num=20),
        "model": ["identity", "analysis", "synthesis"],
        "n_iter": [500],
        "update_dual": [True, False],
        # "model": ["drunet", "analysis", "synthesis", "wavelet", "identity"],
        # "model": ["bm3d", "drunet", "analysis", "synthesis", "wavelet"],
    }

    hyperparams.update(PARAMS_UNROLLED)

    keys, values = zip(*hyperparams.items())
    permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]

    dico_results = {}

    for params in tqdm(permuts_params):
        try:
            # Avoid running again models with useless params
            if params["model"] == "identity":
                tmp = params["reg_par"]
                params["reg_par"] = 0

            if params["model"] in ["identity", "drunet", "wavelet", "bm3d"]:
                tmp_dual = params["update_dual"]
                params["update_dual"] = True

                tmps = {}
                for param_unrolled in PARAMS_UNROLLED:
                    if param_unrolled != "std_noise":
                        tmps[param_unrolled] = params[param_unrolled]
                        params[param_unrolled] = 0

            # Run test
            results = run_test(params)

            # Get values back for useless params
            if params["model"] == "identity":
                params["reg_par"] = tmp

            if params["model"] in ["identity", "drunet", "wavelet", "bm3d"]:
                params["update_dual"] = tmp_dual

                for param_unrolled in PARAMS_UNROLLED:
                    if param_unrolled != "std_noise":
                        params[param_unrolled] = tmps[param_unrolled]

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
    results.to_csv(str("results/results_pnp_fixed_noise.csv"))
