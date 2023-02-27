# %%
import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy

from tqdm import tqdm
from pnp_unrolling.unrolled_cdl import UnrolledCDL
from utils.measurement_tools import get_operators
from utils.tools import op_norm2
from pnp_unrolling.datasets import create_imagewoof_dataloader

PATH_DATA = "/storage/store2/work/bmalezie/imagewoof/"
COLOR = False
DEVICE = "cuda:3"
reg = 0.5
STD_NOISE = 0.05

# %%

params_model = {
    "n_layers": 20,
    "n_components": 50,
    "kernel_size": 5,
    "lmbd": 1e-3,
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
    "D_shared": False,
    "type_unrolling": "dfb_net",
    "lr": 1e-1
}


unrolled_cdl = UnrolledCDL(**params_model)

# %%
img_noise, img = next(iter(unrolled_cdl.train_dataloader))
img_result, _ = unrolled_cdl.unrolled_net(img_noise)
img_result = img_result.to("cpu").detach().numpy()

fig, axs = plt.subplots(1, 3)

if img_noise.shape[1] == 1:
    cmap = "gray"
else:
    cmap = None

axs[0].imshow(img_noise.to("cpu").numpy()[0].transpose(1, 2, 0), cmap=cmap)
axs[0].set_axis_off()
axs[1].imshow(img.to("cpu").numpy()[0].transpose(1, 2, 0), cmap=cmap)
axs[1].set_axis_off()
axs[2].imshow(img_result[0].transpose(1, 2, 0), cmap=cmap)
axs[2].set_axis_off()
plt.tight_layout()

# %%
unrolled_net, train_loss, test_loss = unrolled_cdl.fit()
# %%

plt.clf()
fig, axs = plt.subplots(1, 2)
axs[0].plot(train_loss)
axs[0].set_title("Train loss")
axs[1].plot(test_loss)
axs[1].set_title("Test loss")
plt.tight_layout()
plt.savefig("loss.png")

# %%

plt.clf()
img_result, _ = unrolled_net(img_noise)
img_result = img_result.to("cpu").detach().numpy()

fig, axs = plt.subplots(1, 3)

if img_noise.shape[1] == 1:
    cmap = "gray"
else:
    cmap = None

axs[0].imshow(img_noise.to("cpu").numpy()[0].transpose(1, 2, 0), cmap=cmap)
axs[0].set_axis_off()
axs[1].imshow(img.to("cpu").numpy()[0].transpose(1, 2, 0), cmap=cmap)
axs[1].set_axis_off()
axs[2].imshow(img_result[0].transpose(1, 2, 0), cmap=cmap)
axs[2].set_axis_off()
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


def pnp_deblurring(model, pth_kernel, x_observed, reg_par=0.5 * STD_NOISE,
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

    for k in tqdm(range(0, n_iter)):
        g_n = Phit((Phi(x_n) - x_observed))
        tmp = x_n - gamma * g_n
        x_old = x_n.copy()
        x_n, current_dual = apply_model(model, tmp, current_dual,
                                        reg_par, net, update_dual)
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

pth_kernel = 'blur_models/blur_3.mat'

img_noise, img = next(iter(dataloader))
img_noise, img = img_noise.cpu().numpy()[0, 0], img.cpu().numpy()[0, 0]

h = scipy.io.loadmat(pth_kernel)
h = np.array(h['blur'])

Phi, Phit = get_operators(type_op='deconvolution',
                            pth_kernel=pth_kernel)
x_blurred = Phi(img)
nxb, nyb = x_blurred.shape
x_observed = x_blurred + STD_NOISE * np.random.rand(nxb, nyb)


x_result, energy = pnp_deblurring(
    "analysis",
    pth_kernel,
    x_observed,
    n_iter=500,
    reg_par=1e-3,
    update_dual=True,
    net=unrolled_net
)
# %%
from skimage.metrics import peak_signal_noise_ratio

psnr = peak_signal_noise_ratio(x_result, img)

# %%
psnr
# %%
energy[-1]

# %%

# %%
