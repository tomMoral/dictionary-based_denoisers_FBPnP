# %%
import matplotlib.pyplot as plt
import torch
import numpy as np
from pnp_unrolling.unrolled_cdl import UnrolledCDL


PATH_DATA = "/storage/store2/work/bmalezie/imagewoof/"
COLOR = True
DEVICE = "cuda:3"

# %%


params_model = {
    "n_layers": 20,
    "n_components": 50,
    "kernel_size": 5,
    "lmbd": 1e-3,
    "color": COLOR,
    "device": DEVICE,
    "dtype": torch.float,
    "D_shared": True,
    "optimizer": "adam",
    "path_data": PATH_DATA,
    "sigma_noise": 0.05,
    "mini_batch_size": 1,
    "max_batch": 10,
    "epochs": 50,
    "avg": True
}
# %%

unrolled_cdl_analysis = UnrolledCDL(
    **params_model,
    type_unrolling="analysis"
)
unrolled_cdl_synthesis = UnrolledCDL(
    **params_model,
    type_unrolling="synthesis"
)

unrolled_net_analysis, _, _ = unrolled_cdl_analysis.fit()
unrolled_net_synthesis, _, _ = unrolled_cdl_synthesis.fit()

# %%


def gaussian_kernel(dim, sigma):
    """Generate a 2D gaussian kernel of given size and standard deviation.

    Parameters
    ----------
    dim : int
        Kernel size
    sigma : float
        Kernel standard deviation

    Returns
    -------
    kernel : ndarray, shape (dim, dim)
        Gaussian kernel of size dim*dim
    """
    t = np.linspace(-1, 1, dim)
    gaussian = np.exp(-0.5 * (t / sigma) ** 2)
    kernel = gaussian[None, :] * gaussian[:, None]
    kernel /= kernel.sum()

    return kernel[None, None, :, :]


_, img = next(iter(unrolled_cdl_synthesis.test_dataloader))

blurr = gaussian_kernel(10, 0.3)
blurr_torch = torch.tensor(blurr, device=DEVICE, dtype=torch.float)
blurred_img = torch.nn.functional.conv_transpose2d(
    img.transpose(1, 0),
    blurr_torch
).transpose(1, 0)
blurred_img_display = torch.nn.functional.conv2d(
    img.transpose(1, 0),
    torch.flip(blurr_torch, dims=[2, 3]),
    padding="same"
).transpose(1, 0)

# %%
_, psnrs_analysis = unrolled_cdl_analysis.predict(
    blurred_img,
    blurr,
    n_iter=100,
    img_test=img
)
_, psnrs_synthesis = unrolled_cdl_synthesis.predict(
    blurred_img,
    blurr,
    n_iter=100,
    img_test=img
)

# %%
fig = plt.figure(figsize=(4, 3))

plt.plot(psnrs_synthesis, label="Synthesis")
plt.plot(psnrs_analysis, label="Analysis")
plt.xlabel("PnP iteration")
plt.ylabel("PSNR")
plt.legend()
plt.tight_layout()
plt.savefig("comparison_psnr_pnp.png")


# %%
