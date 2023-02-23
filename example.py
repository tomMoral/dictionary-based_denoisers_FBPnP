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
    "D_shared": False,
    "optimizer": "adam",
    "path_data": PATH_DATA,
    "max_sigma_noise": 0.2,
    "min_sigma_noise": 0.05,
    "mini_batch_size": 1,
    "max_batch": 10,
    "epochs": 50,
    "type_unrolling": "analysis",
    "avg": True,
    "rescale": False,
    "pseudo_gd": False,
    "fixed_noise": False
}


unrolled_cdl = UnrolledCDL(**params_model)

# %%
img_noise, img = next(iter(unrolled_cdl.train_dataloader))
img_result = unrolled_cdl.unrolled_net(img_noise).to("cpu").detach().numpy()

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
img_result = unrolled_net(img_noise).to("cpu").detach().numpy()

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
plt.savefig("example.png")

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


sigma_noise = 0.05
sigma_blurr = 0.1

blurr = gaussian_kernel(10, sigma_blurr)
blurr_torch = torch.tensor(blurr, device=DEVICE, dtype=torch.float)

blurred_img = torch.nn.functional.conv_transpose2d(
    img.transpose(1, 0),
    blurr_torch
).transpose(1, 0)

noise = torch.randn(
    blurred_img.size(),
    dtype=blurred_img.dtype,
    device=blurred_img.device
)
blurred_img += noise * sigma_noise

blurred_img_display = torch.nn.functional.conv2d(
    img.transpose(1, 0),
    torch.flip(blurr_torch, dims=[2, 3]),
    padding="same"
).transpose(1, 0)

noise = torch.randn(
    blurred_img_display.size(),
    dtype=blurred_img.dtype,
    device=blurred_img.device
)
blurred_img_display += noise * sigma_noise

# %%
# unrolled_cdl.set_lmbd(1e-8)
out_regul, _ = unrolled_cdl.predict(
    blurred_img,
    blurr,
    n_iter=500,
    img_test=img,
    regul=True
)

# %%
out, _ = unrolled_cdl.predict(
    blurred_img,
    blurr,
    n_iter=1000,
    img_test=img,
    regul=False,
)

# %%
ols_denoised = unrolled_net(out)

# %%
def psnr(im1, im2):
    return 10 * torch.log(1 / loss(im1, im2)) / np.log(10)


init_pnp = torch.nn.functional.conv2d(
    blurred_img.transpose(0, 1),
    blurr_torch
).transpose(0, 1)
loss = torch.nn.MSELoss()

fig, axs = plt.subplots(2, 3, figsize=(10, 6))

axs[0, 0].imshow(img.to("cpu").numpy()[0].transpose(1, 2, 0),
              cmap=cmap)
axs[0, 0].set_axis_off()

axs[0, 1].imshow(blurred_img_display.to("cpu").numpy()[0].transpose(1, 2, 0),
              cmap=cmap)
axs[0, 1].set_axis_off()
axs[0, 1].set_title(f"Blurred same\nPSNR: {psnr(blurred_img_display, img):.2f}")

axs[0, 2].imshow(init_pnp.to("cpu").numpy()[0].transpose(1, 2, 0),
              cmap=cmap)
axs[0, 2].set_axis_off()
axs[0, 2].set_title(f"Init. PnP\nPSNR: {psnr(init_pnp, img):.2f}")

axs[1, 0].imshow(out.to("cpu").numpy()[0].transpose(1, 2, 0),
              cmap=cmap)
axs[1, 0].set_axis_off()
axs[1, 0].set_title(f"OLS\nPSNR: {psnr(out, img):.2f}")

axs[1, 1].imshow(out_regul.to("cpu").numpy()[0].transpose(1, 2, 0),
              cmap=cmap)
axs[1, 1].set_axis_off()
axs[1, 1].set_title(f"PnP\nPSNR: {psnr(out_regul, img):.2f}")

axs[1, 2].imshow(ols_denoised.to("cpu").numpy()[0].transpose(1, 2, 0),
              cmap=cmap)
axs[1, 2].set_axis_off()
axs[1, 2].set_title(f"OLS + denoising\nPSNR: {psnr(ols_denoised, img):.2f}")


plt.tight_layout()
plt.savefig("example_deblurring.png")

# %%
