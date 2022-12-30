# %%
import matplotlib.pyplot as plt
import torch

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
    "type_unrolling": "analysis"
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
