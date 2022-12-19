# %%
import matplotlib.pyplot as plt
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from pnp_unrolling.models import SynthesisUnrolled
from pnp_unrolling.datasets import create_imagewoof_dataloader
from pnp_unrolling.train import train
# from pnp_unrolling.optimizers import SLS


PATH_DATA = "/storage/store2/work/bmalezie/imagewoof/"
COLOR = True
DEVICE = "cuda:0"


params_model = {
    "n_layers": 20,
    "n_components": 300,
    "kernel_size": 10,
    "lmbd": 1e-2,
    "n_channels": 3 if COLOR else 1,
    "device": DEVICE,
    "dtype": torch.float,
    "avg": False,
    "D_shared": True
}

params_dataloader = {
    "path_data": PATH_DATA,
    "sigma_noise": 0.1,
    "device": DEVICE,
    "dtype": torch.float,
    "mini_batch_size": 10,
    "color": COLOR
}

# %%

unrolled_net = SynthesisUnrolled(**params_model)
train_dataloader = create_imagewoof_dataloader(
    **params_dataloader,
    train=True
)
test_dataloader = create_imagewoof_dataloader(
    **params_dataloader,
    train=False,
)

optimizer = Adam(unrolled_net.parameters(), lr=0.1)
# optimizer = SLS(unrolled_net.parameters(), lr=0.1)
scheduler = ExponentialLR(optimizer, gamma=0.9)


# %%
img_noise, img = next(iter(train_dataloader))
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

# %%

train_loss, test_loss = train(
    unrolled_net,
    train_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
    epochs=50,
    max_batch=20
)
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
