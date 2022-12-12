# %%
import matplotlib.pyplot as plt

from pnp_unrolling.unrolled_denoiser import BaseUnrolling


DATA_PATH = "/storage/store2/work/bmalezie/imagewoof/"


# %%
unrolled = BaseUnrolling(
    n_layers=20,
    n_components=200,
    kernel_size=10,
    n_channels=3,
    lmbd=1e-3,
    path_data=DATA_PATH,
    etamax=1e10,
    etamin=1e3,
    iterations=200
)
# %%
avg_train_losses, avg_test_losses, times = unrolled.fit()
# %%


fig, axs = plt.subplots(1, 2)
axs[0].plot(avg_train_losses)
axs[1].plot(avg_test_losses)
plt.tight_layout()
plt.savefig("loss.png")
plt.clf()
# %%


img_noise, img = next(iter(unrolled.train_dataloader))
img_result = unrolled.eval(img_noise.to("cpu").numpy())

fig, axs = plt.subplots(1, 3)

axs[0].imshow(img_noise.to("cpu").numpy()[0].transpose(1, 2, 0))
axs[0].set_axis_off()
axs[1].imshow(img.to("cpu").numpy()[0].transpose(1, 2, 0))
axs[1].set_axis_off()
axs[2].imshow(img_result[0].transpose(1, 2, 0))
axs[2].set_axis_off()
plt.tight_layout()
plt.savefig("example.png")
plt.clf()
# %%
