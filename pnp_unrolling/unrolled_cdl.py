import torch
import numpy as np

from .models import UnrolledNet
from .datasets import (create_imagewoof_dataloader,
                       create_imagenet_dataloader)
from .optimizers import SLS
from .train import train
from tqdm import tqdm


class UnrolledCDL:

    def __init__(
        self,
        n_components,
        kernel_size,
        lmbd,
        path_data,
        color=True,
        download=False,
        max_sigma_noise=0.1,
        min_sigma_noise=0.,
        std_noise=None,
        type_unrolling="synthesis",
        n_layers=20,
        epochs=20,
        max_batch=10,
        optimizer="adam",
        lr=0.1,
        gamma=0.9,
        mini_batch_size=10,
        device=None,
        dtype=torch.float,
        random_state=2147483647,
        window=False,
        D_shared=False,
        step_size_scaling=None,
        avg=True,
        rescale=False,
        fixed_noise=False,
        activation="soft-thresholding",
        verbose=True,
        dataset="imagenet"
    ):

        self.mini_batch_size = mini_batch_size
        self.random_state = random_state
        self.epochs = epochs
        self.max_batch = max_batch
        self.device = device
        self.dtype = dtype
        self.gamma = gamma
        self.optimizer_name = optimizer
        self.path_data = path_data
        self.download = download
        self.avg = avg
        self.rescale = rescale
        self.verbose = verbose

        n_channels = 3 if color else 1
        self.color = color

        # CSC solver
        self.unrolled_net = UnrolledNet(
            n_layers,
            n_components,
            kernel_size,
            n_channels,
            lmbd,
            device,
            dtype,
            type_layer=type_unrolling,
            avg=avg,
            D_shared=D_shared,
            step_size_scaling=step_size_scaling,
            activation=activation
        )

        # Optimizer
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.unrolled_net.parameters(),
                lr=lr
            )
        elif optimizer == "linesearch":
            self.optimizer = SLS(
                self.unrolled_net.parameters(),
                lr=lr
            )

        if fixed_noise and std_noise is not None:
            max_sigma_noise = std_noise
            min_sigma_noise = std_noise

        if dataset == "imagewoof":
            create_dataloader = create_imagewoof_dataloader
        elif dataset == "imagenet":
            create_dataloader = create_imagenet_dataloader

        # Dataloader
        self.train_dataloader = create_dataloader(
            self.path_data,
            max_sigma_noise,
            min_sigma_noise,
            self.device,
            self.dtype,
            mini_batch_size=self.mini_batch_size,
            train=True,
            random_state=self.random_state,
            color=self.color,
            download=self.download,
            fixed_noise=fixed_noise
        )

        self.test_dataloader = create_dataloader(
            self.path_data,
            max_sigma_noise,
            min_sigma_noise,
            self.device,
            self.dtype,
            mini_batch_size=self.mini_batch_size,
            train=False,
            random_state=self.random_state,
            color=self.color,
            fixed_noise=fixed_noise
        )

        # # Scale lambda max
        # self.unrolled_net.set_lmbd(lmbd)

        # LR scheduler
        if self.max_batch is None:
            self.max_batch = len(self.train_dataloader)

        if self.optimizer_name == "adam":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                np.power(self.gamma, 1 / self.max_batch)
            )
        elif self.optimizer_name == "linesearch":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs * self.max_batch
            )
        else:
            self.scheduler = None

    def fit(self):

        # Train
        train_losses, test_losses = train(
            self.unrolled_net,
            self.train_dataloader,
            self.test_dataloader,
            self.optimizer,
            scheduler=self.scheduler,
            epochs=self.epochs,
            max_batch=self.max_batch,
            rescale=self.rescale,
            verbose=self.verbose
        )

        return self.unrolled_net, train_losses, test_losses

    def set_lmbd(self, lmbd):
        self.unrolled_net.set_lmbd(lmbd)

    def predict(self, x, blurr, step=1., n_iter=100,
                img_test=None, regul=True):

        conv = torch.nn.functional.conv2d
        convt = torch.nn.functional.conv_transpose2d

        if type(blurr) == np.ndarray:
            blurr = torch.tensor(
                blurr,
                device=self.device,
                dtype=self.dtype
            )

        if type(x) == np.ndarray:
            x = torch.tensor(
                x,
                device=self.device,
                dtype=self.dtype
            )

        with torch.no_grad():
            out = conv(
                x.transpose(0, 1),
                blurr
            ).transpose(0, 1)

            pbar = tqdm(range(n_iter))
            loss = torch.nn.MSELoss()
            psnrs = []

            if img_test is not None:
                psnr = 10 * torch.log(1 / loss(out, img_test)) / np.log(10)
                pbar.set_description(
                    f"Initialisation"
                    f" - PSNR: {psnr:.4f}"
                )
                psnrs.append(psnr.item())

            for i in pbar:

                result1 = convt(out.transpose(0, 1), blurr).transpose(0, 1) - x
                result2 = conv(
                    result1.transpose(0, 1),
                    blurr
                ).transpose(0, 1)

                out_old = out.clone()
                if regul:
                    out = self.unrolled_net(out - step * result2)
                else:
                    out = out - step * result2

                if img_test is not None:
                    psnr = 10 * torch.log(1 / loss(out, img_test)) / np.log(10)
                    diff = loss(out, out_old)
                    pbar.set_description(
                        f"Iteration {i + 1}"
                        f" - PSNR: {psnr:.4f}"
                        f" - diff: {diff.item():.4f}"
                    )
                    psnrs.append(psnr.item())
                    if diff < 1e-15:
                        break

            return out, psnrs
