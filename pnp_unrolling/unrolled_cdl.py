import torch
import numpy as np

from .models import UnrolledNet
from .datasets import create_imagewoof_dataloader
from .optimizers import SLS
from .train import train


class UnrolledCDL:

    def __init__(
        self,
        n_components,
        kernel_size,
        lmbd,
        path_data,
        color=True,
        download=False,
        sigma_noise=0.1,
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
        self.sigma_noise = sigma_noise
        self.download = download

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
            type_layer=type_unrolling
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

        # Dataloader
        self.train_dataloader = create_imagewoof_dataloader(
            self.path_data,
            self.sigma_noise,
            self.device,
            self.dtype,
            mini_batch_size=self.mini_batch_size,
            train=True,
            random_state=self.random_state,
            color=self.color,
            download=self.download
        )

        self.test_dataloader = create_imagewoof_dataloader(
            self.path_data,
            self.sigma_noise,
            self.device,
            self.dtype,
            mini_batch_size=self.mini_batch_size,
            train=False,
            random_state=self.random_state,
            color=self.color
        )

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
            max_batch=self.max_batch
        )

        return self.unrolled_net, train_losses, test_losses
