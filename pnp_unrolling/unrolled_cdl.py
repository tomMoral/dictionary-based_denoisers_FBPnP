import torch
import numpy as np

from .models import UnrolledNet
from .optimizers import SLS
from .train import train


class UnrolledCDL:

    def __init__(
        self,
        n_components,
        kernel_size,
        lmbd,
        color=True,
        n_layers=20,
        epochs=20,
        max_batch=10,
        optimizer="adam",
        lr=0.1,
        gamma=0.9,
        mini_batch_size=10,
        device=None,
        dtype=torch.float,
        rescale=False,
        type_unrolling="synthesis",
        D_shared=True,
        init_dual=False,
        accelerated=True,
        activation="soft-thresholding",
        step_size_scaling=None,
        avg=True,
        random_state=None,
        verbose=True,
    ):

        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.max_batch = max_batch
        self.device = device
        self.dtype = dtype
        self.optimizer_name = optimizer
        self.gamma = gamma
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
            D_shared=D_shared,
            init_dual=init_dual,
            accelerated=accelerated,
            activation=activation,
            step_size_scaling=step_size_scaling,
            avg=avg,
            random_state=random_state,
        )

        # # Scale lambda max
        # self.unrolled_net.set_lmbd(lmbd)

        # Setup the optimizer and learning rate scheduler
        if self.max_batch is None:
            self.max_batch = len(self.train_dataloader)
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.unrolled_net.parameters(),
                lr=lr
            )
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                np.power(self.gamma, 1 / self.max_batch)
            )
        elif optimizer == "linesearch":
            self.optimizer = SLS(
                self.unrolled_net.parameters(),
                lr=lr
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs * self.max_batch
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def fit(self, train_dataloader):

        # Train
        train_losses, test_losses = train(
            self.unrolled_net,
            train_dataloader,
            test_dataloader=None,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epochs=self.epochs,
            max_batch=self.max_batch,
            rescale=self.rescale,
            verbose=self.verbose
        )

        return self.unrolled_net, train_losses, test_losses

    def set_lmbd(self, lmbd):
        self.unrolled_net.set_lmbd(lmbd)
