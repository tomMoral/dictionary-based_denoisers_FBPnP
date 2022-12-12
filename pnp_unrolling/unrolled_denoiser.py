import numpy as np
import torch
import torch.nn as nn
import time

from tqdm import tqdm
from .models import SynthesisUnrolled
from .datasets import ImageWoofDataset


class BaseUnrolling(nn.Module):
    """
    Base class for dictionary learning algorithms

    Parameters
    ----------
    n_layers : int
        Number of unrolled layers.
    n_components : int
        Number of channels per dictionary
    kernel_size : int
        Dimension of kernel in convolutional layers
    n_channels : int
        Number of channels in data
    path_data : str
        Path to the dataset
    dataset : str
        Dataset type, default "imagewoof"
    sigma_noise : float
        Noise std
    unrolled_algo : str
        Type of unrolled algorithm, default "synthesis
    lambd : float
        Regularization parameter, default 0.1.
    iterations : int
        Number of iterations in optimization algorithm
    device : str
        Device where the code is run ["cuda", "cpu"].
        If None, "cuda" is chosen if available.
    dtype : torch.type
        Type for torch tensors, default torch.float
    random_state : int
        Random seed
    """
    def __init__(self, n_layers, n_components, kernel_size, n_channels,
                 path_data, dataset="imagewoof", mini_batch_size=10,
                 sigma_noise=0.1, unrolled_algo="synthesis", lmbd=0.1,
                 iterations=100, c=1e-4, etamax=1e2, etamin=1e0, beta=0.5,
                 device=None, dtype=None, random_state=2147483647):

        super().__init__()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if dtype is None:
            self.dtype = torch.float
        else:
            self.dtype = dtype

        # Regularization
        self.lmbd = lmbd

        # Algorithm
        self.n_layers = n_layers
        self.n_components = n_components
        self.kernel_size = kernel_size
        self.n_channels = n_channels

        # Line search
        self.c = c
        self.etamax = etamax
        self.beta = beta
        self.iterations = iterations
        self.etamin = etamin
        self.gamma = 1. / (10. ** ((np.log(self.etamax) -
                           np.log(self.etamin)) / (np.log(10) * iterations)))

        # Random state
        self.random_state = random_state
        self.generator = torch.Generator()
        self.generator.manual_seed(random_state)

        # Data loaders
        if dataset == "imagewoof":
            self.train_dataloader = torch.utils.data.DataLoader(
                ImageWoofDataset(
                    path_data,
                    sigma_noise,
                    device=self.device,
                    dtype=self.dtype,
                    train=True,
                    random_state=random_state
                ),
                batch_size=mini_batch_size,
                shuffle=True,
                generator=self.generator
            )

            self.test_dataloader = torch.utils.data.DataLoader(
                ImageWoofDataset(
                    path_data,
                    sigma_noise,
                    device=self.device,
                    dtype=self.dtype,
                    train=False,
                    random_state=random_state
                ),
                batch_size=mini_batch_size,
                shuffle=True,
                generator=self.generator
            )

        # Network
        if unrolled_algo == "synthesis":
            self.unrolled_net = SynthesisUnrolled(
                n_layers,
                n_components,
                kernel_size,
                n_channels,
                lmbd,
                self.device,
                self.dtype,
                random_state
            )

        # Loss
        self.cost = nn.MSELoss()

    def forward(self, x):
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor (n_batch, n_channels, width, height)
            Data to be processed
        """

        out = self.unrolled_net(x)

        return out

    def stoch_line_search(self, batch_x, batch_y, eta, loss, eps=1e-8):
        """
        Stochastic line search gradient descent
        Parameters
        ----------
        batch_x : torch.tensor (n_batch, n_channels, width, height)
            Ground truth
        batch_y : torch.tensor (n_batch, n_channels, width, height)
            Input
        eta : float
            Starting step size
        loss : float
            Current value of the loss
        eps : float
            Tolerance parameter
        """

        # Compute norm gradient
        norm_grad = torch.sum(
            torch.tensor(
                [torch.sum(param.grad ** 2)
                 for param in self.parameters()]
            )
        )

        with torch.no_grad():
            # Learning step v
            for param in self.parameters():
                param -= self.beta * eta\
                    * param.grad

            init = True
            ok = False

            while not ok:
                if not init:
                    # Backtracking
                    for param in self.parameters():
                        param -= (self.beta-1)\
                            * eta * param.grad
                else:
                    init = False

                # Computing loss with new parameters
                current_cost = self.cost(batch_x, self.forward(batch_y)).item()

                # Stopping criterion
                if current_cost < loss - self.c * eta * norm_grad:
                    ok = True
                else:
                    eta *= self.beta

                if eta < eps:
                    for param in self.parameters():
                        param += eta * param.grad
                    ok = True

        try:
            assert current_cost <= loss
        except AssertionError:
            print("Warning - The cost did not decrease")

    def fit(self):
        """
        Training function

        Returns
        -------
        list, list, list
            List of train loss, test loss, and compute time
        """

        eta = None
        avg_train_losses = []
        avg_test_losses = []
        iter_train_data = iter(self.train_dataloader)
        iter_test_data = iter(self.test_dataloader)
        pbar = tqdm(range(self.iterations))
        times = []

        start = time.time()
        for idx in pbar:

            try:
                batch_x, batch_y = next(iter_train_data)
            except StopIteration:
                iter_train_data = iter(self.train_dataloader)
                batch_x, batch_y = next(iter_train_data)

            if self.device != "cpu":
                batch_x = batch_x.cuda(self.device)
                batch_y = batch_y.cuda(self.device)

            # Forward pass
            out = self.forward(batch_y)

            # Computing loss and gradients
            loss = self.cost(batch_x, out)
            loss.backward()

            avg_train_losses.append(loss.item())

            avg_test_loss = 0
            count = 0
            with torch.no_grad():
                try:
                    test_x, test_y = next(iter_test_data)
                except StopIteration:
                    iter_test_data = iter(self.test_dataloader)
                    test_x, test_y = next(iter_test_data)

                if self.device != "cpu":
                    test_x = test_x.cuda(self.device)
                    test_y = test_y.cuda(self.device)

                # Forward pass
                out = self.forward(test_y)

                # Computing loss and gradients
                avg_test_loss += self.cost(test_x, out).item()
                count += 1
            avg_test_losses.append(avg_test_loss)

            pbar.set_description(
                f"Iteration {idx+1}"
                f" - Average train loss: {loss.item():.4f}"
                f" - Average test loss: {avg_test_loss:.4f}"
            )

            # Optimizing
            if idx == 0:
                eta = self.etamax
            else:
                eta *= self.gamma

            # Compute D update
            self.stoch_line_search(batch_x, batch_y, eta, loss.item())

            # Putting the gradients to zero
            for param in self.parameters():
                param.grad.zero_()

            times.append(time.time()-start)

        return avg_train_losses, avg_test_losses, times

    def eval(self, x):
        """
        Evaluate result for a batch

        Parameters
        ----------
        x : np.array (batch, n_channels, height, width)
            Batch of observations

        Returns
        -------
        np.array (batch, n_channels, height, width)
            Result
        """
        with torch.no_grad():
            return self.forward(
                torch.tensor(x, dtype=torch.float, device=self.device)
            ).to("cpu").detach().numpy()
