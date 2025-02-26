import torch
import torch.nn as nn

from tqdm import tqdm


def train_loop(
    dataloader,
    model,
    loss_fn,
    optimizer,
    max_batch=None,
    scheduler=None,
    rescale=False
):
    avg_loss = 0
    count = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(model.device), y.to(model.device)

        # Compute prediction and loss
        pred, _ = model(X)
        assert pred.shape == y.shape
        loss = loss_fn(pred, y)
        avg_loss += loss.item()
        count += 1

        def closure():
            with torch.no_grad():
                return loss_fn(model(X)[0], y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(closure)

        if rescale:
            model.rescale()

        if scheduler is not None:
            scheduler.step()

        if max_batch is not None and batch >= max_batch:
            break

    return avg_loss / count


def test_loop(dataloader, model, loss_fn, max_batch):
    test_loss = 0
    count = 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            pred, _ = model(X)
            test_loss += loss_fn(pred, y).item()
            count += 1
            if max_batch is not None and batch >= max_batch:
                break

    test_loss /= count
    return test_loss


def train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    scheduler=None,
    epochs=10,
    max_batch=None,
    rescale=False,
    verbose=True
):
    """
    Training process

    Parameters
    ----------
    model : torch.nn.Module
        Torch network
    train_dataloader : torch.utils.data.DataLoader
        Train dataset
    test_dataloader : torch.utils.data.DataLoader
        Test dataset
    optimizer : torch.optim.Optimizer
        Torch optimizer
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Learning step size scheduler, by default None
    epochs : int, optional
        Number of epochs, by default 10
    max_batch : int, optional
        Maximum number of minibatches per epoch, by default None

    Returns
    -------
    list, list
        Train losses, test losses
    """
    loss_fn = nn.MSELoss()

    train_losses, test_losses = [], []
    if test_dataloader is not None:
        test_loss = test_loop(test_dataloader, model, loss_fn, max_batch)
        test_losses.append(test_loss)
    else:
        test_loss = 0

    if verbose:
        pbar = tqdm(range(epochs))

        pbar.set_description(
            f"Initialisation"
            f" - Average test loss: {test_loss:.8f}"
        )
    else:
        pbar = range(epochs)

    for epoch in pbar:

        train_loss = train_loop(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            max_batch=max_batch,
            scheduler=scheduler,
            rescale=rescale
        )
        train_losses.append(train_loss)

        if test_dataloader is not None:
            test_loss = test_loop(test_dataloader, model, loss_fn, max_batch)
            test_losses.append(test_loss)

        if verbose:

            pbar.set_description(
                f"Epoch {epoch+1}"
                f" - Average train loss: {train_loss:.8f}"
                f" - Average test loss: {test_loss:.8f}"
            )

    if verbose:
        print("Done")
    return train_losses, test_losses
