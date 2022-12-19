import torch
import torch.nn as nn

from tqdm import tqdm


def train_loop(dataloader, model, loss_fn, optimizer, max_batch=None):
    avg_loss = 0
    count = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        avg_loss += loss.item()
        count += 1

        def closure():
            return loss_fn(model(X), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(closure)

        if max_batch is not None and batch >= max_batch:
            break

    return avg_loss / count


def test_loop(dataloader, model, loss_fn, max_batch):
    test_loss = 0
    count = 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
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
    max_batch=None
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

    test_loss = test_loop(test_dataloader, model, loss_fn, max_batch)

    train_losses = []
    test_losses = [test_loss]
    pbar = tqdm(range(epochs))

    pbar.set_description(
        f"Initialisation"
        f" - Average test loss: {test_loss:.8f}"
    )

    for epoch in pbar:

        train_loss = train_loop(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            max_batch=max_batch
        )
        test_loss = test_loop(test_dataloader, model, loss_fn, max_batch)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        pbar.set_description(
            f"Epoch {epoch+1}"
            f" - Average train loss: {train_loss:.8f}"
            f" - Average test loss: {test_loss:.8f}"
        )

        if scheduler is not None:
            scheduler.step()

    print("Done")
    return train_losses, test_losses
