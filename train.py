from tqdm.auto import tqdm

def train(model, train_loader, optimizer, criterion, epoch, DEVICE):
    """
    Trains the model with training data.

    Do NOT modify this function.
    """
    batches = len(train_loader)
    model.train()
    tqdm_bar = tqdm(train_loader, total=batches)
    for batch_idx, (image, label) in enumerate(tqdm_bar):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        tqdm_bar.set_description("Epoch {} - train loss: {:.6f}".format(epoch, loss.item()))