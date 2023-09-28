from tqdm.auto import tqdm
from util.utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint
)

def train(model, train_loader, optimizer, criterion, epoch, DEVICE):
    """
    Trains the model with training data.

    Do NOT modify this function.
    """
    batches = len(train_loader)
    model.train()
    #tqdm_bar = tqdm(train_loader, total=batches)
        # image = image.to(DEVICE)
        # label = label.to(DEVICE)
        # optimizer.zero_grad()
        # print("what's input's shape ? :", image.shape)
        # output = model(image)
        # print("what's output's shape ? : ", output.shape)
        # loss = criterion.yolo_multitask_loss(output, label)
        # loss.backward()
        # optimizer.step()
        # tqdm_bar.set_description("Epoch {} - train loss: {:.6f}".format(epoch, loss.item()))
    pred_boxes, target_boxes = get_bboxes(
        train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
    )

    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
    )
    
    print(f"Train mAP: {mean_avg_prec}")

    print(f"Train mAP: {mean_avg_prec}")

        #if mean_avg_prec > 0.9:
        #    checkpoint = {
        #        "state_dict": model.state_dict(),
        #        "optimizer": optimizer.state_dict(),
        #    }
        #    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
        #    import time
        #    time.sleep(10)

    train_fn(train_loader, model, optimizer, criterion, DEVICE)


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")