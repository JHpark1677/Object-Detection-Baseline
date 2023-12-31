import os

import torch
from tqdm.auto import tqdm
from util.utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
)
from utils import resume
from utils import vis_report

def train(model, train_loader, optimizer, criterion, epoch, args, vis, DEVICE):
    """
    Trains the model with training data.

    Do NOT modify this function.
    """
    model.train()

    pred_boxes, target_boxes = get_bboxes(
        train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
    )

    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
    )
    
    print(f" Train mAP: {mean_avg_prec}")

    mean_loss = train_fn(train_loader, model, optimizer, criterion, epoch, args, vis, DEVICE)
    vis_report.vis_map_report(vis, args, epoch, mean_loss, mean_avg_prec, stage="Train")
    resume.save_ckp(model, args)
    

def train_fn(train_loader, model, optimizer, loss_fn, epoch, args, vis, DEVICE):
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
        loop.set_description("Epoch {} - train loss: {:.6f}".format(epoch, loss.item()))
    
    final_loss = sum(mean_loss) / len(mean_loss)    
    print(f"Mean loss was {final_loss}")
    return final_loss
    