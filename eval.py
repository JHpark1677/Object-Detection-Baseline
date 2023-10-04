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
from utils import vis_report


def evaluate(model, test_loader, epoch, args, vis, DEVICE):
    """
    Evaluates the trained model with test data.

    Do NOT modify this function.
    """
    model.eval()

    pred_boxes, target_boxes = get_bboxes(
        test_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
    )

    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
    )

    print(f" Test mAP: {mean_avg_prec}")
    
    mean_loss = 0
    vis_report.vis_map_report(vis, args, epoch, mean_loss, mean_avg_prec, stage="Test")
