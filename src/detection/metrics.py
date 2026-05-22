import pandas as pd
import torch
from groundingdino.util.box_ops import box_iou


def load_annotations(csv_path):
    """Load ground truth boxes from CSV, grouped by image name."""
    df = pd.read_csv(csv_path)
    anns = {}
    
    for _, row in df.iterrows():
        if pd.isna(row["xmin"]):
            continue
        anns.setdefault(row["image_name"], []).append(
            [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
        )
    return anns


def compute_ap(pred_boxes, gt_boxes, iou_thresh=0.5):
    """Compute precision-recall product (AP approximation)."""
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 1.0
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0

    pred = torch.tensor(pred_boxes)
    gt = torch.tensor(gt_boxes)
    
    iou = box_iou(pred, gt)
    max_iou = iou.max(dim=1)[0]

    tp = (max_iou >= iou_thresh).sum().item()
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    
    return precision * recall