import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

from groundingdino.util.inference import load_model, predict, load_image
from src.detection.metrics import load_annotations, compute_ap


def run_inference(images_dir, ann_csv, prompt, box_thresh, save_dir, config_path, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(config_path, ckpt_path)
    model.to(device).eval()

    gt_boxes = load_annotations(ann_csv)
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('jpg', 'png', 'jpeg'))]
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    ap_scores = []
    
    for img_name in tqdm(sorted(image_files)):
        img_path = os.path.join(images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        _, img_tensor = load_image(img_path)
        
        boxes, _, _ = predict(
            model=model,
            image=img_tensor,
            caption=prompt,
            box_threshold=box_thresh,
            text_threshold=box_thresh,
            device=device
        )

        ap = compute_ap(boxes.tolist(), gt_boxes.get(img_name, []))
        ap_scores.append(ap)

        if save_dir:
            image.save(os.path.join(save_dir, img_name))

    print(f"Mean AP: {np.mean(ap_scores):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--ann", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--box_thresh", type=float, default=0.3)
    parser.add_argument("--save_dir", default=None)
    
    args = parser.parse_args()
    
    run_inference(
        images_dir=args.images,
        ann_csv=args.ann,
        prompt=args.prompt,
        box_thresh=args.box_thresh,
        save_dir=args.save_dir,
        config_path=args.config,
        ckpt_path=args.weights
    )