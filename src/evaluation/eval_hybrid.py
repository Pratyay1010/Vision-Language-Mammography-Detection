import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F

from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util.misc import nested_tensor_from_tensor_list
from src.detection.metrics import load_annotations, compute_ap


def encode_text_with_coop(model, prompt, ctx):
    """Encode text with mean-pooled context (Task 3 training protocol)."""
    tokenizer = model.tokenizer
    bert = model.bert
    feat_map = model.feat_map
    device = next(model.parameters()).device
    
    ctx = ctx.to(device)
    ctx_vec = ctx.mean(0, keepdim=True)

    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    bert_out = bert(**tokens).last_hidden_state[:, 0, :]

    return feat_map(bert_out) + ctx_vec


def evaluate(images_dir, ann_csv, coop_ckpt, prompt, box_thresh, config_path, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(config_path, ckpt_path).to(device).eval()

    # Load learned context (mean-pooled during inference)
    ctx = torch.load(coop_ckpt, map_location="cpu")["ctx"]

    gt_boxes = load_annotations(ann_csv)
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('jpg', 'png', 'jpeg'))]

    ap_scores = []
    cosine_sims = []

    for img_name in tqdm(sorted(image_files)):
        img_path = os.path.join(images_dir, img_name)
        _, img_tensor = load_image(img_path)
        img_tensor = img_tensor.to(device)

        # Compute text embedding with mean context
        text_vec = encode_text_with_coop(model, prompt, ctx)

        # Extract image embedding
        with torch.no_grad():
            nt = nested_tensor_from_tensor_list([img_tensor])
            fpn_feats, _ = model.backbone(nt)
            feat = model.input_proj[0](fpn_feats[0].tensors)
            img_feat = feat.mean([2, 3])

        cos_sim = F.cosine_similarity(img_feat, text_vec).item()
        cosine_sims.append(cos_sim)

        # Run detection
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

    print(f"\nMean AP: {np.mean(ap_scores):.4f}")
    print(f"Mean Cosine Similarity: {np.mean(cosine_sims):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--ann", required=True)
    parser.add_argument("--coop", required=True, help="Path to hybrid model checkpoint")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--box_thresh", type=float, default=0.3)
    
    args = parser.parse_args()
    
    evaluate(
        images_dir=args.images,
        ann_csv=args.ann,
        coop_ckpt=args.coop,
        prompt=args.prompt,
        box_thresh=args.box_thresh,
        config_path=args.config,
        ckpt_path=args.weights
    )