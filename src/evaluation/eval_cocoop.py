import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F

from groundingdino.util.inference import load_model, predict, load_image
from groundingdino.util.misc import nested_tensor_from_tensor_list
from src.detection.metrics import load_annotations, compute_ap
from src.prompt_learning.cocoop import MetaCoOp


def encode_text_with_cocoop(model, base_prompt, ctx):
    """Encode text with image-specific context tokens."""
    tokenizer = model.tokenizer
    bert = model.bert
    feat_map = model.feat_map
    device = next(model.parameters()).device
    
    ctx = ctx.to(device)
    ctx_len = ctx.shape[0]

    tokens = tokenizer(base_prompt, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    # Project context to BERT dimension
    word_emb = bert.embeddings.word_embeddings(input_ids)
    W = torch.randn(256, 768, device=device) / (256 ** 0.5)
    ctx_emb = (ctx @ W).unsqueeze(0)
    
    word_emb[:, :ctx_len, :] = ctx_emb

    out = bert(inputs_embeds=word_emb, attention_mask=attention_mask)
    text_feat = out.last_hidden_state[:, 0, :]
    
    return feat_map(text_feat)


def evaluate(images_dir, ann_csv, cocoop_path, base_prompt, box_thresh, save_dir, config_path, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(config_path, ckpt_path).to(device).eval()

    # Load CoCoOp with meta-network
    ctx_len = 4
    cocoop = MetaCoOp(ctx_len, 256).to(device)
    checkpoint = torch.load(cocoop_path, map_location="cpu")
    cocoop.load_state_dict(checkpoint["state_dict"])

    gt_boxes = load_annotations(ann_csv)
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('jpg', 'png', 'jpeg'))]
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    ap_scores = []
    cosine_sims = []
    
    # Prompt with placeholder tokens
    context_tokens = " ".join(["X"] * ctx_len)
    full_prompt = f"{context_tokens} {base_prompt}"

    for img_name in tqdm(sorted(image_files)):
        img_path = os.path.join(images_dir, img_name)
        _, img_tensor = load_image(img_path)
        img_tensor = img_tensor.to(device)

        # Extract image feature
        with torch.no_grad():
            nt = nested_tensor_from_tensor_list([img_tensor])
            fpn_feats, _ = model.backbone(nt)
            feat = model.input_proj[0](fpn_feats[0].tensors)
            img_feat = feat.mean([2, 3])

        # Generate image-specific context
        ctx = cocoop(img_feat)[0]
        text_feat = encode_text_with_cocoop(model, base_prompt, ctx)

        cos_sim = F.cosine_similarity(img_feat, text_feat).item()
        cosine_sims.append(cos_sim)

        # Run detection
        boxes, _, _ = predict(
            model=model,
            image=img_tensor,
            caption=full_prompt,
            box_threshold=box_thresh,
            text_threshold=box_thresh,
            device=device
        )

        ap = compute_ap(boxes.tolist(), gt_boxes.get(img_name, []))
        ap_scores.append(ap)

        if save_dir:
            Image.open(img_path).save(os.path.join(save_dir, img_name))

    print(f"\nMean AP: {np.mean(ap_scores):.4f}")
    print(f"Mean Cosine Similarity: {np.mean(cosine_sims):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--ann", required=True)
    parser.add_argument("--cocoop", required=True, help="Path to CoCoOp checkpoint")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--box_thresh", type=float, default=0.3)
    parser.add_argument("--save_dir", default=None)
    
    args = parser.parse_args()
    
    evaluate(
        images_dir=args.images,
        ann_csv=args.ann,
        cocoop_path=args.cocoop,
        base_prompt=args.prompt,
        box_thresh=args.box_thresh,
        save_dir=args.save_dir,
        config_path=args.config,
        ckpt_path=args.weights
    )