import argparse
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from groundingdino.util.inference import load_model
from groundingdino.util.misc import nested_tensor_from_tensor_list
from src.utils.dataset import SemiSupervisedDataset
from src.utils.losses import cosine_loss
from src.prompt_learning.coop import CoOp


def split_unlabeled_data(csv_path, labeled_ratio=0.1, random_seed=42):
    """Split unlabeled data into labeled (10%) and unlabeled (90%) subsets."""
    df = pd.read_csv(csv_path)
    labeled_df = df.sample(frac=labeled_ratio, random_state=random_seed)
    unlabeled_df = df.drop(labeled_df.index)
    
    labeled_path = "/tmp/hybrid_labeled.csv"
    unlabeled_path = "/tmp/hybrid_unlabeled.csv"
    labeled_df.to_csv(labeled_path, index=False)
    unlabeled_df.to_csv(unlabeled_path, index=False)
    
    return labeled_path, unlabeled_path


def collate_with_boxes(batch):
    images = torch.stack([b[0] for b in batch])
    boxes = [b[1] for b in batch]
    return images, boxes


def collate_without_boxes(batch):
    images = torch.stack([b[0] for b in batch])
    return images, [None] * len(batch)


def train(A_images, A_csv, B_images, B_csv, prompt, save_path, plot_path,
          epochs, lr, batch_size, tau, lambda_u, warmup,
          config_path, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load frozen GroundingDINO
    model = load_model(config_path, ckpt_path).to(device).eval()
    text_encoder = model.bert
    feat_map = model.feat_map
    backbone = model.backbone

    # Prepare dataset B: 10% labeled, 90% unlabeled
    B_labeled_csv, B_unlabeled_csv = split_unlabeled_data(B_csv)

    # Create dataloaders
    dataset_A = SemiSupervisedDataset(A_images, A_csv, labeled=True, strong_aug=False)
    dataset_B_labeled = SemiSupervisedDataset(B_images, B_labeled_csv, labeled=True, strong_aug=False)
    dataset_B_unlabeled_weak = SemiSupervisedDataset(B_images, B_unlabeled_csv, labeled=False, strong_aug=False)
    dataset_B_unlabeled_strong = SemiSupervisedDataset(B_images, B_unlabeled_csv, labeled=False, strong_aug=True)

    dl_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True, collate_fn=collate_with_boxes)
    dl_Bl = DataLoader(dataset_B_labeled, batch_size=batch_size, shuffle=True, collate_fn=collate_with_boxes)
    dl_Bw = DataLoader(dataset_B_unlabeled_weak, batch_size=batch_size, shuffle=True, collate_fn=collate_without_boxes)
    dl_Bs = DataLoader(dataset_B_unlabeled_strong, batch_size=batch_size, shuffle=True, collate_fn=collate_without_boxes)

    # Initialize CoOp
    ctx_len, dim = 4, 256
    coop = CoOp(ctx_len, dim).to(device)
    optimizer = optim.Adam([coop.ctx], lr=lr)

    # Tokenize base prompt
    tokens = model.tokenizer(prompt, return_tensors="pt").to(device)
    base_text_feat = feat_map(text_encoder(**tokens).last_hidden_state[:, 0, :])

    loss_log, sim_log, sup_log, unsup_log = [], [], [], []
    best_sim = -1.0

    print("\n===== Hybrid Training (CoOp + FixMatch) =====\n")

    for epoch in range(epochs):
        total_loss, total_sim, total_sup, total_unsup = 0.0, 0.0, 0.0, 0.0
        steps = 0
        
        lambda_u_current = 0 if epoch < warmup else lambda_u

        loop = tqdm(zip(dl_A, dl_Bl, dl_Bw, dl_Bs), 
                    total=min(len(dl_A), len(dl_Bl), len(dl_Bw), len(dl_Bs)),
                    desc=f"Epoch {epoch+1}/{epochs}")

        for (imgA, _), (imgBl, _), (imgBw, _), (imgBs, _) in loop:
            imgA = imgA.to(device)
            imgBl = imgBl.to(device)
            imgBw = imgBw.to(device)
            imgBs = imgBs.to(device)

            # Extract image features
            with torch.no_grad():
                def extract_feat(imgs):
                    nt = nested_tensor_from_tensor_list(imgs)
                    fpn, _ = backbone(nt)
                    return model.input_proj[0](fpn[0].tensors).mean([2, 3])
                
                fA = extract_feat(imgA)
                fBl = extract_feat(imgBl)
                fBw = extract_feat(imgBw)
                fBs = extract_feat(imgBs)

            # Text embedding with context
            ctx = coop().mean(0, keepdim=True)
            text_vec = base_text_feat + ctx

            # Supervised loss on labeled data
            sup_loss = cosine_loss(fA, text_vec) + cosine_loss(fBl, text_vec)

            # Pseudo-label confidence (cosine similarity threshold)
            pseudo_conf = torch.nn.functional.cosine_similarity(fBw, text_vec).mean().item()
            
            # Unsupervised loss for high-confidence predictions
            unsup_loss = cosine_loss(fBs, text_vec) if pseudo_conf > tau else 0.0

            loss = sup_loss + lambda_u_current * unsup_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_sup += sup_loss.item()
            total_unsup += unsup_loss if isinstance(unsup_loss, float) else unsup_loss.item()
            total_sim += float(torch.nn.functional.cosine_similarity(fA, text_vec).mean())
            steps += 1

        avg_loss = total_loss / steps
        avg_sup = total_sup / steps
        avg_unsup = total_unsup / steps
        avg_sim = total_sim / steps

        loss_log.append(avg_loss)
        sup_log.append(avg_sup)
        unsup_log.append(avg_unsup)
        sim_log.append(avg_sim)

        print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f}, Sup={avg_sup:.4f}, Unsup={avg_unsup:.4f}, Sim={avg_sim:.4f}")

        if avg_sim > best_sim:
            best_sim = avg_sim
            torch.save({"ctx": coop.ctx.detach().cpu()}, save_path)
            print(f"  → Saved best checkpoint")

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(loss_log, label="Total Loss")
    plt.plot(sup_log, label="Supervised Loss")
    plt.plot(unsup_log, label="Unsupervised Loss")
    plt.plot(sim_log, label="Cosine Similarity")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Hybrid Training (FixMatch + CoOp)")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"\nPlot saved: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--A_images", required=True)
    parser.add_argument("--A_csv", required=True)
    parser.add_argument("--B_images", required=True)
    parser.add_argument("--B_csv", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--plot", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--tau", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--lambda_u", type=float, default=1.0, help="Unsupervised loss weight")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup epochs")
    
    args = parser.parse_args()
    
    train(
        A_images=args.A_images,
        A_csv=args.A_csv,
        B_images=args.B_images,
        B_csv=args.B_csv,
        prompt=args.prompt,
        save_path=args.save,
        plot_path=args.plot,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        tau=args.tau,
        lambda_u=args.lambda_u,
        warmup=args.warmup,
        config_path=args.config,
        ckpt_path=args.weights
    )