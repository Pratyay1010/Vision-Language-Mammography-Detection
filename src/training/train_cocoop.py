import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from groundingdino.util.inference import load_model
from groundingdino.util.misc import nested_tensor_from_tensor_list
from src.utils.dataset import ImageDataset
from src.utils.losses import cosine_loss
from src.prompt_learning.cocoop import MetaCoOp


def train(images_dir, ann_csv, save_path, plot_path, prompt, epochs, lr,
          config_path, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load frozen GroundingDINO
    model = load_model(config_path, ckpt_path).to(device).eval()
    text_encoder = model.bert
    feat_map = model.feat_map
    backbone = model.backbone

    # Initialize CoCoOp
    ctx_len, dim = 4, 256
    cocoop = MetaCoOp(ctx_len, dim).to(device)
    optimizer = optim.Adam(cocoop.parameters(), lr=lr)

    # Data
    dataset = ImageDataset(images_dir, ann_csv)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Tokenize base prompt
    tokens = model.tokenizer(prompt, return_tensors="pt").to(device)
    base_text_feat = feat_map(text_encoder(**tokens).last_hidden_state[:, 0, :])

    losses, similarities = [], []
    best_sim = -1.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_sim = 0.0
        
        for img_tensor in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            img_tensor = img_tensor.to(device).squeeze(0)
            
            if img_tensor.shape[0] == 1:
                img_tensor = img_tensor.repeat(3, 1, 1)

            # Extract image features
            with torch.no_grad():
                nt = nested_tensor_from_tensor_list([img_tensor])
                fpn_feats, _ = backbone(nt)
                img_feat = model.input_proj[0](fpn_feats[0].tensors).mean([2, 3])

            # Generate image-specific context
            ctx = cocoop(img_feat)
            text_vec = base_text_feat + ctx.mean(0, keepdim=True)

            sim = torch.nn.functional.cosine_similarity(img_feat, text_vec).mean()
            loss = cosine_loss(img_feat, text_vec)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_sim += sim.item()

        avg_loss = epoch_loss / len(dataloader)
        avg_sim = epoch_sim / len(dataloader)
        losses.append(avg_loss)
        similarities.append(avg_sim)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Sim={avg_sim:.4f}")

        if avg_sim > best_sim:
            best_sim = avg_sim
            torch.save({"state_dict": cocoop.state_dict()}, save_path)
            print(f"  → Saved best checkpoint")

    # Plot training curves
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Loss (1 - cosine)")
    plt.plot(similarities, label="Cosine Similarity")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("CoCoOp Training")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"\nPlot saved: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--ann", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--plot", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    args = parser.parse_args()
    
    train(
        images_dir=args.images,
        ann_csv=args.ann,
        save_path=args.save,
        plot_path=args.plot,
        prompt=args.prompt,
        epochs=args.epochs,
        lr=args.lr,
        config_path=args.config,
        ckpt_path=args.weights
    )