import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    """Basic image dataset for training without bounding boxes."""
    
    def __init__(self, root, csv_path, transform=None):
        df = pd.read_csv(csv_path)
        self.paths = [os.path.join(root, name) for name in df["image_name"]]
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((800, 800)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


class SemiSupervisedDataset(Dataset):
    """Dataset with optional bounding boxes for hybrid training."""
    
    def __init__(self, root, csv_path, labeled=True, strong_aug=False):
        df = pd.read_csv(csv_path)
        if labeled:
            df = df.dropna(subset=["xmin"])
        
        self.paths = []
        self.boxes = []
        for _, row in df.iterrows():
            self.paths.append(os.path.join(root, row["image_name"]))
            if labeled:
                self.boxes.append(torch.tensor(
                    [row["xmin"], row["ymin"], row["xmax"], row["ymax"]], 
                    dtype=torch.float32
                ))
            else:
                self.boxes.append(None)
        
        self.weak_aug = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.strong_aug = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.use_strong = strong_aug

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        transform = self.strong_aug if self.use_strong else self.weak_aug
        return transform(img), self.boxes[idx]