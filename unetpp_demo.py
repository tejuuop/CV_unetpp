import os
import csv
import json
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

# --- MODEL ARCHITECTURE: Tiny UNet++ (Nested Skip Connections) ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class TinyUNetPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()
        nb_filter = [16, 32, 64]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder
        self.conv0_0 = DoubleConv(1, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])

        # UNet++ Nested Skip Connections (The signature feature of the paper)
        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_2 = DoubleConv(nb_filter[0]*2 + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], 1, 1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        # Deep supervision (averaging outputs from nested nodes as per the paper)
        out1 = self.final(x0_1)
        out2 = self.final(x0_2)
        return (out1 + out2) / 2

# --- UTILITIES ---
def dice_loss(logits, targets):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    denominator = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return (1.0 - (2.0 * intersection + 1) / (denominator + 1)).mean()

def combined_loss(logits, masks):
    return F.binary_cross_entropy_with_logits(logits, masks) + dice_loss(logits, masks)

def compute_metrics(logits, masks):
    preds = (torch.sigmoid(logits) > 0.5).float()
    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum() - intersection
    return {
        "iou": (intersection / (union + 1e-6)).item(),
        "dice": ((2.0 * intersection) / (preds.sum() + masks.sum() + 1e-6)).item(),
        "pixel_acc": (preds == masks).float().mean().item()
    }

def get_loaders(args):
    tf = transforms.ToTensor()
    train_ds = Subset(datasets.MNIST('./data', True, download=True, transform=tf), range(2048))
    test_ds = Subset(datasets.MNIST('./data', False, download=True, transform=tf), range(256))
    return DataLoader(train_ds, batch_size=args.batch, shuffle=True), DataLoader(test_ds, batch_size=args.batch)

def evaluate(model, loader, device):
    model.eval()
    t_loss, t_iou, t_dice, t_acc, n = 0, 0, 0, 0, 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs, masks = imgs.to(device), (imgs > 0.5).float().to(device)
            logits = model(imgs)
            b = imgs.size(0)
            t_loss += combined_loss(logits, masks).item() * b
            m = compute_metrics(logits, masks)
            t_iou += m["iou"] * b; t_dice += m["dice"] * b; t_acc += m["pixel_acc"] * b
            n += b
    return {"loss": t_loss/n, "iou": t_iou/n, "dice": t_dice/n, "pixel_acc": t_acc/n}

def save_images(model, loader, device, folder_path):
    model.eval()
    rows = []
    with torch.no_grad():
        for imgs, _ in loader:
            for img in imgs[:6]:
                x = img.unsqueeze(0).to(device)
                pred = (torch.sigmoid(model(x)) > 0.5).float().cpu().squeeze(0)
                mask = (img > 0.5).float()
                rows.extend([img, mask, pred])
            break
    save_image(make_grid(rows, nrow=3, padding=2), os.path.join(folder_path, "segmentation_results.png"))
    print(f"Saved segmentation_results.png in {folder_path}")

# --- MAIN LOGIC ---
def train(args):
    run_dir = os.path.join("runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}. Saving to {run_dir}")

    train_loader, test_loader = get_loaders(args)
    model, opt = TinyUNetPlusPlus().to(device), torch.optim.Adam(TinyUNetPlusPlus().parameters(), lr=1e-3)

    # Fix optimizer to use actual model parameters
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = []
    best_iou = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, n = 0, 0
        for imgs, _ in train_loader:
            imgs, masks = imgs.to(device), (imgs > 0.5).float().to(device)
            opt.zero_grad()
            loss = combined_loss(model(imgs), masks)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * imgs.size(0); n += imgs.size(0)

        val_m = evaluate(model, test_loader, device)
        row = {"epoch": epoch, "train_loss": ep_loss/n, "val_iou": val_m["iou"], "val_dice": val_m["dice"]}
        history.append(row)
        print(f"Epoch {epoch}: Loss {ep_loss/n:.4f} | Val IoU {val_m['iou']:.4f}")

        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_metrics": val_m},
                       os.path.join(run_dir, "best.pkl"))
            print(f"Saved best.pkl at epoch {epoch}")

    with open(os.path.join(run_dir, "epoch_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)

    print("\nTraining complete. Run the --eval command with this folder to show her the results.")
    return run_dir

def evaluate_live(args):
    if not args.run_dir or not os.path.exists(os.path.join(args.run_dir, "best.pkl")):
        print("Error: Provide a valid --run-dir containing best.pkl")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNetPlusPlus().to(device)

    print(f"Loading weights from {os.path.join(args.run_dir, 'best.pkl')}...")
    ckpt = torch.load(os.path.join(args.run_dir, "best.pkl"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, test_loader = get_loaders(args)
    metrics = evaluate(model, test_loader, device)

    print("\n--- LIVE EVALUATION METRICS (Using best.pkl) ---")
    print(f"Best Epoch Stored: {ckpt['epoch']}")
    for k, v in metrics.items(): print(f"{k.upper()}: {v:.4f}")

    with open(os.path.join(args.run_dir, "live_eval_metrics.json"), "w") as f:
        json.dump({"checkpoint_loaded": "best.pkl", "metrics": metrics}, f, indent=4)

    save_images(model, test_loader, device, args.run_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--run-dir", type=str, help="Folder path for --eval")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()

    torch.manual_seed(42)
    if args.train: train(args)
    elif args.eval: evaluate_live(args)
    else: print("Use --train or --eval --run-dir <folder>")
