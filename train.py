import argparse, json, os, random, shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def stratified_split_from_raw(raw_dir, out_dir, train=0.7, val=0.2, test=0.1):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    assert abs(train + val + test - 1.0) < 1e-6
    splits = ["train", "val", "test"]
    for s in splits:
        for c in [d.name for d in raw_dir.iterdir() if d.is_dir()]:
            (out_dir / s / c).mkdir(parents=True, exist_ok=True)
    # copy with split
    for cls_dir in [d for d in raw_dir.iterdir() if d.is_dir()]:
        imgs = [p for p in cls_dir.rglob("*") if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"]]
        random.shuffle(imgs)
        n = len(imgs); n_train = int(n*train); n_val = int(n*val)
        parts = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train+n_val],
            "test": imgs[n_train+n_val:]
        }
        for split, files in parts.items():
            for f in files:
                dst = out_dir / split / cls_dir.name / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)

def build_dataloaders(data_dir, img_size=224, batch_size=32, num_workers=4, weighted_sampler=False):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.05),
        transforms.ToTensor(),
        norm
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        norm
    ])
    ds_train = datasets.ImageFolder(Path(data_dir)/"train", transform=train_tf)
    ds_val   = datasets.ImageFolder(Path(data_dir)/"val",   transform=eval_tf)
    ds_test  = datasets.ImageFolder(Path(data_dir)/"test",  transform=eval_tf)

    # Save class mapping
    (Path(data_dir)/"class_to_idx.json").write_text(json.dumps(ds_train.class_to_idx, indent=2))

    # Weighted sampler for imbalance
    sampler = None
    if weighted_sampler:
        counts = np.bincount([y for _,y in ds_train.samples])
        weights = 1.0 / np.maximum(counts, 1)
        sample_weights = [weights[y] for _, y in ds_train.samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
                          num_workers=num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl_train, dl_val, dl_test, ds_train.class_to_idx

def build_model(arch="resnet18", num_classes=3, pretrained=True):
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Unsupported arch")
    return model

def train_one_epoch(model, loader, optimizer, scaler, device, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.append(preds.cpu().numpy()); all_labels.append(labels.cpu().numpy())
    import numpy as np
    all_preds = np.concatenate(all_preds); all_labels = np.concatenate(all_labels)
    return running_loss/total, correct/total, all_preds, all_labels

def plot_curves(history, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    # Loss
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss")
    plt.savefig(outdir/"loss.png", bbox_inches="tight"); plt.close()
    # Acc
    plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Accuracy")
    plt.savefig(outdir/"acc.png", bbox_inches="tight"); plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="dataset", help="root folder with train/val/test")
    p.add_argument("--raw", type=str, help="optional: dataset_raw/ (class folders) for auto split")
    p.add_argument("--arch", type=str, default="resnet18", choices=["resnet18","mobilenet_v3_small"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--use_sampler", action="store_true")
    p.add_argument("--freeze_backbone_epochs", type=int, default=3)
    p.add_argument("--outdir", type=str, default="runs_cls")
    p.add_argument("--device", type=str, default="auto")  # auto|cpu|cuda
    p.add_argument("--patience", type=int, default=5, help="early stopping patience")
    args = p.parse_args()

    device = torch.device("cuda" if (args.device=="cuda" or (args.device=="auto" and torch.cuda.is_available())) else "cpu")
    print("Device:", device)

    if args.raw and not Path(args.data).exists():
        print("Creating stratified splits from", args.raw)
        stratified_split_from_raw(args.raw, args.data)

    dl_train, dl_val, dl_test, class_to_idx = build_dataloaders(args.data, batch_size=args.batch, weighted_sampler=args.use_sampler)
    num_classes = len(class_to_idx)

    model = build_model(args.arch, num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    # Optionally freeze backbone for warmup
    if args.freeze_backbone_epochs > 0:
        print(f"Freezing backbone for {args.freeze_backbone_epochs} epochs…")
        backbone_params = []
        head_params = []
        for name, p in model.named_parameters():
            if any(k in name for k in ["fc","classifier.3","classifier.4","classifier.6","classifier.7","classifier.8","classifier.9","classifier.10"]):
                head_params.append(p)
            else:
                p.requires_grad = False
                backbone_params.append(p)
        warmup_optimizer = torch.optim.AdamW([p for p in head_params if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc, best_state, epochs_no_improve = 0.0, None, 0
    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        if args.freeze_backbone_epochs > 0 and epoch <= args.freeze_backbone_epochs:
            tl, ta = train_one_epoch(model, dl_train, warmup_optimizer, scaler, device, criterion)
        else:
            # unfreeze if entering fine-tune
            if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
                for p in model.parameters(): p.requires_grad = True
                print("Unfroze backbone for fine-tuning.")
            tl, ta = train_one_epoch(model, dl_train, optimizer, scaler, device, criterion)
            scheduler.step()

        vl, va, _, _ = evaluate(model, dl_val, device, criterion)
        print(f"train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}")

        history["train_loss"].append(tl); history["train_acc"].append(ta)
        history["val_loss"].append(vl);   history["val_acc"].append(va)

        # Early stopping on val acc
        if va > best_val_acc:
            best_val_acc = va
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("Early stopping.")
                break

    # Save best
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, outdir/"best_model.pth")
    (outdir/"class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2))
    plot_curves(history, outdir)

    # Evaluate on test
    model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
    _, test_acc, preds, labels = evaluate(model, dl_test, device, criterion)
    print("TEST ACCURACY:", round(test_acc, 4))

    # Reports
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    y_true = [idx_to_class[i] for i in labels]
    y_pred = [idx_to_class[i] for i in preds]
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    # Save text report
    with open(outdir/"report.txt","w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(classification_report(y_true, y_pred, digits=4))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_true, y_pred)))
    print(f"\nSaved: {outdir/'best_model.pth'}, {outdir/'class_to_idx.json'}, {outdir/'loss.png'}, {outdir/'acc.png'}, {outdir/'report.txt'}")

if __name__ == "__main__":
    main()
