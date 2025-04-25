import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR
from config import *  
from models import SimpleCNN
from losses import LMFLoss
from data.dataset_utils import get_class_sample_counts
from utils import setup_logger
from utils.val_metrics import accuracy, precision_per_class, recall_per_class, macro_f1
import datetime
import os

def get_time_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean = [0.6687, 0.5303, 0.5247]
    std = [0.2227, 0.2022, 0.2130]

    train_transform = transforms.Compose([
        transforms.Resize(model_input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(model_input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    num_classes = len(train_dataset.classes)

    cls_num_list = get_class_sample_counts(train_dataset).to(device)

    class_counts = cls_num_list.float()
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(device)

    batch_size_ = globals().get('batch_size', 32)
    num_epochs_ = globals().get('num_epochs', 50)
    learning_rate_ = globals().get('learning_rate', 1e-3)
    alpha_ = globals().get('alpha', 0.5)
    beta_ = globals().get('beta', 0.5)
    gamma_ = globals().get('gamma', 0.5)
    max_m_ = globals().get('max_m', 0.5)
    s_ = globals().get('s', 30)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_, shuffle=False, num_workers=4, pin_memory=True)

    model = SimpleCNN(num_classes=num_classes)
    model_name = model.__class__.__name__
    model = model.to(device)

    criterion = LMFLoss(cls_num_list, device, weight=class_weights,
                       alpha=alpha_, beta=beta_, gamma=gamma_, max_m=max_m_, s=s_, add_LDAM_weigth=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    log_dir = os.path.join(result_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(model_name=model_name, log_dir=log_dir)

    best_val_acc = 0.0

    for epoch in range(1, num_epochs_ + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_running_loss = 0.0
        val_total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)

                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * targets.size(0)
                val_total += targets.size(0)

                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss = val_running_loss / val_total
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        val_acc = accuracy(all_targets, all_preds)
        labels = list(range(num_classes))
        val_prec_each = precision_per_class(all_targets, all_preds, labels)
        val_rec_each = recall_per_class(all_targets, all_preds, labels)
        val_macro_f1 = macro_f1(all_targets, all_preds, labels)

        class_names = train_dataset.classes
        prec_str = ", ".join([f"{name}: {p:.3f}" for name, p in zip(class_names, val_prec_each)])
        rec_str = ", ".join([f"{name}: {r:.3f}" for name, r in zip(class_names, val_rec_each)])

        logger.info(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val Precision (per class): [{prec_str}] | "
            f"Val Recall (per class): [{rec_str}] | "
            f"Val Macro F1: {val_macro_f1:.4f}"
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch:02d} learning rate: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            time_str = get_time_str()

            model_dir = os.path.join(result_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            save_path = os.path.join(model_dir, f"bestmodel_{model_name}_{time_str}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"New best model saved: {save_path} with Val Acc: {best_val_acc:.4f}")

    logger.info(f"Finished training. Best val acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()