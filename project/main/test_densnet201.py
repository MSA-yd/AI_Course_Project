import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from utils.val_metrics import accuracy, precision_per_class, recall_per_class, macro_f1
from utils import setup_logger
import config  

def get_time_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_roc_curves(y_true, y_scores, classes, save_path=None):
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curves(y_true, y_scores, classes, save_path=None):
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        ap_score = average_precision_score(y_true_bin[:, i], y_scores[:, i])
        plt.plot(recall, precision, lw=2, label=f'{class_name} (AP = {ap_score:.3f})')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multi-class Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    model_path = '/home/user/Desktop/junxie9/skin_cancer_4486/result/model/bestmodel_DenseNet201_84_20250425_144537.pth'  
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    time_str = get_time_str()
    logger = setup_logger(model_name=f"{model_name}_{time_str}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dir = config.test_dir  
    output_dir = config.result_dir  

    input_size = config.model_input_size[0] if hasattr(config, 'model_input_size') else 84

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_dataset = ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(test_dataset.classes)
    class_names = test_dataset.classes

    model = models.densenet201(pretrained=False)

    model.features.conv0 = torch.nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=7,
        stride=1,
        padding=3,
        bias=False
    )
    torch.nn.init.kaiming_normal_(model.features.conv0.weight, mode='fan_out', nonlinearity='relu')

    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

    model = model.to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_key = k[7:]
        else:
            new_key = k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)

    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    labels = list(range(num_classes))

    test_acc = accuracy(all_targets, all_preds)
    test_prec_each = precision_per_class(all_targets, all_preds, labels)
    test_rec_each = recall_per_class(all_targets, all_preds, labels)
    test_macro_f1 = macro_f1(all_targets, all_preds, labels)

    prec_str = ", ".join([f"{name}: {p:.3f}" for name, p in zip(class_names, test_prec_each)])
    rec_str = ", ".join([f"{name}: {r:.3f}" for name, r in zip(class_names, test_rec_each)])

    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test Precision (per class): [{prec_str}]")
    logger.info(f"Test Recall (per class): [{rec_str}]")
    logger.info(f"Test Macro F1: {test_macro_f1:.4f}")

    cm_dir = os.path.join(output_dir, 'confusion_matrix')
    roc_dir = os.path.join(output_dir, 'roc_curve')
    pr_dir = os.path.join(output_dir, 'precision_recall_curve')

    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(roc_dir, exist_ok=True)
    os.makedirs(pr_dir, exist_ok=True)

    cm_save_path = os.path.join(cm_dir, f'confusion_matrix_{model_name}_{time_str}.png')
    plot_confusion_matrix(all_targets, all_preds, class_names, save_path=cm_save_path)
    logger.info(f"Confusion matrix saved to {cm_save_path}")

    roc_save_path = os.path.join(roc_dir, f'roc_curve_{model_name}_{time_str}.png')
    plot_roc_curves(all_targets, all_probs, class_names, save_path=roc_save_path)
    logger.info(f"ROC curve saved to {roc_save_path}")

    pr_save_path = os.path.join(pr_dir, f'precision_recall_curve_{model_name}_{time_str}.png')
    plot_precision_recall_curves(all_targets, all_probs, class_names, save_path=pr_save_path)
    logger.info(f"Precision-Recall curve saved to {pr_save_path}")

if __name__ == "__main__":
    main()