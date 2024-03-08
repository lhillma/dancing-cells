from collections import defaultdict

import click
import numpy as np

from monai.losses.dice import DiceLoss
from torch.nn import BCEWithLogitsLoss

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from models import ResUNet
from monai_dataset import get_dataset

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score



def load_model_snapshot(model_path, device):
    model = ResUNet(n_classes=1, in_channels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model



def eval_model_with_metrics(model: torch.nn.Module, data_loader: DataLoader, device: torch.device):

    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for data in data_loader:
            images, labels = data['image'].to(device), data['label'].to(device)
            outputs = model(images)
            # binary classification so assumed to have a sigmoid output
            preds = torch.sigmoid(outputs).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    # Convert lists to arrays for metric computation
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Binarize predictions for accuracy, F1, precision, and recall
    binarized_predictions = np.round(predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, binarized_predictions)
    f1 = f1_score(true_labels, binarized_predictions)
    precision = precision_score(true_labels, binarized_predictions)
    recall = recall_score(true_labels, binarized_predictions)
    auc = roc_auc_score(true_labels, predictions)
    average_precision = average_precision_score(true_labels, predictions)
    
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"AUC: {auc}")
    print(f"Average Precision: {average_precision}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(true_labels, predictions)
    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, label='Precision-Recall curve (area = %0.2f)' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

    # Return metrics in a dictionary for potential further use
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'average_precision': average_precision,
    }


def print_metrics(metrics: dict[str, float]):
    for k, v in metrics.items():
        print(f"{k}: {v:4f}")


def evaluate_snapshot(model_path, data_loader, device):
    model = load_model_snapshot(model_path, device)
    eval_metrics = eval_model_with_metrics(model, data_loader, device)
    print_metrics(eval_metrics)
    return eval_metrics




if __name__ == "__main__":
    main()
