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

from models import ResUNet
from cell2image import image as cimg



def load_model_snapshot(model_path, device):
    model = ResUNet(n_classes=1, in_channels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model



def eval_model_with_metrics(model: torch.nn.Module, data_loader: DataLoader, device: torch.device):

    model.eval()
    predictions, true_labels = [], []

    cluster_predictions = {}
    cluster_true_labels = {}
    with torch.no_grad():
       
       for data in data_loader:

            data_dict = data[0] if isinstance(data, list) else data
            image = data_dict["image"].to(device).float()
            nucleus = data_dict["nucleus"].to(device).float()
            labels = data_dict["label"].to(device).float()
            cluster_ids = data_dict["cluster_id"].cpu().numpy()  # Assuming cluster_id is accessible and batched

            model_input = torch.cat((image, nucleus), dim=1)
            outputs = model(model_input)
            preds = torch.sigmoid(outputs)
           
            for i, cluster_id in enumerate(cluster_ids):
                preds_flat = preds[i].view(-1).cpu().numpy()
                labels_flat = labels[i].view(-1).cpu().numpy()

                if cluster_id not in cluster_predictions:
                    cluster_predictions[cluster_id] = []
                    cluster_true_labels[cluster_id] = []

                cluster_predictions[cluster_id].extend(preds_flat)
                cluster_true_labels[cluster_id].extend(labels_flat)



    # Compute metrics for each cluster
    cluster_metrics = {}
    for cluster_id in cluster_predictions:
        predictions = np.array(cluster_predictions[cluster_id])
        true_labels = np.array(cluster_true_labels[cluster_id])

        # Binarize predictions for accuracy, F1, precision, and recall
        binarized_predictions = np.round(predictions)

        accuracy = accuracy_score(true_labels, binarized_predictions)
        f1 = f1_score(true_labels, binarized_predictions, zero_division=1)
        precision = precision_score(true_labels, binarized_predictions, zero_division=1)
        recall = recall_score(true_labels, binarized_predictions, zero_division=1)
        auc = roc_auc_score(true_labels, predictions)
        average_precision = average_precision_score(true_labels, predictions)

        cluster_metrics[cluster_id] = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'average_precision': average_precision,
        }

    # Here you can print or process `cluster_metrics` as needed
    return cluster_metrics


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
