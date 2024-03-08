import numpy as np

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def eval_model_with_metrics(
    model: torch.nn.Module, data_loader: DataLoader, device: torch.device
):
    model.eval()
    predictions, true_labels = [], []

    cluster_predictions = {}
    cluster_true_labels = {}
    with torch.no_grad():
        for data in tqdm(data_loader):
            data_dict = data[0] if isinstance(data, list) else data
            image = data_dict["image"].to(device).float()
            nucleus = data_dict["nucleus"].to(device).float()
            labels = data_dict["label"].float()
            cluster_ids = data_dict["cluster_id"].int().cpu().numpy()
            unique_cluster_ids = sorted(np.unique(cluster_ids))
            model_input = torch.cat((image, nucleus), dim=0)
            outputs = model(model_input.unsqueeze(0)).squeeze(0).cpu()
            preds = torch.sigmoid(outputs)

            for unique_cluster_id in unique_cluster_ids:  # unique_cluster_ids:
                bool_array = np.where(cluster_ids == unique_cluster_id, 1, 0)
                mean_pred = np.mean(preds[bool_array == 1.0])
                mean_label = np.mean(labels[bool_array == 1.0])
                if unique_cluster_id not in cluster_predictions:
                    cluster_predictions[unique_cluster_id] = []
                    cluster_true_labels[unique_cluster_id] = []

                cluster_predictions[unique_cluster_id].append(mean_pred)
                cluster_true_labels[unique_cluster_id].append(mean_label)

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
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "auc": auc,
            "average_precision": average_precision,
        }

    # Here you can print or process `cluster_metrics` as needed
    return cluster_metrics


def print_metrics(metrics: dict[str, float]):
    for k, v in metrics.items():
        print(f"{k}: {v:4f}")
