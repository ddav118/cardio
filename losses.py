import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import glob
import torch.nn.functional as F
from tqdm import tqdm
import timm
from IPython.display import display
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset


class OrdinalRegressionLoss(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLoss, self).__init__()
        self.loss_ft = nn.MSELoss(reduction="none")

    def forward(self, pred, gt):
        pred = torch.sigmoid(pred)
        losses = self.loss_ft(pred, gt).sum(axis=1)
        return losses.mean()


class CoGOLLoss(nn.Module):
    def __init__(self, alpha=0.01, beta=0.05):
        super(CoGOLLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets, weights, deltas):
        # logits: The output from the ordinal regression head
        # targets: The true class labels
        # weights: The weight vectors (w_j)
        # deltas: The class-specific adjustments (delta_j)

        batch_size = logits.size(0)
        num_classes = logits.size(1) + 1  # Adding 1 because logits are for k-1 classes

        # Creating the cumulative logits
        cum_logits = torch.cat(
            [torch.zeros(batch_size, 1, device=logits.device), logits], dim=1
        )

        # Loss for each sample
        loss = 0
        for i in range(batch_size):
            # print(targets[i])
            for j in range(1, targets[i]):
                loss += torch.log(torch.sigmoid(-cum_logits[i, j - 1]))
            for j in range(targets[i], num_classes - 1):
                loss += torch.log(torch.sigmoid(cum_logits[i, j]))

        # Regularization terms
        reg_loss_w = self.alpha * sum(torch.norm(w) ** 2 for w in weights) / 2
        reg_loss_delta = (
            self.beta * sum(torch.norm(d) ** 2 for d in deltas[1:]) / 2
        )  # Skipping the first delta

        # Total loss
        total_loss = -loss / batch_size + reg_loss_w + reg_loss_delta
        return total_loss


def map_to_class(logits, threshold=0.5):
    """
    Maps the output logits from the model to an ordinal class (1-4) for a batch of outputs.

    Args:
    logits (torch.Tensor): The output logits from the model for a batch. Shape: [batch_size, num_classes-1]
    threshold (float): The threshold to determine the class.

    Returns:
    torch.Tensor: The predicted classes for the batch. Shape: [batch_size]
    """
    # Apply sigmoid to convert logits to probabilities
    probabilities = torch.sigmoid(logits)

    # Calculate cumulative probabilities
    cumulative_probabilities = torch.cumprod(probabilities, dim=1)

    # Determine the class for each instance in the batch
    classes = torch.sum(cumulative_probabilities > threshold, dim=1)

    return classes


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        tasks: list,
        loss_ft,
        loss_weights: dict,
    ):
        super(MultiTaskLoss, self).__init__()
        # assert set(tasks) == set(loss_ft.keys())
        # assert set(tasks) == set(loss_weights.keys())
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt):
        out = {}
        for task in self.tasks:
            if task == "enl":
                # pred[task] = pred[task].squeeze()
                gt[task] = gt[task].unsqueeze(1)
                out[task] = self.loss_ft[task](pred[task], gt[task])

            else:
                # pred[task] = pred[task].squeeze()
                out[task] = self.loss_ft[task](pred[task], gt[task])

                # if task == "enl":
                #     pred[task] = pred[task].squeeze()
                #     out[task] = self.loss_ft[task](pred[task], gt[task])
                # elif task == "la":
                #     out[task] = self.loss_ft[task](
                #         pred[task],
                #         gt[task],
                #         self.model.LA_ord.weight,
                #         self.model.LA_ord.bias,
                #     )
                # elif task == "ra":
                #     out[task] = self.loss_ft[task](
                #         pred[task],
                #         gt[task],
                #         self.model.RA_ord.weight,
                #         self.model.RA_ord.bias,
                #     )
                # elif task == "lv":
                #     out[task] = self.loss_ft[task](
                #         pred[task],
                #         gt[task],
                #         self.model.LV_ord.weight,
                #         self.model.LV_ord.bias,
                #     )
                # elif task == "rv":
                #     out[task] = self.loss_ft[task](
                #         pred[task],
                #         gt[task],
                #         self.model.RV_ord.weight,
                #         self.model.RV_ord.bias,
                #     )
                # else:
                # pred[task] = pred[task].squeeze()
                # out[task] = self.loss_ft[task](pred[task], gt[task])
            # print(out[task], out[task].shape)
        out["total"] = torch.sum(
            torch.stack([self.loss_weights[t] * out[t] for t in self.tasks])
        )
        return out


def get_loss(task):
    # task is one of "enl", "LVIDd", "LA", "combo"

    tasks = ["enl", "la_bin", "ra_bin", "lv_bin", "rv_bin"]
    if task == "LA":
        tasks.append("la")
    elif task == "LVIDd":
        tasks.append("lvidd")
    elif task == "combo":
        tasks.extend(("lvidd", "la"))

    criterion = MultiTaskLoss(
        tasks=tasks,
        loss_ft=nn.ModuleDict(
            {
                "enl": nn.BCELoss(),
                "la_bin": nn.BCELoss(),
                "ra_bin": nn.BCELoss(),
                "lv_bin": nn.BCELoss(),
                "rv_bin": nn.BCELoss(),
                "la": nn.L1Loss(),
                "lvidd": nn.L1Loss(),
            }
        ),
        loss_weights={
            "enl": 1.6,
            "la_bin": 1.5,
            "ra_bin": 1,
            "lv_bin": 1.5,
            "rv_bin": 1,
            "la": 1.5,
            "lvidd": 1.5,
        },
    )

    return criterion


def prediction2label(pred: np.ndarray):
    """Convert ordinal predictions to class labels, e.g.

    [0.9, 0.1, 0.1, 0.1] -> 0
    [0.9, 0.9, 0.1, 0.1] -> 1
    [0.9, 0.9, 0.9, 0.1] -> 2
    etc.
    """

    return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1


# prediction2label(np.array([[0.9, 0.1, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1], [0.9, 0.9, 0.9, 0.1]]))


# def prediction2bin(pred: np.ndarray, enlargement_threshold=2):
#     """
#     Convert ordinal predictions to binary chamber enlargement probability.

#     Args:
#     pred (np.ndarray): Array of ordinal predictions.
#     enlargement_threshold (int): The class index from which enlargement is considered.

#     Returns:
#     torch.Tensor: Binary probabilities of chamber enlargement.
#     """
#     predictions = torch.sigmoid(torch.from_numpy(pred))
#     normalized = predictions / predictions.sum(axis=1, keepdims=True)
#     cum_probs = torch.cumsum(normalized, dim=1)

#     # Get the cumulative probability up to the enlargement threshold
#     return cum_probs[:, enlargement_threshold]


# def prediction2bin(pred):
#     """
#     Convert ordinal predictions to binary chamber enlargement probability.

#     Args:
#     pred (np.ndarray): Array of ordinal predictions.

#     Returns:
#     np.ndarray: Binary probabilities of chamber enlargement as a numpy array.
#     """
#     # Convert numpy array to torch tensor
#     try:
#         pred_tensor = torch.from_numpy(pred)
#     except TypeError:
#         pred_tensor = pred

#     # Apply softmax to convert logits to probabilities
#     probabilities = torch.sigmoid(pred_tensor)

#     predictions = probabilities[:, 1]
#     print(predictions)
#     return predictions


def prediction2bin(pred):
    """
    Convert ordinal predictions to binary chamber enlargement probability.

    Args:
    pred (np.ndarray): Array of ordinal predictions.

    Returns:
    np.ndarray: Binary probabilities of chamber enlargement as a numpy array.
    """
    # Convert numpy array to torch tensor
    try:
        pred_tensor = torch.from_numpy(pred)
    except TypeError:
        pred_tensor = pred

    # Apply sigmoid to convert logits to probabilities
    probabilities = torch.sigmoid(pred_tensor)

    return probabilities[:, 1].detach().numpy()

prediction2bin(np.array([[0.9, 0.1, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1], [0.9, 0.9, 0.9, 0.1]]))


# class OrdinalRegressionLoss(nn.Module):
#     def __init__(self):
#         super(OrdinalRegressionLoss, self).__init__()
#         self.loss_ft = nn.MSELoss(reduction="none")

#     def forward(self, pred, gt):
#         pred = torch.sigmoid(pred)
#         return self.loss_ft(pred, gt).sum(axis=1)


# ordinal_regression(
#     torch.tensor([[0.8, 0.7, 0.8, 0.3], [0.9, 1, 0, 0]], dtype=torch.float32),
#     torch.tensor(
#         [[1, 0, 0, 0], [1, 1, 1, 1]],
#         dtype=torch.float32,
#     ),
# )
