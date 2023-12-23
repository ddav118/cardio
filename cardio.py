import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm
from IPython.display import display
import subprocess
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import wandb
import random
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
)
import torch.nn.functional as F
import models
import getters
import losses
import eval
from torchinfo import summary
import pprint
from collections import defaultdict

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def setup_project():
    process = subprocess.Popen(
        "pip install wandb --upgrade",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    print("output:", stdout.decode())
    print("error:", stderr.decode())
    print("returncode:", process.returncode)
    # Device configuration
    print("Pytorch Version:", torch.__version__)
    print(
        f"CUDA is {'locked and loaded! <3' if torch.cuda.is_available() else 'not cooperating! :('}"
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(0))
    return device


DEVICE = setup_project()
global sweep_config
sweep_config = {
    "method": "grid",
}
metric = {"name": "val_auprc_enl", "goal": "maximize"}
sweep_config["metric"] = metric
parameters_dict = {
    "pretrained": {"value": True},  # ImageNet pretrained
    "pretraining": {"value": True},  # Creating pretrained model
    "output": {"value": "ordinal"},  # output type: binary, ordinal, multi
    "epochs": {"value": 350},  # Number of epochs
    "batch_size": {"value": 128},  # Batch size
    "num_classes": {"value": 4},  # Number of classes, not helpful
    "original_image_res": {"value": 256},  # Original image resolution
    "image_res": {"value": 256},  # Used image resolution
    "architecture": {"values": ["efficientnet_b2"]},  # Model architecture
    "drop_rate": {"value": 0.3},  # Dropout rate
    "drop_path_rate": {"value": 0.2},  # Stochastic depth rate
    "activations": {"value": "LeakyReLU"},  # Activation function
    "BN_momentum": {"value": True},  # Batch normalization momentum
    "optimizer": {"value": "adam"},  # Optimizer
    "weight_decay": {"value": 1e-5},  # Weight decay
    "learning_rate": {"values": [1e-5]},  # Learning rate
    "loss": {"value": "BCELoss"},  # Loss function
    "seed": {"values": [42, 43, 44, 45, 46]},  # Random seed
    "patience": {"value": 35},  # Patience for early stopping
    "transfer": {"values": [True]},  # Transfer learning
    "transfer_path": {
        "value": "/home/ddavilag/cardio/models/cardiomegaly/transfer_bin/efficientnet_b2_best_256_true.pt"
    },  # Path to transfer model
    "save_dir": {
        "value": "/home/ddavilag/cardio/models/cardiomegaly/multi/"
    },  # Path to save model
}
parameters_dict["save_dir"][
    "value"
] = f'{parameters_dict["save_dir"]["value"]}{parameters_dict["output"]["value"]}/'
sweep_config["parameters"] = parameters_dict


def make(config=None):
    getters.deterministic(seed=config.seed)
    train_df, val_df, test_df = getters.get_data(dataset="aha", output=config.output)

    train_set, val_set, test_set = (
        getters.make_set(
            train_df,
            train=True,
            keys=getters.get_keys(config.output),
            ordinal=config.output in ["ordinal"],
        ),
        getters.make_set(
            val_df,
            keys=getters.get_keys(config.output, test=True),
            ordinal=config.output in ["ordinal"],
        ),
        getters.make_set(
            test_df,
            keys=getters.get_keys(config.output, test=True),
            ordinal=config.output in ["ordinal"],
        ),
    )

    train_loader = getters.make_loader(
        train_set, batch_size=config.batch_size, train=True
    )
    val_loader = getters.make_loader(val_set, batch_size=config.batch_size)
    test_loader = getters.make_loader(test_set, batch_size=config.batch_size)
    model, criterion, optimizer = getters.get_stuff(config)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    return (
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
    )


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    config,
    best_loss=np.inf,
):
    wandb.watch(
        model,
    )

    epochs_no_improve = 0  # Counter for early stopping

    for epoch in range(config.epochs):
        print("Epoch #:", epoch)

        model.train()
        total_loss = 0
        total_batches = len(train_loader)
        for _, batch in tqdm(enumerate(train_loader), total=total_batches):
            loss = train_batch(batch, model, optimizer, criterion, config)
            if config.output in ["ordinal", "ord2bin"]:
                loss = {k: v.item() for k, v in loss.items()}
                total_loss += loss["total"]
            else:
                total_loss += loss

        average_train_loss = total_loss / total_batches
        val_loss = evaluate(model, val_loader, criterion, epoch, config)
        if config.output in ["ordinal", "ord2bin"]:
            val_loss = {k: v.item() for k, v in val_loss.items()}
            wandb.log(
                {
                    "train_loss": average_train_loss,
                    "train_enl_loss": loss["enl"],
                    "train_la_loss": loss["la"],
                    "train_ra_loss": loss["ra"],
                    "train_lv_loss": loss["lv"],
                    "train_rv_loss": loss["rv"],
                    "val_loss": val_loss["total"],
                    "val_enl_loss": val_loss["enl"],
                    "val_la_loss": val_loss["la"],
                    "val_ra_loss": val_loss["ra"],
                    "val_lv_loss": val_loss["lv"],
                    "val_rv_loss": val_loss["rv"],
                },
                step=epoch,
            )
        else:
            wandb.log(
                {
                    "train_loss": average_train_loss,
                    "val_loss": val_loss,
                },
                step=epoch,
            )

        # Save the model if validation loss has improved
        if val_loss["total"] < best_loss:
            best_loss = val_loss["total"]
            epochs_no_improve = 0  # Reset the counter

            getters.ensure_directory_exists(f"{config.save_dir}")
            if epoch == 0:
                print(config.save_dir)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                f"{config.save_dir}{config.architecture}_epoch{epoch}_{config.image_res}_{config.seed}.pt",
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                f"{config.save_dir}{config.architecture}_best_{config.image_res}_{config.seed}.pt",
            )
            artifact = wandb.Artifact(
                f"best_model{config.output}_{config.seed}", type="model"
            )
            artifact.add_file(
                f"{config.save_dir}{config.architecture}_best_{config.image_res}_{config.seed}.pt"
            )
            wandb.log_artifact(artifact)
        else:
            epochs_no_improve += 1

        # Early stopping check
        if epochs_no_improve == config.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs!")

            break


def train_batch(batch, model, optimizer, criterion, config):
    if config.output == "continuous":
        images, enl, edema = batch
        images, enl, edema = (
            images.to(DEVICE, non_blocking=True),
            enl.to(DEVICE, non_blocking=True),
            edema.to(DEVICE, non_blocking=True),
        )

        # Forward pass ➡
        outputs = model(images)
        outputs = outputs.squeeze()
        # Calculate loss
        loss = criterion(outputs, enl)
        optimizer.zero_grad()
        # Backward pass ⬅
        loss.backward()
    if (config.output == "binary") | (config.output == "ord2binenl"):
        images, enl = batch["img"], batch["enl"]
        images, enl = images.to(DEVICE, non_blocking=True), enl.to(
            DEVICE, non_blocking=True
        )

        # Forward pass ➡
        outputs = model(images)
        outputs = outputs.squeeze()  # Flatten the outputs to match the target's shape

        outputs = torch.sigmoid(outputs)  # Apply sigmoid to the outputs
        # Calculate loss
        loss = criterion(outputs, enl)
        optimizer.zero_grad()
        # Backward pass ⬅
        loss.backward()

    elif config.output == "ordinal":
        (
            images,
            enl,
            la,
            ra,
            lv,
            rv,
        ) = (
            batch["img"],
            batch["enl"],
            batch["LA"],
            batch["RA"],
            batch["LV"],
            batch["RV"],
        )

        # print(images.shape, enl.shape, la.shape, ra.shape, lv.shape, rv.shape)
        (
            images,
            enl,
            la,
            ra,
            lv,
            rv,
        ) = (
            images.to(DEVICE, non_blocking=True),
            enl.to(DEVICE, non_blocking=True),
            la.to(DEVICE, non_blocking=True),
            ra.to(DEVICE, non_blocking=True),
            lv.to(DEVICE, non_blocking=True),
            rv.to(DEVICE, non_blocking=True),
        )

        # Forward pass ➡
        outputs = model(images)
        preds = {
            "enl": torch.sigmoid(outputs[0]),
            "la": outputs[1],
            "ra": outputs[2],
            "lv": outputs[3],
            "rv": outputs[4],
        }
        loss = criterion(
            preds,
            {
                "enl": enl,
                "la": la,
                "ra": ra,
                "lv": lv,
                "rv": rv,
            },
        )
        # Backward pass ⬅
        optimizer.zero_grad()
        loss["total"].backward()

    # Step with optimizer
    optimizer.step()

    return loss


def bootstrap_results(
    y_test, y_preds, y_pred_probs, predict="enl", bootstrap_rounds=100, test=False
):
    """
    Calculate bootstrap metrics for evaluating machine learning model performance.

    Args:
        y_test: Array-like, true labels.
        y_preds: Array-like, predicted labels.
        y_pred_probs: Array-like, predicted probabilities.
        bootstrap_rounds: int, optional. Number of bootstrap rounds (default: 10000).

    Returns:
        dict: Dictionary containing bootstrap metrics for each specified metric.
    """
    if test:
        metrics = ["acc", "auroc", "auprc", "precision", "recall"]
        metric_methods = {
            "acc": accuracy_score,
            "auroc": roc_auc_score,
            "auprc": average_precision_score,
            "precision": precision_score,
            "recall": recall_score,
        }
    else:
        metrics = ["acc", "auroc", "auprc"]
        metric_methods = {
            "acc": accuracy_score,
            "auroc": roc_auc_score,
            "auprc": average_precision_score,
        }
    bootstrap_metrics = {metric: [] for metric in metrics}
    idx = np.arange(y_test.shape[0])

    for _ in range(bootstrap_rounds):
        pred_idx = np.random.choice(idx, idx.shape[0], replace=True)
        current_y_preds = y_preds[pred_idx]
        current_y_pred_probs = y_pred_probs[pred_idx]
        current_y_test = y_test[pred_idx]

        for metric in metrics:
            if metric in ["acc", "precision", "recall"]:
                bootstrap_metrics[metric].append(
                    metric_methods[metric](current_y_test, current_y_preds)
                )

            else:
                bootstrap_metrics[metric].append(
                    metric_methods[metric](current_y_test, current_y_pred_probs)
                )

    bootstrap_metrics = {
        metric: (
            np.mean(results),
            np.percentile(results, 2.5),
            np.percentile(results, 97.5),
        )
        for metric, results in bootstrap_metrics.items()
    }
    bootstrap_metrics = {
        metric: tuple([np.round(r, 3) for r in results])
        for metric, results in bootstrap_metrics.items()
    }
    print(f"{predict}", bootstrap_metrics)
    if test:
        metric_table = wandb.Table(columns=["Metric", "Mean", "Lower CI", "Upper CI"])
        for metric, (mean, lower_ci, upper_ci) in bootstrap_metrics.items():
            metric_table.add_data(metric, mean, lower_ci, upper_ci)
        wandb.log({"Bootstrap Metrics": metric_table}, step=0)

    return bootstrap_metrics


def evaluate(model, loader, criterion, epoch, config, test=False):
    model.eval()
    losses_dict = {"enl": [], "la": [], "ra": [], "lv": [], "rv": [], "total": []}
    loss = 0
    if config.output in ["ord2bin", "ordinal"]:
        y_preds = {"enl": [], "la_bin": [], "ra_bin": [], "lv_bin": [], "rv_bin": []}
        y_pred_probs = {
            "enl": [],
            "la_bin": [],
            "ra_bin": [],
            "lv_bin": [],
            "rv_bin": [],
        }
        y_true = {"enl": [], "la_bin": [], "ra_bin": [], "lv_bin": [], "rv_bin": []}
    else:
        y_preds = []
        y_pred_probs = []
        y_true = []
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            if config.output == "continuous":
                images, enl, edema = batch
                images, enl = images.to(DEVICE, non_blocking=True), enl.to(
                    DEVICE, non_blocking=True
                )
                # edema.to(DEVICE, non_blocking=True),

                outputs = model(images)
                outputs = outputs.squeeze()
                loss += criterion(outputs, enl).item()

                # aggregate outputs and labels to calculate accuracy, AUC, AUPRC
                predicted_probs = torch.sigmoid(outputs).cpu().numpy()
                y_pred_probs.extend(predicted_probs)
                y_preds.extend(predicted_probs > 0.5)
                y_true.extend(edema)

            if (config.output == "binary") | (config.output == "ord2binenl"):
                images, labels = batch["img"], batch["enl"]
                images, labels = (
                    images.to(DEVICE, non_blocking=True),
                    labels.to(DEVICE, non_blocking=True),
                )
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                outputs = outputs.squeeze()
                loss += criterion(outputs, labels).item()

                y_pred_probs.extend(outputs.cpu().numpy())
                y_preds.extend(outputs.cpu().numpy() > 0.5)
                y_true.extend(labels.cpu().numpy())
            elif config.output == "ordinal":
                images, enl, la, ra, lv, rv, la_bin, ra_bin, lv_bin, rv_bin = (
                    batch["img"],
                    batch["enl"],
                    batch["LA"],
                    batch["RA"],
                    batch["LV"],
                    batch["RV"],
                    batch["LA_bin"],
                    batch["RA_bin"],
                    batch["LV_bin"],
                    batch["RV_bin"],
                )
                (
                    images,
                    enl,
                    la,
                    ra,
                    lv,
                    rv,
                ) = (
                    images.to(DEVICE, non_blocking=True),
                    enl.to(DEVICE, non_blocking=True),
                    la.to(DEVICE, non_blocking=True),
                    ra.to(DEVICE, non_blocking=True),
                    lv.to(DEVICE, non_blocking=True),
                    rv.to(DEVICE, non_blocking=True),
                )
                outputs = model(images)

                preds = {
                    "enl": torch.sigmoid(outputs[0]),
                    "la": outputs[1],
                    "ra": outputs[2],
                    "lv": outputs[3],
                    "rv": outputs[4],
                }
                losses_vals = criterion(
                    preds,
                    {
                        "enl": enl,
                        "la": la,
                        "ra": ra,
                        "lv": lv,
                        "rv": rv,
                    },
                )
                loss += losses_vals["total"].item()
                for component in ["enl", "la", "ra", "lv", "rv", "total"]:
                    losses_dict[component].append(losses_vals[component].item())
                y_true["enl"].extend(enl.cpu().numpy())
                y_true["la_bin"].extend(la_bin.numpy())
                y_true["ra_bin"].extend(ra_bin.numpy())
                y_true["lv_bin"].extend(lv_bin.numpy())
                y_true["rv_bin"].extend(rv_bin.numpy())
                preds["enl"] = torch.sigmoid(preds["enl"])
                y_pred_probs["la_bin"].extend(losses.prediction2bin(preds["la"].cpu()))
                y_pred_probs["ra_bin"].extend(losses.prediction2bin(preds["ra"].cpu()))
                y_pred_probs["lv_bin"].extend(losses.prediction2bin(preds["lv"].cpu()))
                y_pred_probs["rv_bin"].extend(losses.prediction2bin(preds["rv"].cpu()))
                y_pred_probs["enl"].extend(preds["enl"].cpu().numpy())
                y_preds["la_bin"].extend(np.round(y_pred_probs["la_bin"]))
                y_preds["ra_bin"].extend(np.round(y_pred_probs["ra_bin"]))
                y_preds["lv_bin"].extend(np.round(y_pred_probs["lv_bin"]))
                y_preds["rv_bin"].extend(np.round(y_pred_probs["rv_bin"]))
                y_preds["enl"].extend(np.round(y_pred_probs["enl"]))

            elif config.output == "ord2bin":
                (images, enl, la, ra, lv, rv) = batch

                images, la, ra, lv, rv = (
                    images.to(DEVICE, non_blocking=True),
                    la.to(DEVICE, non_blocking=True),
                    ra.to(DEVICE, non_blocking=True),
                    lv.to(DEVICE, non_blocking=True),
                    rv.to(DEVICE, non_blocking=True),
                )

                outputs = model(images)
                preds = {
                    "la_bin": torch.sigmoid(outputs[0]),
                    "ra_bin": torch.sigmoid(outputs[1]),
                    "lv_bin": torch.sigmoid(outputs[2]),
                    "rv_bin": torch.sigmoid(outputs[3]),
                }
                losses_vals = criterion(
                    preds,
                    {
                        "la_bin": la,
                        "ra_bin": ra,
                        "lv_bin": lv,
                        "rv_bin": rv,
                    },
                )

                y_true["la_bin"].extend(la.cpu().numpy())
                y_true["ra_bin"].extend(ra.cpu().numpy())
                y_true["lv_bin"].extend(lv.cpu().numpy())
                y_true["rv_bin"].extend(rv.cpu().numpy())
                pred_probs, preds = get_component_preds(preds)
                y_pred_probs["la_bin"].extend(pred_probs["la_bin"])
                y_pred_probs["ra_bin"].extend(pred_probs["ra_bin"])
                y_pred_probs["lv_bin"].extend(pred_probs["lv_bin"])
                y_pred_probs["rv_bin"].extend(pred_probs["rv_bin"])
                y_preds["la_bin"].extend(preds["la_bin"])
                y_preds["ra_bin"].extend(preds["ra_bin"])
                y_preds["lv_bin"].extend(preds["lv_bin"])
                y_preds["rv_bin"].extend(preds["rv_bin"])

    # log metrics from the ord2bin model

    if config.output == "ordinal":
        for component in ["enl", "la_bin", "ra_bin", "lv_bin", "rv_bin"]:
            if test:
                metrics = bootstrap_results(
                    np.array(y_true[component]),
                    np.array(y_preds[component]),
                    np.array(y_pred_probs[component]),
                    bootstrap_rounds=10000,
                    predict=component,
                    test=True,
                )

            else:
                metrics = bootstrap_results(
                    np.array(y_true[component]),
                    np.array(y_preds[component]),
                    np.array(y_pred_probs[component]),
                    predict=component,
                )
                for metric, (mean, lower_ci, upper_ci) in metrics.items():
                    # if test:
                    #     wandb.log(
                    #         {
                    #             f"test_{metric}_{component}": mean,
                    #             f"test_{metric}_{component}_ci_lower": lower_ci,
                    #             f"test_{metric}_{component}_ci_upper": upper_ci,
                    #         },
                    #         step=epoch,
                    #     )
                    wandb.log(
                        {
                            f"val_{metric}_{component}": mean,
                            f"val_{metric}_{component}_ci_lower": lower_ci,
                            f"val_{metric}_{component}_ci_upper": upper_ci,
                        },
                        step=epoch,
                    )
    elif config.output == "binary":
        metrics = bootstrap_results(
            np.array(y_true), np.array(y_preds), np.array(y_pred_probs)
        )

        # Log metrics with confidence intervals
        for metric, (mean, lower_ci, upper_ci) in metrics.items():
            if test:
                wandb.log(
                    {
                        f"test_{metric}": mean,
                        f"test_{metric}_ci_lower": lower_ci,
                        f"test_{metric}_ci_upper": upper_ci,
                    },
                    step=epoch,
                )
            else:
                wandb.log(
                    {
                        f"val_{metric}": mean,
                        f"val_{metric}_ci_lower": lower_ci,
                        f"val_{metric}_ci_upper": upper_ci,
                    },
                    step=epoch,
                )
    if test:
        return y_pred_probs, y_true
    else:
        # Calculate and log metrics

        losses_dict = {k: np.mean(v) for k, v in losses_dict.items()}

        return losses_dict if config.output in ["ordinal", "ord2bin"] else loss


def model_pipeline(hparams=None):
    # tell wandb to get started
    # savedir = f"{'scratch' if hparams['pretrained']==False else 'pre'}_{'bin' if hparams['output']=='binary' else 'continuous'}"
    with wandb.init(
        # set the wandb project where this run will be logged
        # project="cardiomegaly sweep",
        # track hyperparameters and run metadata
        # name=f"{savedir}_{hparams['architecture']}_{hparams['image_res']}",
        notes="Ordinal Model - 5 Outputs: (LA, RA, LV, RV, ENL))",
        # magic=True,
        config=hparams,
    ) as run:
        config = run.config
        wandb.save("cardio.py")
        wandb.save("models.py")
        wandb.save("getters.py")
        wandb.save("losses.py")
        # wandb.save("trainer.py")
        wandb.save("AHA.py")
        wandb.save('eval.py')

        # make the model, data, and optimization
        (
            model,
            train_loader,
            val_loader,
            test_loader,
            criterion,
            optimizer,
        ) = make(config)

        # and use them to train the model
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
        )
        checkpoint = torch.load(
            f"{config.save_dir}{config.architecture}_best_{config.image_res}_{config.seed}.pt"
        )
        fixed_state_dict = {
            k.replace("model.", ""): v for k, v in checkpoint["model_state_dict"].items()
        }
        model.load_state_dict(fixed_state_dict)
        model.eval()
        preds, prob_preds, targets, gender = eval.test(
            model,
            test_loader,
            config,
            target_keys=["LA_bin", "RA_bin", "LV_bin", "RV_bin", "enl"],
        )
        boot_dict = defaultdict(list)
        fig_dict = defaultdict(list)
        for key in preds.keys():
            boot_dict[key], fig_dict[key] = eval.bootstrap_results(
            targets[key],
            preds[key],
            prob_preds[key],
            predict=key,
            bootstrap_rounds=1000,
            test=True,
        )

        wandb.finish()


sweep_id = wandb.sweep(sweep_config, project="cardiomegaly sweep")
wandb.agent(sweep_id, function=model_pipeline, project="cardiomegaly sweep")
