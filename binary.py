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
    roc_curve,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
import torch.nn.functional as F
import models
import getters
import losses
from torchinfo import summary
import pprint

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
    # wandb.login()

    return device


DEVICE = setup_project()

global sweep_config
sweep_config = {
    "method": "grid",
}
metric = {"name": "val_loss", "goal": "minimize"}
sweep_config["metric"] = metric
parameters_dict = {
    "pretrained": {"value": True},  # ImageNet pretrained
    "pretraining": {"value": False},  # Creating pretrained model
    "output": {"value": "binary"},  # output type: binary, ordinal, continuous, ord2bin
    "epochs": {"value": 350},  # Number of epochs
    "batch_size": {"value": 128},  # Batch size
    "num_classes": {"value": 1},  # Number of classes, not helpful
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
    "seed": {"values": [42]},  # Random seed
    "patience": {"value": 35},  # Patience for early stopping
    "transfer": {"values": [True]},  # Transfer learning
    "transfer_path": {
        "value": "/home/ddavilag/cardio/models/cardiomegaly/transfer_bin/efficientnet_b2_best_256_true.pt"
    },  # Path to transfer model
    "save_dir": {
        "value": "/home/ddavilag/cardio/models/cardiomegaly/"
    },  # Path to save model
    "task": {"values": ["enl", "LA_bin", "RA_bin", "LV_bin", "RV_bin"]},
}
sweep_config["parameters"] = parameters_dict
pprint.pprint(sweep_config)


def make(config=None):
    getters.deterministic(seed=config.seed)
    train_df, val_df, test_df = getters.get_data(dataset="aha", output=config.output)

    train_set, val_set, test_set = (
        getters.make_set(train_df, train=True, keys=["img", "enl"]),
        getters.make_set(val_df, keys=["img", "enl"]),
        getters.make_set(test_df, keys=["img", "enl"]),
    )

    train_loader = getters.make_loader(
        train_set, batch_size=config.batch_size, train=True
    )
    val_loader = getters.make_loader(val_set, batch_size=config.batch_size)
    test_loader = getters.make_loader(test_set, batch_size=config.batch_size)

    if config.output == "binary":
        criterion = nn.BCELoss()
        model = models.EffNet(
            model_name=config.architecture,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            pretraining=config.pretraining,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            activations=config.activations,
            BN_momentum=config.BN_momentum,
            transfer=config.transfer,
            transfer_path=config.transfer_path,
        ).to(DEVICE)
        summary(
            model,
            mode="train",
            input_size=(
                config.batch_size,
                3,
                config.image_res,
                config.image_res,
            ),
            verbose=2,
        )
    elif config.output == "ordinal":
        model = models.OrdinalEffNet(
            model_name=config.architecture,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            activations=config.activations,
            BN_momentum=config.BN_momentum,
            transfer=config.transfer,
            transfer_path=config.transfer_path,
        ).to(DEVICE)
        summary(
            model.model,
            mode="train",
            input_size=(
                config.batch_size,
                3,
                config.image_res,
                config.image_res,
            ),
            verbose=2,
        )
        criterion = losses.MultiTaskLoss(
            tasks=[
                # "enl",
                "la",
                "ra",
                "lv",
                "rv",
            ],
            loss_ft=nn.ModuleDict(
                {
                    # "enl": nn.BCELoss(),
                    "la": losses.CoGOLLoss(),
                    "ra": losses.CoGOLLoss(),
                    "lv": losses.CoGOLLoss(),
                    "rv": losses.CoGOLLoss(),
                }
            ),
            loss_weights={
                # "enl": 1,
                "la": 1,
                "ra": 1,
                "lv": 2,
                "rv": 1.5,
            },
            model=model,
        )
    elif config.output == "ord2bin":
        model = models.Ord2Bin(
            model_name=config.architecture,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            activations=config.activations,
            BN_momentum=config.BN_momentum,
            transfer=config.transfer,
            transfer_path=config.transfer_path,
        ).to(DEVICE)
        summary(
            model.model,
            mode="train",
            input_size=(
                config.batch_size,
                3,
                config.image_res,
                config.image_res,
            ),
            verbose=2,
        )
        criterion = losses.MultiTaskLoss(
            tasks=[
                # "enl",
                "la_bin",
                "ra_bin",
                "lv_bin",
                "rv_bin",
            ],
            loss_ft=nn.ModuleDict(
                {
                    # "enl": nn.BCELoss(),
                    "la_bin": nn.BCELoss(),
                    "ra_bin": nn.BCELoss(),
                    "lv_bin": nn.BCELoss(),
                    "rv_bin": nn.BCELoss(),
                }
            ),
            loss_weights={
                # "enl": 1,
                "la_bin": 1,
                "ra_bin": 1,
                "lv_bin": 2,
                "rv_bin": 1.5,
            },
            model=model,
        )
    elif config.output == "ord2binenl":
        model = models.Ord2BinEnl(
            model_name=config.architecture,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            activations=config.activations,
            BN_momentum=config.BN_momentum,
            transfer=config.transfer,
            transfer_path=config.transfer_path,
        ).to(DEVICE)
        summary(
            model.model,
            mode="train",
            input_size=(
                config.batch_size,
                3,
                config.image_res,
                config.image_res,
            ),
            verbose=2,
        )
        criterion = nn.BCELoss()

    elif config.output == "continuous":
        criterion = nn.L1Loss()
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


def get_component_preds(preds):
    y_pred_probs = {
        "la_bin": preds["la_bin"].squeeze().cpu().numpy(),
        "ra_bin": preds["ra_bin"].squeeze().cpu().numpy(),
        "lv_bin": preds["lv_bin"].squeeze().cpu().numpy(),
        "rv_bin": preds["rv_bin"].squeeze().cpu().numpy(),
    }

    y_preds = {
        "la_bin": preds["la_bin"].squeeze().cpu().numpy() > 0.5,
        "ra_bin": preds["ra_bin"].squeeze().cpu().numpy() > 0.5,
        "lv_bin": preds["lv_bin"].squeeze().cpu().numpy() > 0.5,
        "rv_bin": preds["rv_bin"].squeeze().cpu().numpy() > 0.5,
    }
    return y_pred_probs, y_preds


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
        # criterion,
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

            # Log metrics once per epoch
            wandb.log(
                {
                    "train_loss": average_train_loss,
                    # "train_enl_loss": loss["enl"],
                    "train_la_loss": loss["la_bin"],
                    "train_ra_loss": loss["ra_bin"],
                    "train_lv_loss": loss["lv_bin"],
                    "train_rv_loss": loss["rv_bin"],
                    "val_loss": val_loss["total"],
                    # "val_enl_loss": val_loss["enl"],
                    "val_la_loss": val_loss["la_bin"],
                    "val_ra_loss": val_loss["ra_bin"],
                    "val_lv_loss": val_loss["lv_bin"],
                    "val_rv_loss": val_loss["rv_bin"],
                },
                step=epoch,
            )
        else:
            # Log metrics once per epoch
            wandb.log(
                {
                    "train_loss": average_train_loss,
                    "val_loss": val_loss,
                },
                step=epoch,
            )

        # Save the model if validation loss has improved
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0  # Reset the counter
            savedir = f"{config.task}"
            getters.ensure_directory_exists(f"{config.save_dir}{savedir}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                f"{config.save_dir}{savedir}/{config.architecture}_epoch{epoch}_{config.image_res}_{config.task}.pt",
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                f"{config.save_dir}{savedir}/{config.architecture}_best_{config.image_res}_{config.task}.pt",
            )
            # wandb.save(
            #     f"/home/ddavilag/cardio/models/pe/{savedir}/{config.architecture}_epoch{epoch}_{config.image_res}.pt"
            # )
            artifact = wandb.Artifact("best_model", type="model")
            artifact.add_file(
                f"{config.save_dir}{savedir}/{config.architecture}_best_{config.image_res}_{config.task}.pt"
            )
            wandb.log_artifact(artifact)
            # wandb.save(
            #     f"/home/ddavilag/cardio/models/cardiomegaly/{savedir}/{config.architecture}_epoch{epoch}_{config.image_res}.pt"
            # )
            # wandb.save(
            #     f"/home/ddavilag/cardio/models/cardiomegaly/{savedir}/{config.architecture}_best_{config.image_res}.pt"
            # )
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
        images, enl = batch["img"], batch[config.task]
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
            _,  # enl,
            la,
            ra,
            lv,
            rv,
        ) = batch
        (
            images,
            # enl,
            la,
            ra,
            lv,
            rv,
        ) = (
            images.to(DEVICE, non_blocking=True),
            # enl.to(DEVICE, non_blocking=True),
            la.to(DEVICE, non_blocking=True),
            ra.to(DEVICE, non_blocking=True),
            lv.to(DEVICE, non_blocking=True),
            rv.to(DEVICE, non_blocking=True),
        )

        # Forward pass ➡
        outputs = model(images)
        preds = {
            # "enl": torch.sigmoid(outputs[0]),
            "la": outputs[0],
            "ra": outputs[1],
            "lv": outputs[2],
            "rv": outputs[3],
        }
        loss = criterion(
            preds,
            {
                # "enl": enl,
                "la": la,
                "ra": ra,
                "lv": lv,
                "rv": rv,
            },
        )
        # Backward pass ⬅
        optimizer.zero_grad()
        loss["total"].backward()
    elif config.output == "ord2bin":
        # print(batch)
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
        loss = criterion(
            preds,
            {
                "la_bin": la,
                "ra_bin": ra,
                "lv_bin": lv,
                "rv_bin": rv,
            },
        )
        optimizer.zero_grad()
        loss["total"].backward()

    # Step with optimizer
    optimizer.step()

    return loss


def bootstrap_results(y_test, y_preds, y_pred_probs, bootstrap_rounds=1000):
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
            if metric == "acc":
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

    return bootstrap_metrics


def evaluate(model, loader, criterion, epoch, config, test=False):
    model.eval()

    loss = 0
    if config.output == "ord2bin":
        y_preds = {"la_bin": [], "ra_bin": [], "lv_bin": [], "rv_bin": []}
        y_pred_probs = {"la_bin": [], "ra_bin": [], "lv_bin": [], "rv_bin": []}
        y_true = {"la_bin": [], "ra_bin": [], "lv_bin": [], "rv_bin": []}
    else:
        y_preds = []
        y_pred_probs = []
        y_true = []
    # la_preds = [], ra_preds = [], lv_preds = [], rv_preds = []
    # la_true = [], ra_true = [], lv_true = [], rv_true = []
    # la_pred_probs = [], ra_pred_probs = [], lv_pred_probs = [], rv_pred_probs = []
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
                images, labels = batch["img"], batch[config.task]
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
                (
                    images,
                    _,  # enl,
                    la,
                    ra,
                    lv,
                    rv,
                ) = batch
                (
                    images,
                    # enl,
                    la,
                    ra,
                    lv,
                    rv,
                ) = (
                    images.to(DEVICE, non_blocking=True),
                    # enl.to(DEVICE, non_blocking=True),
                    la.to(DEVICE, non_blocking=True),
                    ra.to(DEVICE, non_blocking=True),
                    lv.to(DEVICE, non_blocking=True),
                    rv.to(DEVICE, non_blocking=True),
                )
                outputs = model(images)
                preds = {
                    # "enl": torch.sigmoid(outputs[0]),
                    "la": outputs[0],
                    "ra": outputs[1],
                    "lv": outputs[2],
                    "rv": outputs[3],
                }
                losses = criterion(
                    preds,
                    {
                        # "enl": enl,
                        "la": la,
                        "ra": ra,
                        "lv": lv,
                        "rv": rv,
                    },
                )
                loss += losses["total"].item()
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
                losses = criterion(
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
    # if config.output == "ord2bin":
    #     for component in ["la_bin", "ra_bin", "lv_bin", "rv_bin"]:
    #         metrics = bootstrap_results(
    #             np.array(y_true[component]),
    #             np.array(y_preds[component]),
    #             np.array(y_pred_probs[component]),
    #         )
    #         for metric, (mean, lower_ci, upper_ci) in metrics.items():
    #             if test:
    #                 wandb.log(
    #                     {
    #                         f"test_{metric}_{component}": mean,
    #                         f"test_{metric}_{component}_ci_lower": lower_ci,
    #                         f"test_{metric}_{component}_ci_upper": upper_ci,
    #                     },
    #                     step=epoch,
    #                 )
    #             else:
    #                 wandb.log(
    #                     {
    #                         f"val_{metric}_{component}": mean,
    #                         f"val_{metric}_{component}_ci_lower": lower_ci,
    #                         f"val_{metric}_{component}_ci_upper": upper_ci,
    #                     },
    #                     step=epoch,
    #                 )

    # Calculate and log metrics

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

    return loss / len(loader)  # , losses


def model_pipeline(hparams=None):
    # tell wandb to get started
    # savedir = f"{'scratch' if hparams['pretrained']==False else 'pre'}_{'bin' if hparams['output']=='binary' else 'continuous'}"
    with wandb.init(
        # set the wandb project where this run will be logged
        # project="cardiomegaly sweep",
        # track hyperparameters and run metadata
        # name=f"{savedir}_{hparams['architecture']}_{hparams['image_res']}",
        # notes="Experiment 2 - Ordinal to Binary Model - 5 Outputs (LA, RA, LV, RV, ENL))",
        # magic=True,
        config=hparams,
    ) as run:
        config = run.config
        wandb.save("cardio.py")
        wandb.save("models.py")
        wandb.save("getters.py")
        wandb.save("losses.py")
        wandb.save("trainer.py")

        # make the model, data, and optimization
        (
            model,
            train_loader,
            val_loader,
            test_loader,
            criterion,
            optimizer,
        ) = make(config)
        # print(model)
        train(model, train_loader, val_loader, criterion, optimizer, config)
        checkpoint = torch.load(config.transfer_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(DEVICE)
        _, _, test_df = getters.get_data(
            dataset="aha", output=parameters_dict["output"]
        )

        # and use them to train the model
        # train(
        #     model=model,
        #     train_loader=train_loader,
        #     val_loader=val_loader,
        #     criterion=criterion,
        #     optimizer=optimizer,
        #     config=config,
        # )
        # validate model
        import trainer

        test_preds, test_imgs = trainer.evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            config=config,
            epoch=0,
            test=True,
        )
        print(test_preds, len(test_preds))

        # squeeze the predictions which is a list of numpy arrs
        # test_preds = np.squeeze(test_preds)
        # print(test_preds, test_preds.shape)
        # test_imgs = np.squeeze(test_imgs)

        test_df["preds"] = test_preds
        test_df["imgs"] = test_imgs
        test_df.to_csv(
            f"/home/ddavilag/cardio/data/csv_files/results/test_preds_{config.output}.csv"
        )
        table = wandb.Table(
            columns=[
                "Enlargement",
                "Predictions",
                "LA",
                "RA",
                "LV",
                "RV",
                "Sex",
                "Age",
            ]
        )
        [
            table.add_data(label, pred, la, ra, lv, rv, sex, age)
            for label, pred, la, ra, lv, rv, sex, age in zip(
                test_df["enl"],
                test_preds,
                test_df["LA_bin"],
                test_df["RA_bin"],
                test_df["LV_bin"],
                test_df["RV_bin"],
                test_df["Sex"],
                test_df["Age"],
            )
        ]
        wandb.log({"Test Predictions": table}, step=0)
        wandb.finish()


sweep_id = wandb.sweep(sweep_config, project="cardiomegaly sweep")
wandb.agent(sweep_id, function=model_pipeline, project="cardiomegaly sweep")
