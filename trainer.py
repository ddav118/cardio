from pprint import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import GradScaler
import wandb
from torchmetrics.classification import (
    AUROC,
    AveragePrecision,
    Accuracy,
    Precision,
    Recall,
    ConfusionMatrix,
    BinaryGroupStatRates,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score
from torchmetrics.wrappers import BootStrapper
from torchvision.utils import make_grid
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            savedir = f"{'transfer' if config.transfer else 'imagenet'}_ord2bin_enl"
            getters.ensure_directory_exists(f"{config.save_dir}{savedir}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                f"{config.save_dir}{savedir}/{config.architecture}_epoch{epoch}_{config.image_res}.pt",
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                f"{config.save_dir}{savedir}/{config.architecture}_best_{config.image_res}.pt",
            )
            # wandb.save(
            #     f"/home/ddavilag/cardio/models/pe/{savedir}/{config.architecture}_epoch{epoch}_{config.image_res}.pt"
            # )
            artifact = wandb.Artifact("best_model", type="model")
            artifact.add_file(
                f"{config.save_dir}{savedir}/{config.architecture}_best_{config.image_res}.pt"
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
    model.train()
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
    y_preds = []
    y_pred_probs = []
    y_true = []

    conf_matrices = {
        # "LA_bin": ConfusionMatrix(num_classes=2, task="binary").to(DEVICE),
        # "RA_bin": ConfusionMatrix(num_classes=2, task="binary").to(DEVICE),
        # "lv_bin": ConfusionMatrix(num_classes=2, task="binary").to(DEVICE),
        # "LV_bin": ConfusionMatrix(num_classes=2, task="binary").to(DEVICE),
        "enl": ConfusionMatrix(num_classes=2, task="binary"),
    }
    binary_bootstrappers = {
        metric: BootStrapper(
            metric(
                task="binary",
            ),
            num_bootstraps=10000,
            quantile=torch.tensor([0.025, 0.975]),
        )
        for metric in [AUROC, AveragePrecision, Precision, Recall, Accuracy]
    }
    preds = []
    imgs = []
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            # if config.output == "continuous":
            #     images, enl, edema = batch
            #     images, enl = images.to(DEVICE, non_blocking=True), enl.to(
            #         DEVICE, non_blocking=True
            #     )
            #     # edema.to(DEVICE, non_blocking=True),

            #     outputs = model(images)
            #     outputs = outputs.squeeze()
            #     loss += criterion(outputs, enl).item()

            #     # aggregate outputs and labels to calculate accuracy, AUC, AUPRC
            #     predicted_probs = torch.sigmoid(outputs).cpu().numpy()
            #     y_pred_probs.extend(predicted_probs)
            #     y_preds.extend(predicted_probs > 0.5)
            #     y_true.extend(edema)

            if (config.output == "binary") | (config.output == "ord2binenl"):
                images, labels = (
                    batch["img"],
                    batch[config.task],
                    # batch["la_bin"],
                    # batch["ra_bin"],
                    # batch["lv_bin"],
                    # batch["rv_bin"],
                )
                imgs.extend(images)
                images, labels = (
                    images.to(DEVICE, non_blocking=True),
                    labels.to(DEVICE, non_blocking=True),
                    # la_bin.to(DEVICE, non_blocking=True),
                    # ra_bin.to(DEVICE, non_blocking=True),
                    # lv_bin.to(DEVICE, non_blocking=True),
                    # rv_bin.to(DEVICE, non_blocking=True),
                )

                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                outputs = outputs.squeeze()
                loss += criterion(outputs, labels).item()
                pred = torch.round(outputs)
                preds.extend(pred.cpu().numpy())
                for i, bootstrapper in enumerate(binary_bootstrappers.values()):
                    if i == 0:
                        pred = pred.cpu()
                        labels = labels.cpu()
                        bootstrapper.update(pred, labels)
                    else:
                        outputs = outputs.cpu()
                        labels = labels.cpu().to(torch.int64)
                        bootstrapper.update(outputs, labels)
                conf_matrices["enl"].update(outputs, labels)

                # y_pred_probs.extend(outputs.cpu().numpy())
                # y_preds.extend(outputs.cpu().numpy() > 0.5)
                # y_true.extend(labels.cpu().numpy())
            # elif config.output == "ordinal":
            #     (
            #         images,
            #         _,  # enl,
            #         la,
            #         ra,
            #         lv,
            #         rv,
            #     ) = batch
            #     (
            #         images,
            #         # enl,
            #         la,
            #         ra,
            #         lv,
            #         rv,
            #     ) = (
            #         images.to(DEVICE, non_blocking=True),
            #         # enl.to(DEVICE, non_blocking=True),
            #         la.to(DEVICE, non_blocking=True),
            #         ra.to(DEVICE, non_blocking=True),
            #         lv.to(DEVICE, non_blocking=True),
            #         rv.to(DEVICE, non_blocking=True),
            #     )
            #     outputs = model(images)
            #     preds = {
            #         # "enl": torch.sigmoid(outputs[0]),
            #         "la": outputs[0],
            #         "ra": outputs[1],
            #         "lv": outputs[2],
            #         "rv": outputs[3],
            #     }
            #     losses = criterion(
            #         preds,
            #         {
            #             # "enl": enl,
            #             "la": la,
            #             "ra": ra,
            #             "lv": lv,
            #             "rv": rv,
            #         },
            #     )
            #     loss += losses["total"].item()
            # elif config.output == "ord2bin":
            #     (images, enl, la, ra, lv, rv) = batch

            #     images, la, ra, lv, rv = (
            #         images.to(DEVICE, non_blocking=True),
            #         la.to(DEVICE, non_blocking=True),
            #         ra.to(DEVICE, non_blocking=True),
            #         lv.to(DEVICE, non_blocking=True),
            #         rv.to(DEVICE, non_blocking=True),
            #     )

            #     outputs = model(images)
            #     preds = {
            #         "la_bin": torch.sigmoid(outputs[0]),
            #         "ra_bin": torch.sigmoid(outputs[1]),
            #         "lv_bin": torch.sigmoid(outputs[2]),
            #         "rv_bin": torch.sigmoid(outputs[3]),
            #     }
            #     losses = criterion(
            #         preds,
            #         {
            #             "la_bin": la,
            #             "ra_bin": ra,
            #             "lv_bin": lv,
            #             "rv_bin": rv,
            #         },
            #     )
            #     y_true["la_bin"].extend(la.cpu().numpy())
            #     y_true["ra_bin"].extend(ra.cpu().numpy())
            #     y_true["lv_bin"].extend(lv.cpu().numpy())
            #     y_true["rv_bin"].extend(rv.cpu().numpy())
            #     pred_probs, preds = get_component_preds(preds)
            #     y_pred_probs["la_bin"].extend(pred_probs["la_bin"])
            #     y_pred_probs["ra_bin"].extend(pred_probs["ra_bin"])
            #     y_pred_probs["lv_bin"].extend(pred_probs["lv_bin"])
            #     y_pred_probs["rv_bin"].extend(pred_probs["rv_bin"])
            #     y_preds["la_bin"].extend(preds["la_bin"])
            #     y_preds["ra_bin"].extend(preds["ra_bin"])
            #     y_preds["lv_bin"].extend(preds["lv_bin"])
            #     y_preds["rv_bin"].extend(preds["rv_bin"])

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
    metric_dict = {
        0: "AUROC",
        1: "AUPRC",
        2: "Precision",
        3: "Recall",
        4: "Accuracy",
    }
    metric_results = {}
    for i, (name, bootstrapper) in enumerate(binary_bootstrappers.items()):
        ci = bootstrapper.compute()
        pprint(ci)
        wandb.log(
            {
                f"test_enl_{metric_dict[i]}_mean": ci["mean"].item(),
                f"test_enl_{metric_dict[i]}_std": ci["std"].item(),
                f"test_enl_{metric_dict[i]}_ci_lower": ci["quantile"][0].item(),
                f"test_enl_{metric_dict[i]}_ci_upper": ci["quantile"][1].item(),
            },
            step=epoch,
        )
        mean = ci["mean"].item()
        std = ci["std"].item()
        q2_25 = ci["quantile"][0].item()
        q97_75 = ci["quantile"][1].item()
        metric_results[f"{metric_dict[i]}"] = [mean, std, q2_25, q97_75]
    print(metric_results)

    conf_matrix = {key: conf_matrices[key].compute().tolist() for key in conf_matrices}
    pprint(conf_matrix)
    confusion_matrix = conf_matrix["enl"]
    confusion_matrix = [item for sublist in confusion_matrix for item in sublist]
    table = wandb.Table(
        data=[confusion_matrix],
        columns=[
            "True negatives",
            "False positives",
            "False negatives",
            "True positives",
        ],
    )
    metric_df = pd.DataFrame.from_dict(metric_results, orient="index")
    metric_df.columns = ["mean", "std", "q2_5", "q97_5"]
    metric_df = metric_df.T

    metrics = wandb.Table(
        dataframe=metric_df,
        columns=[
            "AUROC",
            "AUPRC",
            "Precision",
            "Recall",
            "Accuracy",
        ],
    )
    wandb.log({"Test Metrics 95% CI": metrics}, step=epoch)

    # Log the table
    wandb.log({"Confusion Matrix - Test": table}, step=epoch)

    # Calculate and log metrics

    # metrics = bootstrap_results(
    #     np.array(y_true), np.array(y_preds), np.array(y_pred_probs)
    # )

    # # Log metrics with confidence intervals
    # for metric, (mean, lower_ci, upper_ci) in metrics.items():
    #     if test:
    #         wandb.log(
    #             {
    #                 f"test_{metric}": mean,
    #                 f"test_{metric}_ci_lower": lower_ci,
    #                 f"test_{metric}_ci_upper": upper_ci,
    #             },
    #             #step=epoch,
    #         )
    #     else:
    #         wandb.log(
    #             {
    #                 f"val_{metric}": mean,
    #                 f"val_{metric}_ci_lower": lower_ci,
    #                 f"val_{metric}_ci_upper": upper_ci,
    #             },
    #             #step=epoch,
    #         )

    return preds, imgs  # , losses
