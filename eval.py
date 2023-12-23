import pandas as pd
import numpy as np
import torch
import getters
import models
import losses
import pprint
from ignite.metrics import Accuracy, Precision, Recall, ConfusionMatrix
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
)
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
commonpath = "/home/ddavilag/cardio/"
experiments = {
    "expt1": {
        "efficientnet_b2_pre": f"{commonpath}models/pe/pre/efficientnet_b2_best_256.pt",
        "densenet121_pre": f"{commonpath}models/pe/pre/densenet121_best_256.pt",
        "resnet50_pre": f"{commonpath}models/pe/pre/resnet50_best_256.pt",
    },  # imagenet weights + pretraining on pulmonary edema dataset
    "expt2": {
        "efficientnet_b2_scratch": f"{commonpath}models/pe/scratch/efficientnet_b2_best_256.pt",
        "densenet121_scratch": f"{commonpath}models/pe/scratch/densenet121_best_256.pt",
        "resnet50_scratch": f"{commonpath}models/pe/scratch/resnet50_best_256.pt",
    },  # random weights + pretraining on pulmonary edema dataset
    "expt3": {
        "transfer": f"{commonpath}models/pe/pre/efficientnet_b2_best_256.pt",
        "efficientnet_b2_binary": f"{commonpath}models/cardiomegaly/transfer_bin/efficientnet_b2_best_256.pt",
    },  # BASELINE BINARY - transfer learning weights from pulmonary edema for cardiomegaly dataset - allowed all layers to be trained (no freezing)
    "expt4": {
        "transfer": f"{commonpath}models/pe/pre/efficientnet_b2_best_256.pt",
        "efficientnet_b2_ordinal": f"{commonpath}models/cardiomegaly/multi/ordinal/efficientnet_b2_best_256.pt",
    },  # ORDINAL - transfer learning weights from BASELINE BINARY - allowed final conv. layer onwards to be trained (no freezing)
    "expt5": {},  # ORDINAL w/ LVIDd - transfer learning weights from pulmonary edema for cardiomegaly dataset - allowed all layers to be trained (no freezing)
}

config = {
    "pretrained": True,
    "pretraining": True,
    "output": "ordinal",
    "epochs": 350,
    "batch_size": 128,
    "num_classes": 4,
    "original_image_res": 256,
    "image_res": 256,
    "architecture": "efficientnet_b2",
    "drop_rate": 0.3,
    "drop_path_rate": 0.2,
    "activations": "LeakyReLU",
    "BN_momentum": True,
    "optimizer": "adam",
    "weight_decay": 1e-5,
    "learning_rate": 1e-5,
    "loss": "BCELoss",
    "seed": 42,
    "patience": 35,
    "transfer": True,
    "transfer_path": "/home/ddavilag/cardio/models/cardiomegaly/transfer_bin/efficientnet_b2_best_256_true.pt",
    "save_dir": "/home/ddavilag/cardio/models/cardiomegaly/multi/",
}
pprint.pprint(config)
getters.deterministic(seed=config["seed"])
_, _, test_df = getters.get_data(output=config["output"])
test_set = getters.make_set(test_df, keys=getters.get_keys(config["output"], test=True))
test_loader = getters.make_loader(test_set, 1, train=False)
model = models.OrdinalEffNet(
    model_name=config["architecture"],
    num_classes=config["num_classes"],
    pretrained=config["pretrained"],
    drop_rate=config["drop_rate"],
    drop_path_rate=config["drop_path_rate"],
    activations=config["activations"],
    BN_momentum=config["BN_momentum"],
    transfer=config["transfer"],
    transfer_path=config["transfer_path"],
).to(DEVICE)
checkpoint = torch.load(
    experiments["expt4"]["efficientnet_b2_ordinal"], map_location=DEVICE
)
state_dict = checkpoint["model_state_dict"]
correct_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
print("loaded transfer model")


def create_roc_curve(y_test, y_pred_probs, target, plot=False):
    """
    Create a ROC curve for a given target.

    Args:
        y_test: Array-like, true labels.
        y_pred_probs: Array-like, predicted probabilities.
        target: str, target to create ROC curve for.
        plot: bool, optional. Whether to plot the ROC curve (default: False).

    Returns:
        fpr: Array-like, false positive rates.
        tpr: Array-like, true positive rates.
        thresholds: Array-like, thresholds.
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {target}")
        # create legend with area under curve
        legend = plt.legend(
            [f"AUC: {roc_auc_score(y_test, y_pred_probs):.3f}"],
            loc="lower right",
        )
        plt.show()

    return fpr, tpr, thresholds


def create_precision_recall_curve(y_test, y_pred_probs, target, plot=False):
    """
    Create a precision-recall curve for a given target.

    Args:
        y_test: Array-like, true labels.
        y_pred_probs: Array-like, predicted probabilities.
        target: str, target to create precision-recall curve for.
        plot: bool, optional. Whether to plot the precision-recall curve (default: False).

    Returns:
        precision: Array-like, precision values.
        recall: Array-like, recall values.
        thresholds: Array-like, thresholds.
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs)
    if plot:
        fig, ax = plt.subplots()
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve for {target}")
        # create legend with area under curve
        legend = plt.legend(
            [f"AUPRC: {average_precision_score(y_test, y_pred_probs):.3f}"],
            loc="lower left",
        )
        plt.show()
    return precision, recall, thresholds


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
    metrics = ["acc", "auroc", "auprc", "precision", "recall"]
    metric_methods = {
        "acc": accuracy_score,
        "auroc": roc_auc_score,
        "auprc": average_precision_score,
        "precision": precision_score,
        "recall": recall_score,
    }

    bootstrap_metrics = {metric: [] for metric in metrics}
    y_test = np.array(y_test)
    y_preds = np.array(y_preds)
    y_pred_probs = np.array(y_pred_probs)
    fpr, tpr, thresholds = create_roc_curve(y_test, y_pred_probs, predict, plot=True)
    precision, recall, thresholds = create_precision_recall_curve(
        y_test, y_pred_probs, predict, plot=True
    )
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
    # print(f"{predict}", bootstrap_metrics)
    # if test:
    #     metric_table = wandb.Table(columns=["Metric", "Mean", "Lower CI", "Upper CI"])
    #     for metric, (mean, lower_ci, upper_ci) in bootstrap_metrics.items():
    #         metric_table.add_data(metric, mean, lower_ci, upper_ci)
    #     wandb.log({"Bootstrap Metrics": metric_table}, step=0)
    fig_dict = {
        "roc": [fpr, tpr, thresholds],
        "auprc": [precision, recall, thresholds],
    }

    return bootstrap_metrics, fig_dict


def ordinal2binary(pred: np.ndarray):
    """Convert ordinal predictions to binary, e.g.

    x < 0.5 -> 0
    x >= 0.5 -> 1
    etc.
    """

    return (pred > 0.5).astype(int)


def test(loader, model, config, target_keys):
    model.eval()
    with torch.no_grad():
        preds = {key: [] for key in target_keys}  # use this to store predictions
        pred_probs = {
            key: [] for key in target_keys
        }  # use this to store prediction probabilities
        targets = {key: [] for key in target_keys}
        gender = []
        for batch in loader:
            image = batch["img"].to(DEVICE)
            output = model(image)
            gender.append(batch["Sex"].cpu().numpy()[0])
            enl = torch.sigmoid(output[0]).cpu().numpy()[0][0]
            enl_bin = np.round(enl)
            preds["enl"].append(enl_bin)
            pred_probs["enl"].append(enl)

            for target in target_keys:
                targets[target].append(batch[target].cpu().numpy()[0])

            if config["output"] == "ordinal":
                la = torch.sigmoid(output[1]).cpu().numpy()[0][1]
                ra = torch.sigmoid(output[2]).cpu().numpy()[0][1]
                lv = torch.sigmoid(output[3]).cpu().numpy()[0][1]
                rv = torch.sigmoid(output[4]).cpu().numpy()[0][1]

                la_bin = ordinal2binary(la)
                ra_bin = ordinal2binary(ra)
                lv_bin = ordinal2binary(lv)
                rv_bin = ordinal2binary(rv)

                preds["LA_bin"].append(la_bin)
                preds["RA_bin"].append(ra_bin)
                preds["LV_bin"].append(lv_bin)
                preds["RV_bin"].append(rv_bin)

                pred_probs["LA_bin"].append(la)
                pred_probs["RA_bin"].append(ra)
                pred_probs["LV_bin"].append(lv)
                pred_probs["RV_bin"].append(rv)

    return preds, pred_probs, targets, gender


preds, prob_preds, targets, gender = test(
    test_loader,
    model,
    config,
    target_keys=["LA_bin", "RA_bin", "LV_bin", "RV_bin", "enl"],
)
# make dataframe with prob_preds, targets and gender
prob_preds = pd.DataFrame(prob_preds)
prob_preds.columns = ["Prob LA", "Prob RA", "Prob LV", "Prob RV", "Prob ENL"]
targets = pd.DataFrame(targets)
targets.columns = ["True LA", "True RA", "True LV", "True RV", "True ENL"]
gender = pd.DataFrame(gender)
merged = pd.concat([prob_preds, targets, gender], axis=1)
merged.rename(columns={0: "Gender"}, inplace=True)
merged.to_csv('merged.csv')

# find the accuracy, auroc, auprc, precision, and recall
from collections import defaultdict

boot_dict = defaultdict(dict)
fig_dict = defaultdict(dict)

for key in preds.keys():
    boot_dict[key], fig_dict[key] = bootstrap_results(
        targets[key],
        preds[key],
        prob_preds[key],
        predict=key,
        bootstrap_rounds=1000,
        test=True,
    )

boot_dict


preds = pd.DataFrame(preds)
preds.columns = ["Pred LA", "Pred RA", "Pred LV", "Pred RV", "Pred ENL"]
prob_preds = pd.DataFrame(prob_preds)
prob_preds.columns = ["Prob LA", "Prob RA", "Prob LV", "Prob RV", "Prob ENL"]
targets = pd.DataFrame(targets)
targets.columns = ["True LA", "True RA", "True LV", "True RV", "True ENL"]
gender = pd.DataFrame(gender)
gender.columns = ["Gender"]

merged = pd.concat([preds, prob_preds, targets, gender], axis=1)
gender_names = {
    0: "Male",
    1: "Female",
}
boot_gender = defaultdict(dict)
fig_gender = defaultdict(dict)
from IPython.display import display

for gender in [0, 1]:
    gender_df = merged[merged["Gender"] == gender]
    keys = ["LA", "RA", "LV", "RV", "ENL"]
    # get the columns containing the key
    for key in keys:
        cols = [col for col in gender_df.columns if key in col]
        df = gender_df[cols]
        (
            boot_gender[f"{gender_names[gender]}_" + key],
            fig_gender[f"{gender_names[gender]}_" + key],
        ) = bootstrap_results(
            df.iloc[:, 2].values,
            df.iloc[:, 0].values,
            df.iloc[:, 1].values,
            bootstrap_rounds=1000,
            test=True,
            predict=f"{gender_names[gender]} " + key,
        )
pprint.pprint(boot_gender)


# turn defaultdict into dataframe
def boot2df(boot_dict):
    res_df = pd.DataFrame()
    for key in boot_dict.keys():
        df = pd.DataFrame(boot_dict[key])
        df = df.transpose()
        df.columns = ["Mean", "Lower CI", "Upper CI"]
        df["Metric"] = key
        res_df = pd.concat([res_df, df])
    return res_df


df = boot2df(boot_gender)
# rename Metric to Classification
df = df.rename(columns={"Metric": "Classification"})
df["Metric"] = df.index
df = df.reset_index(drop=True)
# split Classification by _ into 2 columns: Gender and Classification

df["Gender"] = df.Classification.str.split("_").str[0]
df.Classification = df.Classification.str.split("_").str[1]

boot2df(boot_dict)
