from matplotlib import pyplot as plt
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
import random
from sklearn.preprocessing import MinMaxScaler
import models
from torchinfo import summary
import losses

WORKDIR = os.getcwd()
DATADIR = os.path.join("/home", "ddavilag", "cardio", "data")


def map2ordinal(x):
    if x == 0:
        return torch.tensor([1, 0, 0, 0], dtype=torch.float32)
    elif x == 1:
        return torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    elif x == 2:
        return torch.tensor([1, 1, 1, 0], dtype=torch.float32)
    else:
        return torch.tensor([1, 1, 1, 1], dtype=torch.float32)


def deterministic(seed=42):
    """
    Set the random seed for deterministic behavior.

    Args:
        seed (int): The seed value to set for random number generation.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def min_max_by_sex(df, col):
    """
    Normalize a column by group
    """
    scalers = {}
    df_scaled = df.copy()
    for sex in df["Sex"].unique():
        scaler = MinMaxScaler()
        train_data = df[(df["Sex"] == sex) & (df["split"] == "train")][
            col
        ].values.reshape(-1, 1)
        scaler.fit(train_data)
        scalers[sex] = scaler
    for sex in df["Sex"].unique():
        scaler = scalers[sex]
        data_to_scale = df[df["Sex"] == sex][col].values.reshape(-1, 1)
        df_scaled.loc[df["Sex"] == sex, col] = scaler.transform(data_to_scale).flatten()
    return df_scaled, scalers


def inverse_min_max_normalize_by_sex(df, col, sex_column="Sex", scalers_dict=None):
    # Copy the DataFrame to avoid modifying the original one
    df_original = df.copy()

    if scalers_dict is None:
        raise ValueError("Scalers dictionary is required for inverse transformation.")

    # Apply the inverse transformation using the stored scalers
    for sex, scaler in scalers_dict.items():
        data_to_inverse = df[df[sex_column] == sex][col].values.reshape(-1, 1)
        df_original.loc[df[sex_column] == sex, col] = scaler.inverse_transform(
            data_to_inverse
        ).flatten()

    return df_original


def get_data(dataset="aha", output="binary"):
    """
    Load data from csv files
    output: binary, ordinal, or continuous
    """
    scalers_LVIDd_cm, scalers_LA_cm = None, None
    if dataset == "edema":
        df = pd.read_csv(
            os.path.join(DATADIR, "csv_files", "DSC180B_data", "dsc180_cleaned.csv")
        )
        df = df[["path", "bnpp_value_num", "split", "edema"]]
        df["bnpp_value_num"] = np.log(df["bnpp_value_num"])
    elif dataset == "aha":
        df = pd.read_csv(
            os.path.join(DATADIR, "csv_files", "preprocessed", "AHA_data_final.csv")
        )
        display(df)

        if output == "ord2bin":
            df = df[["Path", "enl", "LA", "RA", "LV", "RV", "split"]]
            df["enl"] = np.where(df["enl"] != 0, 1, df["enl"])
            df["LA"] = np.where(df["LA"] != 0, 1, df["LA"])
            df["RA"] = np.where(df["RA"] != 0, 1, df["RA"])
            df["LV"] = np.where(df["LV"] != 0, 1, df["LV"])
            df["RV"] = np.where(df["RV"] != 0, 1, df["RV"])
        # rename LVIDd and Left Atrium
        df.rename(
            columns={"LVIDd (cm)": "LVIDd_cm", "Left Atrium (cm)": "LA_cm"},
            inplace=True,
        )
        # df = df[
        #     [
        #         "Phonetic ID",
        #         "Path",
        #         "enl",
        #         "LA",
        #         "RA",
        #         "LV",
        #         "RV",
        #         "LVIDd_cm",
        #         "LA_cm",
        #         "split",
        #         "Sex",
        #     ]
        # ]

        display(df.LVIDd_cm.describe())
        display(df.LA_cm.describe())
        df, scalers_LVIDd_cm = min_max_by_sex(df, "LVIDd_cm")
        df, scalers_LA_cm = min_max_by_sex(df, "LA_cm")
        df["enl"] = np.where(df["enl"] != 0, 1, df["enl"])
        df["LA_bin"] = np.where(df["LA"] != 0, 1, df["LA"])
        df["RA_bin"] = np.where(df["RA"] != 0, 1, df["RA"])
        df["LV_bin"] = np.where(df["LV"] != 0, 1, df["LV"])
        df["RV_bin"] = np.where(df["RV"] != 0, 1, df["RV"])
        replace_sex = {"M": 0, "F": 1, "O": 2}
        # replace sex with 0,1,2
        df["Sex"] = df["Sex"].replace(replace_sex)
        if output == "ordinal":
            df["LA"] = df["LA"].apply(map2ordinal)
            df["RA"] = df["RA"].apply(map2ordinal)
            df["LV"] = df["LV"].apply(map2ordinal)
            df["RV"] = df["RV"].apply(map2ordinal)

        # df.drop(columns=["Height (in)", "Weight (lb)"], inplace=True)
    # if output != "multi":
    # df = df.dropna()
    df = df.reset_index(drop=True)
    display(df)
    train_df = df[df["split"] == "train"]
    train_df.reset_index(drop=True, inplace=True)
    val_df = df[df["split"] == "val"]
    val_df.reset_index(drop=True, inplace=True)
    test_df = df[df["split"] == "test"]
    test_df.reset_index(drop=True, inplace=True)
    train_df.pop("split")
    val_df.pop("split")
    test_df.pop("split")
    print(train_df.shape, val_df.shape, test_df.shape)
    if output == "multi":
        return train_df, val_df, test_df, scalers_LVIDd_cm, scalers_LA_cm
    else:
        return train_df, val_df, test_df


# t, v, test, scalers_LVIDd_cm, scalers_LA_cm = get_data(dataset="aha", output="multi")


class CardioDataset(Dataset):
    def __init__(
        self, df, image_res=256, transform=None, transform_prob=0.5, ordinal=False
    ):
        self.df = df
        self.image_res = image_res
        self.transform = transform
        self.transform_prob = transform_prob
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        self.ordinal = ordinal

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = torch.load(row["Path"]).type(torch.FloatTensor)
        if self.transform and random.random() < self.transform_prob:
            img = self.transform(img)
        assert img.shape == (3, self.image_res, self.image_res)
        if self.ordinal:
            return {
                "img": img,
                "enl": torch.tensor(row["enl"], dtype=torch.float32),
                "LA": row["LA"],
                "RA": row["RA"],
                "LV": row["LV"],
                "RV": row["RV"],
                "pid": row["Phonetic ID"],
                "LA_bin": torch.tensor(row["LA_bin"], dtype=torch.float32),
                "RA_bin": torch.tensor(row["RA_bin"], dtype=torch.float32),
                "LV_bin": torch.tensor(row["LV_bin"], dtype=torch.float32),
                "RV_bin": torch.tensor(row["RV_bin"], dtype=torch.float32),
                "LVIDd_cm": torch.tensor(row["LVIDd_cm"], dtype=torch.float32),
                "LA_cm": torch.tensor(row["LA_cm"], dtype=torch.float32),
                "Sex": torch.tensor(row["Sex"], dtype=torch.int32),
                "Age": torch.tensor(row["Age"], dtype=torch.int32),
            }

        return {
            "img": img,
            "enl": torch.tensor(row["enl"], dtype=torch.float32),
            "LA_bin": torch.tensor(row["LA_bin"], dtype=torch.float32),
            "RA_bin": torch.tensor(row["RA_bin"], dtype=torch.float32),
            "LV_bin": torch.tensor(row["LV_bin"], dtype=torch.float32),
            "RV_bin": torch.tensor(row["RV_bin"], dtype=torch.float32),
            "LVIDd_cm": torch.tensor(row["LVIDd_cm"], dtype=torch.float32),
            "LA_cm": torch.tensor(row["LA_cm"], dtype=torch.float32),
            "pid": row["Phonetic ID"],
            "Sex": torch.tensor(row["Sex"], dtype=torch.int32),
            "LA": torch.tensor(row["LA"], dtype=torch.float32),
            "RA": torch.tensor(row["RA"], dtype=torch.float32),
            "LV": torch.tensor(row["LV"], dtype=torch.float32),
            "RV": torch.tensor(row["RV"], dtype=torch.float32),
            "Age": torch.tensor(row["Age"], dtype=torch.int32),
        }

    def get_output_subset(self, keys):
        def get_subset(idx):
            output_data = self.__getitem__(idx)
            return {key: output_data[key] for key in keys}

        return get_subset


class SubsetWrapper(Dataset):
    def __init__(self, dataset, subset_keys):
        """
        Args:
            dataset (TestSet): An instance of the TestSet dataset.
            subset_keys (list of str): Keys of the subset of outputs to fetch.
        """
        self.dataset = dataset
        self.subset_keys = subset_keys

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Fetch the full data for this index
        full_data = self.dataset[idx]

        # Return only the data corresponding to the subset_keys
        return {key: full_data[key] for key in self.subset_keys}


def make_set(data, keys, train=False, image_res=256, ordinal=False):
    if not train:
        full_dataset = CardioDataset(data, ordinal=ordinal)
        return SubsetWrapper(full_dataset, keys)

    train_trans = transforms.Compose(
        [
            # transforms.RandomResizedCrop(size=(image_res, image_res), scale=(0.9, 1.0)),
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.ColorJitter(brightness=0.01, contrast=0.01),
        ]
    )
    full_dataset = CardioDataset(data, transform=train_trans, ordinal=ordinal)
    return SubsetWrapper(full_dataset, keys)


# t,v,test,_,_ = get_data(dataset="aha", output="multi")
# train = make_set(t, keys=get_keys(output="multi", test=False), train=True)
# train[0]
# trainloader = make_loader(train, batch_size=16, train=True)
# batch = next(iter(trainloader))

# import viz
# viz.show_batch(
#     trainloader,
#     classes={0: "LA", 1: "RA", 2: "LV", 3: "RV"},
#     keys=get_keys(output="multi", test=True),
#     writer=None,
# )


def make_loader(dataset, batch_size, train=False):
    if train:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )


def ensure_directory_exists(path):
    """
    Ensure that a directory exists. If it doesn't, create it.

    Args:
        path (str): The path of the directory to check/create.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_keys(output, test=False):
    # sourcery skip: merge-else-if-into-elif, swap-if-else-branches
    if not test:
        if output == "binary":
            keys = ["img", "enl", "LA_bin", "RA_bin", "LV_bin", "RV_bin"]
        elif output == "ordinal":
            keys = ["img", "enl", "LA", "RA", "LV", "RV"]
        elif output == "ord2bin":
            keys = ["img", "enl", "LA_bin", "RA_bin", "LV_bin", "RV_bin"]
        elif output == "ord2binenl":
            keys = ["img", "enl", "LA", "RA", "LV", "RV"]
        elif output == "multi":
            keys = [
                "img",
                "enl",
                "LA_bin",
                "RA_bin",
                "LV_bin",
                "RV_bin",
                "LVIDd_cm",
                "LA_cm",
            ]
        else:
            raise ValueError("Invalid output type.")
    else:
        keys = [
            "img",
            "enl",
            "pid",
            "Sex",
            "LA",
            "RA",
            "LV",
            "RV",
            "LA_cm",
            "LVIDd_cm",
            "LA_bin",
            "RA_bin",
            "LV_bin",
            "RV_bin",
            "Age",
        ]
    return keys


# def get_model(config):
#     if config.output == 'backbone':
#         return models.EffNet(model_name=config.model_name, pretrained=config.pretrained, num_classes=config.num_classes)
#     ...


def get_stuff(config):
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
        )
    elif config.output == "ordinal":
        criterion = losses.MultiTaskLoss(
            tasks=["enl", "la", "ra", "lv", "rv"],
            loss_ft=nn.ModuleDict(
                {
                    "enl": nn.BCELoss(),
                    "la": losses.OrdinalRegressionLoss(),
                    "ra": losses.OrdinalRegressionLoss(),
                    "lv": losses.OrdinalRegressionLoss(),
                    "rv": losses.OrdinalRegressionLoss(),
                }
            ),
            loss_weights={"enl": 1, "la": 1, "ra": 1, "lv": 1, "rv": 1},
        )
        model = models.OrdinalEffNet(
            model_name=config.architecture,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            # pretraining=config.pretraining,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            activations=config.activations,
            BN_momentum=config.BN_momentum,
            transfer=config.transfer,
            transfer_path=config.transfer_path,
        )
    elif config.output == "multi":
        model = models.OrdinalEffNet_Multi(
            model_name=config.architecture,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            # pretraining=config.pretraining,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            activations=config.activations,
            BN_momentum=config.BN_momentum,
            transfer=config.transfer,
            transfer_path=config.transfer_path,
            tasks=config.tasks,
        )
        criterion = losses.MultiTaskLoss(
            tasks=["enl", "la_bin", "ra_bin", "lv_bin", "rv_bin", "la", "lvidd"],
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
                "enl": 1,
                "la_bin": 1,
                "ra_bin": 1,
                "lv_bin": 1,
                "rv_bin": 1,
                "la": 1,
                "lvidd": 1,
            },
        )
    else:
        raise ValueError("Invalid output type.")

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

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    return model, criterion, optimizer
