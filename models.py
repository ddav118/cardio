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

WORKDIR = os.getcwd()
DATADIR = os.path.join("/home", "ddavilag", "cardio", "data")


def set_bn_momentum(model, momentum=0.99):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = momentum


def replace_activations(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.LeakyReLU())
        elif isinstance(child, nn.Module):
            replace_activations(child)


class EffNet(nn.Module):
    def __init__(
        self,
        model_name: str = "efficientnet_b2",
        num_classes: int = 1,
        pretrained: bool = True,
        pretraining: bool = True,
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.2,
        activations: str = "SiLU",
        BN_momentum: bool = True,
        transfer: bool = True,
        transfer_path=None,
    ):
        """
        Initializes the baseline model. ```config.output = 'baseline'```

        Args:
            model_name: The name of the model architecture to use. Defaults to "efficientnet_b2".
            num_classes: The number of output classes. Defaults to 1.
            pretrained: Whether to use pre-trained weights. Defaults to True.
            pretraining: Whether to enable pre-training. Defaults to True.
            drop_rate: The dropout rate. Defaults to 0.2.
            drop_path_rate: The drop path rate. Defaults to 0.2.
            activations: The activation function to use. Defaults to "SiLU".
            BN_momentum: Whether to use batch normalization momentum. Defaults to True.
            transfer: Whether to enable transfer learning. Defaults to True.
            transfer_path: The path to the pre-trained model checkpoint. Defaults to None.

        Returns:
            None
        """
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name,
            num_classes=1,
            pretrained=pretrained,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            exportable=True,
            in_chans=3,
        )
        replace_activations(self.model)
        if BN_momentum:
            set_bn_momentum(self.model)
        if transfer:
            checkpoint = torch.load(transfer_path)
            state_dict = checkpoint["model_state_dict"]
            correct_state_dict = {
                k.replace("model.", ""): v for k, v in state_dict.items()
            }
            self.model.load_state_dict(correct_state_dict)
            print("loaded transfer model")
            self.model.reset_classifier(num_classes=num_classes, global_pool="avg")
        if not pretraining:
            for param in self.model.parameters():
                param.requires_grad = False
        # check if the model is an efficientnet or resnet
        if "efficientnet" in model_name:
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        else:
            for param in self.model.get_classifier().parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


class OrdinalEffNet(EffNet):
    def __init__(
        self,
        model_name,
        num_classes,
        pretrained=True,
        drop_rate=0.2,
        drop_path_rate=0.2,
        activations="SiLU",
        BN_momentum=True,
        transfer=True,
        transfer_path=None,
    ):
        # Initialize EffNet with the provided arguments
        super().__init__(
            model_name=model_name,
            num_classes=1,
            pretrained=pretrained,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            activations=activations,
            BN_momentum=BN_momentum,
            transfer=transfer,
            transfer_path=transfer_path,
        )
        # Assuming 'num_features' is the number of features from the model's penultimate layer
        num_features = 1408
        print(num_features)

        # Copying all layers except the classifier
        self.features = nn.Sequential(*list(self.model.children())[:-1])

        # reinitialize the classifier
        self.model.reset_classifier(num_classes=1, global_pool="avg")
        # Adding ordinal classifiers for different parts
        self.model.LA_ord = nn.Linear(num_features, num_classes)
        self.model.RA_ord = nn.Linear(num_features, num_classes)
        self.model.LV_ord = nn.Linear(num_features, num_classes)
        self.model.RV_ord = nn.Linear(num_features, num_classes)

        for param in self.model.parameters():
            param.requires_grad = False

        # Make the last convolutional layer trainable
        for layer in reversed(self.features):
            if isinstance(layer, nn.Conv2d):
                for param in layer.parameters():
                    param.requires_grad = True
                break

        for param in self.model.classifier.parameters():
            param.requires_grad = True
        for param in self.model.LA_ord.parameters():
            param.requires_grad = True
        for param in self.model.RA_ord.parameters():
            param.requires_grad = True
        for param in self.model.LV_ord.parameters():
            param.requires_grad = True
        for param in self.model.RV_ord.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        enl = self.model.classifier(x)
        la = self.model.LA_ord(x)
        ra = self.model.RA_ord(x)
        lv = self.model.LV_ord(x)
        rv = self.model.RV_ord(x)
        return enl, la, ra, lv, rv


class OrdinalEffNet_Multi(EffNet):
    def __init__(
        self,
        model_name,
        num_classes,
        pretrained=True,
        drop_rate=0.2,
        drop_path_rate=0.2,
        activations="SiLU",
        BN_momentum=True,
        transfer=True,
        transfer_path=None,
        tasks = 'lvidd'# 'lad' or 'combo
    ):
        # Initialize EffNet with the provided arguments
        super().__init__(
            model_name=model_name,
            num_classes=1,
            pretrained=pretrained,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            activations=activations,
            BN_momentum=BN_momentum,
            transfer=transfer,
            transfer_path=transfer_path,
        )
        # Assuming 'num_features' is the number of features from the model's penultimate layer
        num_features = 1408
        print(num_features)

        # Copying all layers except the classifier
        self.features = nn.Sequential(*list(self.model.children())[:-1])

        # reinitialize the classifier
        self.model.reset_classifier(num_classes=1, global_pool="avg")
        # Adding ordinal classifiers for different parts
        self.model.LA_ord = nn.Linear(num_features, num_classes)
        self.model.RA_ord = nn.Linear(num_features, num_classes)
        self.model.LV_ord = nn.Linear(num_features, num_classes)
        self.model.RV_ord = nn.Linear(num_features, num_classes)
        self.tasks = tasks
        self.model.LAD = nn.Linear(num_features, 1)
        self.model.LVIDD = nn.Linear(num_features, 1)

        for param in self.model.parameters():
            param.requires_grad = False

        # Make the last convolutional layer trainable
        for layer in reversed(self.features):
            if isinstance(layer, nn.Conv2d):
                for param in layer.parameters():
                    param.requires_grad = True
                break
        if self.tasks == 'lvidd':
            for param in self.model.LVIDD.parameters():
                param.requires_grad = True
        elif self.tasks == 'lad':
            for param in self.model.LAD.parameters():
                param.requires_grad = True
        else:
            for param in self.model.LVIDD.parameters():
                param.requires_grad = True
            for param in self.model.LAD.parameters():
                param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        for param in self.model.LA_ord.parameters():
            param.requires_grad = True
        for param in self.model.RA_ord.parameters():
            param.requires_grad = True
        for param in self.model.LV_ord.parameters():
            param.requires_grad = True
        for param in self.model.RV_ord.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        if self.tasks == 'lvidd':
            lvidd = self.model.LVIDD(x)
        if self.tasks == 'lad':
            lad = self.model.LAD(x)
        if self.tasks == 'combo':
            lvidd = self.model.LVIDD(x)
            lad = self.model.LAD(x)
        enl = self.model.classifier(x)
        la = self.model.LA_ord(x)
        ra = self.model.RA_ord(x)
        lv = self.model.LV_ord(x)
        rv = self.model.RV_ord(x)
        if self.tasks == 'lvidd':
            return enl, la, ra, lv, rv, lvidd
        if self.tasks == 'lad':
            return enl, la, ra, lv, rv, lad
        if self.tasks == 'combo':
            return enl, la, ra, lv, rv, lad, lvidd
        