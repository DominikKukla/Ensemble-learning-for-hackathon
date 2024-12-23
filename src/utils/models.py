from pathlib import Path
from typing import Dict

import kaggle  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import models  # type: ignore


class Models:
    def __init__(self) -> None:
        kaggle.api.authenticate()
        root_folder = Path(__file__).parent.parent.parent
        self.download_folder = root_folder / "data" / "models"
        self.size_tuple = (224, 224)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_weights: Dict[str, float] = dict()

    def download_from_kaggle(self) -> None:
        self.download_folder.mkdir(exist_ok=True)
        kaggle.api.dataset_download_files(  # type: ignore
            "dzmitrypihulski/models-for-hackathon",
            path=self.download_folder,
            unzip=True,
            quiet=False,
        )

    def create_new_efficientnet_b0(self, num_classes: int = 8) -> nn.Module:
        model = models.efficientnet_b0(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model.to(self.device)

    def create_new_efficientnet_b1(self, num_classes: int = 8) -> nn.Module:
        model = models.efficientnet_b1(weights="IMAGENET1K_V2")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model.to(self.device)

    def create_new_mobilenet_v3_large(self, num_classes: int = 8) -> nn.Module:
        model = models.mobilenet_v3_large(weights="IMAGENET1K_V2")
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model.to(self.device)

    def create_new_shufflenet_v2_x2_0(self, num_classes: int = 8) -> nn.Module:
        model = models.shufflenet_v2_x2_0(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(self.device)

    def prepare_finetuned_efficientnet_b0(
        self, model_name: str, num_classes: int = 8
    ) -> nn.Module:
        model = models.efficientnet_b0(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

        checkpoint = torch.load(  # type: ignore
            str(self.download_folder) + f"/{model_name}_best_f1.pth"
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()
        self.model_weights[model_name] = checkpoint["best_val_f1"]
        return model.to(self.device)

    def prepare_finetuned_efficientnet_b1(
        self, model_name: str, num_classes: int = 8
    ) -> nn.Module:
        model = models.efficientnet_b1(weights="IMAGENET1K_V2")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

        checkpoint = torch.load(  # type: ignore
            str(self.download_folder) + f"/{model_name}_best_f1.pth"
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()
        self.model_weights[model_name] = checkpoint["best_val_f1"]
        return model.to(self.device)

    def prepare_finetuned_mobilenet_v3_large(
        self, model_name: str, num_classes: int = 8
    ) -> nn.Module:
        model = models.mobilenet_v3_large(weights="IMAGENET1K_V2")
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

        checkpoint = torch.load(  # type: ignore
            str(self.download_folder) + f"/{model_name}_best_f1.pth"
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()
        self.model_weights[model_name] = checkpoint["best_val_f1"]

        return model.to(self.device)

    def prepare_finetuned_shufflenet_v2_x2_0(
        self, model_name: str, num_classes: int = 8
    ) -> nn.Module:
        model = models.shufflenet_v2_x2_0(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        checkpoint = torch.load(  # type: ignore
            str(self.download_folder) + f"/{model_name}_best_f1.pth"
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()
        self.model_weights[model_name] = checkpoint["best_val_f1"]

        return model.to(self.device)

    def plot_training_metrics(self, checkpoint_path: str, name_of_the_net: str) -> None:
        checkpoint = torch.load(checkpoint_path)  # type: ignore
        losses_val = checkpoint["val_loss"]
        losses_train = checkpoint["train_loss"]
        f1_metric_val = checkpoint["f1_metric_val"]

        plt.figure(figsize=(10, 7))  # type: ignore
        marker_on = [losses_val.index(min(losses_val))]

        # Create the first plot with the left y-axis
        fig, ax1 = plt.subplots(figsize=(10, 7))  # type: ignore

        # Plot train and validation loss on the left y-axis
        ax1.plot(  # type: ignore
            np.arange(1, len(losses_train) + 1),  # type: ignore
            losses_train,
            color="r",
            label="Train loss",
        )
        ax1.plot(  # type: ignore
            np.arange(1, len(losses_val) + 1),  # type: ignore
            losses_val,
            "-gD",
            label="Test loss",
            markevery=marker_on,
        )

        # Annotate the minimum loss point
        bbox = dict(boxstyle="round", fc="0.8")
        ax1.annotate(  # type: ignore
            text=f"Max F1 score after {f1_metric_val.index(max(f1_metric_val))+1} epochs, equals: {max(f1_metric_val)}",
            xy=(losses_val.index(min(losses_val)) + 1, min(losses_val)),
            xytext=(losses_val.index(min(losses_val)) + 1, min(losses_val) + 0.05),
            arrowprops=dict(facecolor="green", shrink=0.2),
            bbox=bbox,
        )

        ax1.set_xlabel("Epoch")  # type: ignore
        ax1.set_ylabel("CrossEntropyLoss")  # type: ignore
        ax1.grid()  # type: ignore

        # Create the second y-axis for F1 metric
        ax2 = ax1.twinx()
        ax2.plot(  # type: ignore
            np.arange(1, len(f1_metric_val) + 1),  # type: ignore
            f1_metric_val,
            color="blue",
            label="F1 Metric (Val)",
        )
        ax2.set_ylabel("f1_metric_val")  # type: ignore

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")  # type: ignore

        plt.title(f"Train and Val loss, F1 Metric after {len(losses_train)} epochs")  # type: ignore
        plt.savefig(f"data/plots/{name_of_the_net}.png")  # type: ignore
