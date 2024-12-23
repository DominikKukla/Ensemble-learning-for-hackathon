from pathlib import Path
from typing import Dict

import kaggle  # type: ignore
import torch
import torch.nn as nn
from torchvision import models  # type: ignore


class Models:
    def __init__(self):
        kaggle.api.authenticate()
        root_folder = Path(__file__).parent.parent.parent
        self.download_folder = root_folder / "data" / "models"
        self.size_tuple = (224, 224)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_weights: Dict[str, float] = dict()

    def download_from_kaggle(self):
        self.download_folder.mkdir(exist_ok=True)
        kaggle.api.dataset_download_files(  # type: ignore
            "dzmitrypihulski/models-for-hackathon",
            path=self.download_folder,
            unzip=True,
            quiet=False,
        )

    def create_new_efficientnet_b0(self, num_classes: int = 8):
        model = models.efficientnet_b0(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model.to(self.device)

    def create_new_efficientnet_b1(self, num_classes: int = 8):
        model = models.efficientnet_b1(weights="IMAGENET1K_V2")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model.to(self.device)

    def create_new_mobilenet_v3_large(self, num_classes: int = 8):
        model = models.mobilenet_v3_large(weights="IMAGENET1K_V2")
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model.to(self.device)

    def create_new_shufflenet_v2_x2_0(self, num_classes: int = 8):
        model = models.shufflenet_v2_x2_0(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(self.device)

    def prepare_finetuned_efficientnet_b0(self, model_name: str, num_classes: int = 8):
        model = models.efficientnet_b0(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

        checkpoint = torch.load(  # type: ignore
            str(self.download_folder) + f"/{model_name}_best_f1.pth"
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()
        self.model_weights[model_name] = checkpoint["best_val_f1"]
        return model.to(self.device)

    def prepare_finetuned_efficientnet_b1(self, model_name: str, num_classes: int = 8):
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
    ):
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
    ):
        model = models.shufflenet_v2_x2_0(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        checkpoint = torch.load(  # type: ignore
            str(self.download_folder) + f"/{model_name}_best_f1.pth"
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()
        self.model_weights[model_name] = checkpoint["best_val_f1"]

        return model.to(self.device)
