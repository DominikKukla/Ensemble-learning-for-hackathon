from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data.dataloader
from sklearn.metrics import classification_report, f1_score  # type: ignore
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms  # type: ignore

from utils.utils import set_seed

device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore
print(device)


def load_test_data(root_dir: str = "test_data"):  # type: ignore
    size_tuple = (224, 224)
    # Define a mapping from folder names to class numbers
    class_mapping = {
        "barszcz": 1,
        "bigos": 2,
        "Kutia": 3,
        "makowiec": 4,
        "piernik": 5,
        "pierogi": 6,
        "sernik": 7,
        "grzybowa": 8,
    }

    batch_size = 32

    # Load the original dataset
    test_dataset = datasets.ImageFolder(
        root=root_dir,
        transform=transforms.Compose(
            [
                transforms.Resize(size_tuple),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalize
            ]
        ),
    )

    test_dataset.class_to_idx = {
        k: class_mapping[k] for k in test_dataset.class_to_idx.keys()
    }

    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)  # type: ignore
    return test_loader  # type: ignore


class EnsembleTrainer:
    def __init__(self, num_classes: int, device: torch.device):
        self.device = device
        self.model_weights: Dict[Any, Any] = dict()

        self.models: Dict[str, nn.Module] = {
            "efficientnet_b0": self._prepare_efficientnet_b0(
                num_classes, "efficientnet_b0"
            ),
            "efficientnet_b1": self._prepare_efficientnet_b1(
                num_classes, "efficientnet_b1"
            ),
            "mobilenet_v3_large": self._prepare_mobilenet_v3_large(
                num_classes, "mobilenet_v3_large"
            ),
            "shufflenet_v2_x2_0": self._prepare_shufflenet_v2_x2_0(
                num_classes, "shufflenet_v2_x2_0"
            ),
        }

    def _prepare_efficientnet_b0(self, num_classes: int, model_name: str):
        model = models.efficientnet_b0(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

        # Load the saved checkpoint
        checkpoint = torch.load(f"data/models/{model_name}_best_f1.pth")  # type: ignore

        # Load the model weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Optional: Set the model to evaluation mode
        model.eval()
        self.model_weights[model_name] = checkpoint["best_val_f1"]
        return model.to(self.device)

    def _prepare_efficientnet_b1(self, num_classes: int, model_name: str):
        model = models.efficientnet_b1(weights="IMAGENET1K_V2")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

        # Load the saved checkpoint
        checkpoint = torch.load(f"data/models/{model_name}_best_f1.pth")  # type: ignore

        # Load the model weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Optional: Set the model to evaluation mode
        model.eval()
        self.model_weights[model_name] = checkpoint["best_val_f1"]
        return model.to(self.device)

    def _prepare_mobilenet_v3_large(self, num_classes: int, model_name: str):
        model = models.mobilenet_v3_large(weights="IMAGENET1K_V2")
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

        # Load the saved checkpoint
        checkpoint = torch.load(f"data/models/{model_name}_best_f1.pth")  # type: ignore

        # Load the model weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Optional: Set the model to evaluation mode
        model.eval()
        self.model_weights[model_name] = checkpoint["best_val_f1"]

        return model.to(self.device)

    def _prepare_shufflenet_v2_x2_0(self, num_classes: int, model_name: str):
        model = models.shufflenet_v2_x2_0(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # Load the saved checkpoint
        checkpoint = torch.load(f"data/models/{model_name}_best_f1.pth")  # type: ignore

        # Load the model weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Optional: Set the model to evaluation mode
        model.eval()
        self.model_weights[model_name] = checkpoint["best_val_f1"]

        return model.to(self.device)

    def sequential_batch_evaluation(self, dataloader, batch_size: int = 20):  # type: ignore
        # Total number of classes
        num_classes = len(dataloader.dataset.classes)  # type: ignore

        # Create a list to track F1 scores for each sequential batch
        sequential_f1_scores = []

        # Get the total number of images
        total_images = len(dataloader.dataset)  # type: ignore

        # Calculate how many full batches of 10 images per class we can process
        images_per_class = total_images // num_classes
        batches_per_class = images_per_class // batch_size

        # Iterate through sequential batches
        for batch_group in range(batches_per_class):
            # Reset ensemble probabilities and labels for this batch group
            all_ensemble_probs: dict[Any, Any] = defaultdict(list)
            all_labels = []

            # Process 10 images from each class
            for class_idx in range(num_classes):
                start_idx = (batch_group * batch_size) + (class_idx * images_per_class)
                end_idx = start_idx + batch_size

                # Create a subset for this specific slice of images
                subset_indices = list(range(start_idx, end_idx))
                subset = Subset(dataloader.dataset, subset_indices)  # type: ignore

                # Create a new dataloader for this subset
                subset_loader = torch.utils.data.DataLoader(  # type: ignore
                    subset,  # type: ignore
                    batch_size=batch_size,
                    shuffle=False,
                )

                # Process this subset
                for images, labels in subset_loader:
                    images = images.to(self.device)

                    # Collect probabilities from each model
                    for model_name, model in self.models.items():
                        probs = torch.softmax(model(images), dim=1)
                        all_ensemble_probs[model_name].append(probs.detach().cpu())

                    all_labels.extend(labels.cpu().numpy())  # type: ignore

            # Normalize model weights
            n = 5
            self.model_weights_now = {k: v**n for k, v in self.model_weights.items()}
            total_weight = sum(self.model_weights_now.values())
            model_weights_normalized = {
                model: weight / total_weight
                for model, weight in self.model_weights_now.items()
            }

            # Calculate ensemble predictions
            all_ensemble_preds: List[Any] = []
            for i in range(len(list(all_ensemble_probs.values())[0])):
                weighted_preds: List[Any] = []
                for model_name in list(self.models.keys()):
                    weight = model_weights_normalized.get(model_name, 0.0)
                    if weight == 0.0:
                        print(f"Warning: No weight for model {model_name}")
                        break
                    weighted_preds.append(all_ensemble_probs[model_name][i] * weight)

                ensemble_probs = torch.sum(torch.stack(weighted_preds), dim=0)
                _, ensemble_pred = torch.max(ensemble_probs, 1)
                all_ensemble_preds.extend(ensemble_pred.cpu().numpy())  # type: ignore

            # Calculate F1 score for this batch group
            batch_f1 = f1_score(all_labels, all_ensemble_preds, average="weighted")  # type: ignore
            sequential_f1_scores.append(batch_f1)  # type: ignore

            # Print detailed report for this batch group
            print(f"\nBatch Group {batch_group + 1} Results:")
            print(f"Weighted Ensemble F1 Score: {batch_f1}")
            print("\nDetailed Classification Report:")
            print(
                classification_report(  # type: ignore
                    all_labels, all_ensemble_preds, digits=4, zero_division=0
                )
            )

        # Print overall summary of F1 scores
        print("\nSequential Batch F1 Scores:")
        print(sequential_f1_scores)  # type: ignore
        print(f"Average F1 Score: {np.mean(sequential_f1_scores)}")  # type: ignore
        print(f"F1 Score Standard Deviation: {np.std(sequential_f1_scores)}")  # type: ignore

        return sequential_f1_scores  # type: ignore


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(23)

    test_loader: DataLoader = load_test_data("data/test_data")  # type: ignore

    ensembler = EnsembleTrainer(num_classes=8, device=device)
    our_results = ensembler.sequential_batch_evaluation(test_loader)  # type: ignore
