from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score  # type: ignore
from torch.utils.data import Subset
from torchvision import models  # type: ignore

from utils.datset import Data
from utils.models import Models
from utils.utils import set_seed


class EnsembleTester:
    def __init__(
        self,
        models_loader: Models,
        num_classes: int = 8,
        patience: int = 10,
        min_delta: float = 0.01,
        max_epochs: int = 50,
    ):
        root_folder = Path(__file__).parent.parent
        self.models_folder = root_folder / "data" / "models"
        self.max_epochs = max_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patience = patience
        self.min_delta = min_delta

        self.models: Dict[str, Any] = {
            "efficientnet_b0": models_loader.prepare_finetuned_efficientnet_b0(
                "efficientnet_b0", num_classes
            ),
            "efficientnet_b1": models_loader.prepare_finetuned_efficientnet_b1(
                "efficientnet_b1", num_classes
            ),
            "mobilenet_v3_large": models_loader.prepare_finetuned_mobilenet_v3_large(
                "mobilenet_v3_large", num_classes
            ),
            "shufflenet_v2_x2_0": models_loader.prepare_finetuned_shufflenet_v2_x2_0(
                "shufflenet_v2_x2_0", num_classes
            ),
        }

        self.model_weights = models_loader.model_weights

        # Optimizers for each model
        self.optimizers = {
            name: optim.AdamW(model.parameters()) for name, model in self.models.items()
        }

        self.criterion = nn.CrossEntropyLoss()

        # Tracking metrics
        self.results: Dict[str, Dict[str, Any]] = {
            name: {
                "train_losses": [],
                "val_losses": [],
                "f1_scores": [],
                "best_val_f1": 0,
                "patience_counter": 0,
            }
            for name in self.models.keys()
        }

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


# -------------------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(23)

    data_loader = Data()
    # data_loader.download_from_kaggle()

    models = Models()
    # models.download_from_kaggle()

    test_loader = data_loader.load_test_data()

    ensemble_tester = EnsembleTester(num_classes=8, max_epochs=50, models_loader=models)

    ensemble_f1 = ensemble_tester.sequential_batch_evaluation(test_loader)  # type: ignore
