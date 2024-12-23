import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score  # type: ignore
from torch.utils.data import DataLoader
from torchvision import models  # type: ignore

from utils.datset import Data
from utils.models import Models
from utils.utils import set_seed


class EnsembleTrainer:
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
            "efficientnet_b0": models_loader.create_new_efficientnet_b0(num_classes),
            "efficientnet_b1": models_loader.create_new_efficientnet_b1(num_classes),
            "mobilenet_v3_large": models_loader.create_new_mobilenet_v3_large(
                num_classes
            ),
            "shufflenet_v2_x2_0": models_loader.create_new_shufflenet_v2_x2_0(
                num_classes
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

    def _train_single_model(
        self,
        model: Any,
        optimizer: optim.AdamW,
        train_loader: DataLoader[Any],
        validation_loader: DataLoader[Any],
        model_name: str,
    ):
        for epoch in range(1, self.max_epochs + 1):  # Your original epoch range
            current_time = time.time()

            # Training phase
            model.train()
            epoch_loss_train = 0
            running_loss = 0

            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()  # type: ignore

                running_loss += loss.item()
                epoch_loss_train += loss.item()

                if i % 20 == 19:
                    print(
                        f"{model_name} - Current time: {round(time.time() - current_time, 0)} s, "
                        f"epoch: {epoch}/50, minibatch: {i + 1:5d}/{len(train_loader)}, "
                        f"running loss: {running_loss / 500:.3f}"
                    )
                    running_loss = 0

            # Validation phase
            model.eval()
            epoch_loss_val = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for images, labels in validation_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = self.criterion(outputs, labels)
                    epoch_loss_val += loss.item()

                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())  # type: ignore
                    all_labels.extend(labels.cpu().numpy())  # type: ignore

            # Compute metrics
            avg_train_loss = epoch_loss_train / len(train_loader)
            avg_val_loss = epoch_loss_val / len(validation_loader)
            val_f1 = float(f1_score(all_labels, all_preds, average="macro"))  # type: ignore

            # Update results
            results = self.results[model_name]
            results["train_losses"].append(avg_train_loss)
            results["val_losses"].append(avg_val_loss)
            results["f1_scores"].append(val_f1)

            print(
                f"{model_name} - Epoch {epoch}: "
                f"Train Loss = {avg_train_loss:.4f}, "
                f"Val Loss = {avg_val_loss:.4f}, "
                f"Val F1 = {val_f1:.4f}"
            )

            # Early stopping
            if val_f1 > results["best_val_f1"] + self.min_delta:
                results["best_val_f1"] = val_f1
                results["patience_counter"] = 0

                # Save best model
                torch.save(  # type: ignore
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": results["train_losses"],
                        "val_loss": results["val_losses"],
                        "f1_metric_val": results["f1_scores"],
                        "best_val_f1": results["best_val_f1"],
                    },  # type: ignore
                    str(self.models_folder) + f"/{model_name}_best_f1.pth",
                )
                self.model_weights[model_name] = val_f1
            else:
                results["patience_counter"] += 1

            if results["patience_counter"] >= self.patience:
                print(
                    f"{model_name} - Early stopping triggered at epoch {epoch}, best f1 score {results['f1_scores'][-11]}"
                )
                break

    def train_ensemble(
        self, train_loader: DataLoader[Any], validation_loader: DataLoader[Any]
    ):
        # Train each model
        for name, model in self.models.items():
            print(f"\nTraining {name}")
            self._train_single_model(
                model, self.optimizers[name], train_loader, validation_loader, name
            )

    def plot_models(self, Models: Models):
        # Plot each model
        for name in self.models.keys():
            Models.plot_training_metrics(
                str(self.models_folder) + f"/{name}_best_f1.pth", name
            )

    def ensemble_predict(self, dataloader: DataLoader[Any]) -> float:
        # Weighted voting ensemble prediction
        all_ensemble_preds = []
        all_labels = []

        # Normalizacja wag, jeśli nie sumują się do 1
        total_weight = sum(self.model_weights.values())
        model_weights_normalized = {
            model: weight / total_weight for model, weight in self.model_weights.items()
        }

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zbieramy predykcje każdego modelu pomnożone przez ich wagi
            weighted_preds: List[Any] = []
            for model_name, model in self.models.items():
                probs = torch.softmax(model(images), dim=1)
                weight = model_weights_normalized.get(
                    model_name, 0.0
                )  # Domyślna waga 0, jeśli model nie ma wagi
                weighted_preds.append(probs * weight)  # type: ignore

            # Suma ważonych predykcji
            ensemble_probs = torch.sum(torch.stack(weighted_preds), dim=0)

            # Ostateczne predykcje
            _, ensemble_preds = torch.max(ensemble_probs, 1)

            all_ensemble_preds.extend(ensemble_preds.cpu().numpy())  # type: ignore
            all_labels.extend(labels.cpu().numpy())  # type: ignore

        # Obliczenie F1-score dla ostatecznych predykcji
        ensemble_f1 = f1_score(all_labels, all_ensemble_preds, average="macro")  # type: ignore
        print(f"Weighted Ensemble F1 Score: {ensemble_f1:.4f}")

        return float(ensemble_f1)  # type: ignore


# -------------------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(23)

    data_loader = Data()
    data_loader.download_from_kaggle()

    models = Models()
    models.download_from_kaggle()

    train_loader, validation_loader = data_loader.load_train_val_data(1)

    ensemble_trainer = EnsembleTrainer(
        num_classes=8, max_epochs=50, models_loader=models
    )

    # Train all models
    ensemble_trainer.train_ensemble(train_loader, validation_loader)

    # Plot all models
    ensemble_trainer.plot_models(models)

    # Perform ensemble prediction
    ensemble_f1 = ensemble_trainer.ensemble_predict(validation_loader)
