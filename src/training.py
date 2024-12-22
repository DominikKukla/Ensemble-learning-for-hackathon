import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torchvision import models

from utils.data_loading import prepare_data_loaders
from utils.utils import set_seed


class EnsembleTrainer:
    def __init__(self, num_classes, device, patience=10, min_delta=0.01, max_epochs=50):
        self.max_epochs = max_epochs
        self.device = device
        self.patience = patience
        self.min_delta = min_delta

        self.model_weights = dict()

        self.models = {
            "efficientnet_b0": self._prepare_efficientnet_b0(num_classes),
            "efficientnet_b1": self._prepare_efficientnet_b1(num_classes),
            "mobilenet_v3_large": self._prepare_mobilenet_v3_large(num_classes),
            "shufflenet_v2_x2_0": self._prepare_shufflenet_v2_x2_0(num_classes),
        }

        # Optimizers for each model
        self.optimizers = {
            name: optim.AdamW(model.parameters()) for name, model in self.models.items()
        }

        self.criterion = nn.CrossEntropyLoss()

        # Tracking metrics
        self.results = {
            name: {
                "train_losses": [],
                "val_losses": [],
                "f1_scores": [],
                "best_val_f1": 0,
                "patience_counter": 0,
            }
            for name in self.models.keys()
        }

    def _prepare_efficientnet_b0(self, num_classes):
        model = models.efficientnet_b0(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model.to(self.device)

    def _prepare_efficientnet_b1(self, num_classes):
        model = models.efficientnet_b1(weights="IMAGENET1K_V2")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model.to(self.device)

    def _prepare_mobilenet_v3_large(self, num_classes):
        model = models.mobilenet_v3_large(weights="IMAGENET1K_V2")
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model.to(self.device)

    def _prepare_shufflenet_v2_x2_0(self, num_classes):
        model = models.shufflenet_v2_x2_0(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(self.device)

    def train_single_model(
        self, model, optimizer, train_loader, validation_loader, model_name
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
                optimizer.step()

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
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Compute metrics
            avg_train_loss = epoch_loss_train / len(train_loader)
            avg_val_loss = epoch_loss_val / len(validation_loader)
            val_f1 = f1_score(all_labels, all_preds, average="macro")

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
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": results["train_losses"],
                        "val_loss": results["val_losses"],
                        "f1_metric_val": results["f1_scores"],
                        "best_val_f1": results["best_val_f1"],
                    },
                    f"src/data/models/{model_name}_best_f1.pth",
                )
                self.model_weights[model_name] = val_f1
            else:
                results["patience_counter"] += 1

            if results["patience_counter"] >= self.patience:
                print(
                    f"{model_name} - Early stopping triggered at epoch {epoch}, best f1 score {results['f1_scores'][-11]}"
                )
                break

    def train_ensemble(self, train_loader, validation_loader):
        # Train each model
        for name, model in self.models.items():
            print(f"\nTraining {name}")
            self.train_single_model(
                model, self.optimizers[name], train_loader, validation_loader, name
            )

    def ensemble_predict(self, dataloader):
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
            weighted_preds = []
            for model_name, model in self.models.items():
                probs = torch.softmax(model(images), dim=1)
                weight = model_weights_normalized.get(
                    model_name, 0.0
                )  # Domyślna waga 0, jeśli model nie ma wagi
                weighted_preds.append(probs * weight)

            # Suma ważonych predykcji
            ensemble_probs = torch.sum(torch.stack(weighted_preds), dim=0)

            # Ostateczne predykcje
            _, ensemble_preds = torch.max(ensemble_probs, 1)

            all_ensemble_preds.extend(ensemble_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Obliczenie F1-score dla ostatecznych predykcji
        ensemble_f1 = f1_score(all_labels, all_ensemble_preds, average="macro")
        print(f"Weighted Ensemble F1 Score: {ensemble_f1:.4f}")

        return ensemble_f1


# -------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(23)

train_loader, validation_loader = prepare_data_loaders((224, 224), 1)

ensemble_trainer = EnsembleTrainer(num_classes=8, device=device, max_epochs=50)

# Train all models
ensemble_trainer.train_ensemble(train_loader, validation_loader)

# Perform ensemble prediction
ensemble_f1 = ensemble_trainer.ensemble_predict(validation_loader)
