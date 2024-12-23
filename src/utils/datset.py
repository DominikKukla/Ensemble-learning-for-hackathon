from pathlib import Path
from typing import Any, Tuple

import kaggle  # type: ignore
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms  # type: ignore


class Data:
    def __init__(self) -> None:
        kaggle.api.authenticate()
        root_folder = Path(__file__).parent.parent.parent
        self.download_folder = root_folder / "data"
        self.size_tuple = (224, 224)

    def download_from_kaggle(self) -> None:
        self.download_folder.mkdir(exist_ok=True)
        kaggle.api.dataset_download_files(  # type: ignore
            "dzmitrypihulski/hackathon-dataset",
            path=self.download_folder,
            unzip=True,
            quiet=False,
        )

    def load_train_val_data(
        self, augmented_count: int = 0, validation_split: float = 0.2
    ) -> Tuple[DataLoader[Any], DataLoader[Any]]:
        # Define the path to the root directory containing the class folders
        train_folder = self.download_folder / "train_data"

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

        # Data Augmentation and Transformations
        transform = transforms.Compose(
            [
                transforms.Resize(self.size_tuple),
                transforms.RandomHorizontalFlip(),  # Random horizontal flip
                transforms.RandomCrop(
                    size=(self.size_tuple[0] - 20, self.size_tuple[1] - 20), padding=20
                ),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, hue=0.1),
                transforms.RandomVerticalFlip(0.1),
                transforms.Resize(self.size_tuple),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load the original dataset
        original_dataset = datasets.ImageFolder(
            root=train_folder,
            transform=transforms.Compose(
                [
                    transforms.Resize(self.size_tuple),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )

        original_dataset.class_to_idx = {
            k: class_mapping[k] for k in original_dataset.class_to_idx.keys()
        }

        for _ in range(augmented_count):
            # Load the original dataset
            augmented_dataset = datasets.ImageFolder(
                root=train_folder, transform=transform
            )

            augmented_dataset.class_to_idx = {
                k: class_mapping[k] for k in augmented_dataset.class_to_idx.keys()
            }

            # Combine original and augmented datasets
            original_dataset: ConcatDataset[Any] | datasets.ImageFolder = ConcatDataset(  # type: ignore
                [original_dataset, augmented_dataset]
            )

        # Split into train and validation sets
        dataset_size = len(original_dataset)
        validation_size = int(validation_split * dataset_size)
        train_size = dataset_size - validation_size
        train_dataset, validation_dataset = random_split(
            original_dataset, [train_size, validation_size]
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False
        )
        return (train_loader, validation_loader)

    def load_test_data(self) -> DataLoader[Any]:
        test_folder = self.download_folder / "test_data"

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
            root=test_folder,
            transform=transforms.Compose(
                [
                    transforms.Resize(self.size_tuple),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )

        test_dataset.class_to_idx = {
            k: class_mapping[k] for k in test_dataset.class_to_idx.keys()
        }

        # Create data loader
        test_loader: DataLoader[Any] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True
        )
        return test_loader
