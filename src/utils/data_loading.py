from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms


def prepare_data_loaders(size_tuple, augmented_count):
    # Define the path to the root directory containing the class folders

    root_dir = "src/data/data"

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
    validation_split = 0.2  # Proportion of data for validation

    # Data Augmentation and Transformations
    transform = transforms.Compose(
        [
            transforms.Resize(size_tuple),
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.RandomCrop(
                size=(size_tuple[0] - 20, size_tuple[1] - 20), padding=20
            ),  # Random crop
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, hue=0.1),
            transforms.RandomVerticalFlip(0.1),
            transforms.Resize(size_tuple),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize
        ]
    )

    # Load the original dataset
    original_dataset = datasets.ImageFolder(
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

    original_dataset.class_to_idx = {
        k: class_mapping[k] for k in original_dataset.class_to_idx.keys()
    }

    for _ in range(augmented_count):
        # Load the original dataset
        augmented_dataset = datasets.ImageFolder(root=root_dir, transform=transform)

        augmented_dataset.class_to_idx = {
            k: class_mapping[k] for k in augmented_dataset.class_to_idx.keys()
        }

        # Combine original and augmented datasets
        original_dataset = ConcatDataset([original_dataset, augmented_dataset])

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
