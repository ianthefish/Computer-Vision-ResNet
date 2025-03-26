import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import random
from torch.optim.lr_scheduler import OneCycleLR
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Dataset paths
dataset_dir = "./data"
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")
val_dir = os.path.join(dataset_dir, "val")


# Set seed for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

stats = ((0.4914, 0.4828, 0.4475), (0.2471, 0.2436, 0.2617))

# Define transformations using albumentations
train_transform = A.Compose(
    [
        A.Resize(512, 512),
        A.RandomCrop(384, 384),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
        ),
        A.OneOf(
            [
                A.CoarseDropout(p=0.5),
                A.GridDistortion(p=0.5),
            ],
            p=0.5,
        ),
        A.Normalize(mean=stats[0], std=stats[1]),
        ToTensorV2(),
    ]
)

valid_transform = A.Compose(
    [
        A.Resize(512, 512),
        A.CenterCrop(384, 384),
        A.Normalize(mean=stats[0], std=stats[1]),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Resize(512, 512),
        A.CenterCrop(384, 384),
        A.Normalize(mean=stats[0], std=stats[1]),
        ToTensorV2(),
    ]
)


# Wrapper datasets for albumentations
class AlbumentationsDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.image_paths = [
            os.path.join(root, filename)
            for root, _, filenames in os.walk(dataset_dir)
            for filename in filenames
            if not filename.startswith(".")
        ]
        self.transform = transform

        # Extract class labels from directory structure
        self.class_names = sorted(
            [
                d
                for d in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, d))
            ]
        )
        self.class_map = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }

        # Get labels for each image
        self.labels = []
        for path in self.image_paths:
            class_name = os.path.basename(os.path.dirname(path))
            self.labels.append(self.class_map[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


# Custom dataset for test set (no labels)
class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_filenames = sorted(
            [f for f in os.listdir(image_dir) if not f.startswith(".")]
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, self.image_filenames[idx]


# Load datasets
train_ds = AlbumentationsDataset(train_dir, train_transform)
valid_ds = AlbumentationsDataset(val_dir, valid_transform)
test_ds = TestDataset(test_dir, test_transform)

# Data loaders
batch_size = 16  # Smaller batch size for better generalization
num_workers = 4

train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)
valid_dl = DataLoader(
    valid_ds,
    batch_size=batch_size * 2,
    num_workers=num_workers,
    pin_memory=True,
)
test_dl = DataLoader(
    test_ds,
    batch_size=batch_size * 2,
    num_workers=num_workers,
    pin_memory=True,
)


# Define different ResNet models for ensemble
def create_model(model_name="resnet50", num_classes=None):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes)
        )
    elif model_name == "resnet101":
        model = models.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V2
        )
        model.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes)
        )
    elif model_name == "resnext50_32x4d":
        model = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
        )
        model.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes)
        )
    elif model_name == "wide_resnet50_2":
        model = models.wide_resnet50_2(
            weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2
        )
        model.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes)
        )
    elif model_name == "resnet152":
        model = models.resnet152(
            weights=models.ResNet152_Weights.IMAGENET1K_V2
        )
        model.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Replace the classifier
    if num_classes is None:
        num_classes = len(train_ds.class_names)

    # Add dropout to prevent overfitting
    # model.fc = nn.Sequential(
    #     # nn.Dropout(0.5),
    #     nn.Dropout(0.7),
    #     nn.Linear(model.fc.in_features, num_classes)
    # )

    return model.to(device)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# Function for Mixup and CutMix Data Augmentation
def mixup_data(x, y, alpha=1.0, cutmix_prob=0.5, beta=1.0):
    use_cutmix = np.random.rand() < cutmix_prob

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    if alpha > 0 and beta > 0:
        if use_cutmix:
            # CutMix
            lam = np.random.beta(beta, beta)
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2])
            )
        else:
            # Mixup
            lam = np.random.beta(alpha, alpha)
            x = lam * x + (1 - lam) * x[index]
    else:
        lam = 1

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Training function with additional optimizations
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=30,
    model_name="model",
    mixup_alpha=0.4,
    cutmix_beta=0.4,
    patience=8,
):
    best_val_acc = 0.0
    counter = 0
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            # Apply mixup or cutmix
            images, labels_a, labels_b, lam = mixup_data(
                images, labels, mixup_alpha, cutmix_prob=0.5, beta=cutmix_beta
            )

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)

            # Calculate loss
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

            # Backward and optimize
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Training metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)

            # Approximation of accuracy for mixed samples
            correct_a = (predicted == labels_a).float()
            correct_b = (predicted == labels_b).float()
            correct_predictions += (
                (lam * correct_a + (1 - lam) * correct_b).sum().item()
            )

            train_bar.set_postfix(
                loss=loss.item(), acc=correct_predictions / total_samples
            )

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_acc = correct_predictions / total_samples
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_losses = []

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(
                    device
                )
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)
                val_losses.append(val_loss.item())

                _, val_predicted = torch.max(val_outputs, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_loss = sum(val_losses) / len(val_losses)
        val_accuracy = val_correct / val_total
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, \
              Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f},  \
              Val Acc: {val_accuracy:.4f}"
        )

        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), f"{model_name}_best.pth")
            print(
                f"New best model saved with validation accuracy: \
                {val_accuracy:.4f}"
            )
            counter = 0
        else:
            counter += 1

        # Early stopping
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": (
                        scheduler.state_dict() if scheduler else None
                    ),
                    "best_val_acc": best_val_acc,
                },
                f"{model_name}_epoch_{epoch+1}.pth",
            )

    return train_losses, val_accuracies, best_val_acc


def test_ensemble(ensemble, test_loader, class_names):
    predictions, filenames = [], []
    test_bar = tqdm(test_loader, desc="Testing")

    # Create a list to hold model weights if you want to use weighted averaging
    # For simplicity, we'll use equal weights here

    with torch.no_grad():
        for images, file_names in test_bar:
            images = images.to(device)

            # Collect outputs from all models with TTA
            ensemble_predictions = []

            for model in ensemble:
                model.eval()
                # Original image
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                ensemble_predictions.append(predicted.cpu().numpy())

                # Test-time augmentation (TTA)
                # Horizontal flip
                flipped_images = torch.flip(images, dims=[3])
                flipped_outputs = model(flipped_images)
                _, flipped_predicted = torch.max(flipped_outputs, 1)
                ensemble_predictions.append(flipped_predicted.cpu().numpy())

                # Center crop
                h, w = images.shape[2:]
                crop_size = int(min(h, w) * 0.9)
                start_h = (h - crop_size) // 2
                start_w = (w - crop_size) // 2
                cropped_images = images[
                    :,
                    :,
                    start_h: start_h + crop_size,
                    start_w: start_w + crop_size,
                ]
                cropped_images = F.interpolate(
                    cropped_images,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
                cropped_outputs = model(cropped_images)
                _, cropped_predicted = torch.max(cropped_outputs, 1)
                ensemble_predictions.append(cropped_predicted.cpu().numpy())

            # Use mode (most frequent prediction) for each sample
            ensemble_predictions = np.array(ensemble_predictions).T
            mode_predictions = stats.mode(ensemble_predictions, axis=1)[
                0
            ].flatten()

            predictions.extend([class_names[p] for p in mode_predictions])

            # Remove .jpg extension from filenames
            clean_filenames = [
                os.path.splitext(fname)[0] for fname in file_names
            ]
            filenames.extend(clean_filenames)

    return predictions, filenames


def plot_training_curves(
    train_losses_list,
    val_accuracies_list,
    model_names,
    save_path="training_curves.png",
):
    """
    Plot training losses and validation accuracies for multiple models.

    Parameters:
    - train_losses_list: List of lists containing training losses
    - val_accuracies_list: List of lists containing validation accuracies
    - model_names: List of model names to the losses and accuracies
    - save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))

    # Subplot for Training Loss
    plt.subplot(2, 1, 1)
    plt.title("Training Loss Curves", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Training Loss", fontsize=12)

    for losses, name in zip(train_losses_list, model_names):
        plt.plot(range(1, len(losses) + 1), losses, label=name)

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Subplot for Validation Accuracy
    plt.subplot(2, 1, 2)
    plt.title("Validation Accuracy Curves", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)

    for accuracies, name in zip(val_accuracies_list, model_names):
        plt.plot(range(1, len(accuracies) + 1), accuracies, label=name)

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    os.makedirs("./improved_ensemble", exist_ok=True)

    # Define ensemble models with different ResNet variants (removed resnet152)
    model_configs = [
        {"name": "resnet50", "epochs": 60, "lr": 1e-3},
        {"name": "resnet101", "epochs": 60, "lr": 5e-4},
        {"name": "resnext50_32x4d", "epochs": 60, "lr": 1e-3},
        {"name": "wide_resnet50_2", "epochs": 60, "lr": 1e-3},
        {"name": "resnet152", "epochs": 60, "lr": 3e-4},
    ]

    num_classes = len(train_ds.class_names)
    criterion = LabelSmoothingLoss(classes=num_classes, smoothing=0.1)
    trained_models = []
    best_accs = []

    train_losses_list = []
    val_accuracies_list = []
    model_names = []

    for i, config in enumerate(model_configs):
        print(f"\n{'='*50}")
        print(f"Training model {i+1}/{len(model_configs)}: {config['name']}")
        print(f"{'='*50}")

        # Create model
        model = create_model(config["name"], num_classes)

        # Freeze the first few layers to prevent overfitting
        ct = 0
        for child in model.children():
            ct += 1
            if ct < 6:  # Freeze first few layers
                for param in child.parameters():
                    param.requires_grad = False

        # Prepare optimizer with differential learning rates
        params = [
            {
                "params": [
                    p for n, p in model.named_parameters() if "fc" not in n
                ],
                "lr": config["lr"] / 10,
            },
            {"params": model.fc.parameters(), "lr": config["lr"]},
        ]

        optimizer = optim.AdamW(params, weight_decay=1e-4)

        # Use OneCycleLR scheduler for better convergence
        steps_per_epoch = len(train_dl)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[config["lr"] / 10, config["lr"]],
            steps_per_epoch=steps_per_epoch,
            epochs=config["epochs"],
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100,
        )

        # Train the model
        train_losses, val_accuracies, best_acc = train_model(
            model=model,
            train_loader=train_dl,
            val_loader=valid_dl,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config["epochs"],
            model_name=f"./improved_ensemble/{config['name']}",
            mixup_alpha=0.4,
            cutmix_beta=0.4,
            patience=12,
        )

        train_losses_list.append(train_losses)
        val_accuracies_list.append(val_accuracies)
        model_names.append(config["name"])

        plot_training_curves(
            train_losses_list,
            val_accuracies_list,
            model_names,
            save_path="./improved_ensemble/training_curves.png",
        )

        print(
            "\nTraining curves saved to \
            './improved_ensemble/training_curves.png'"
        )

        # Load the best model weights
        model.load_state_dict(
            torch.load(f"./improved_ensemble/{config['name']}_best.pth")
        )
        trained_models.append(model)
        best_accs.append(best_acc)

        print(
            f"Model {i+1} ({config['name']}) best validation accuracy: \
            {best_acc:.4f}"
        )

    # Print ensemble summary
    print("\nEnsemble Summary:")
    for i, (config, acc) in enumerate(zip(model_configs, best_accs)):
        print(
            f"Model {i+1}: {config['name']} - Validation Accuracy: {acc:.4f}"
        )

    # Make predictions with the ensemble
    print(
        "\nGenerating predictions with ensemble and test-time augmentation..."
    )
    predictions, filenames = test_ensemble(
        trained_models, test_dl, train_ds.class_names
    )

    # Save predictions to CSV
    df = pd.DataFrame({"image_name": filenames, "pred_label": predictions})
    df.to_csv("prediction.csv", index=False)
    print("Predictions saved to 'prediction.csv'.")
