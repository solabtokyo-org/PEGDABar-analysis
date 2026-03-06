import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from fl_bar_processing.config import FOCUS_CHECK


class FocusClassifier(nn.Module):
    def __init__(self):
        super(FocusClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# 256×256 crop dataset (16-bit compatible)


class CropDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.data = []

        print("Generating 256×256 crops (16-bit processing)...")
        for path, label in zip(image_paths, labels):
            # Generate 8×8=64 crops
            for row in range(8):
                for col in range(8):
                    crop_x, crop_y = col * 256, row * 256
                    self.data.append((path, label, crop_x, crop_y))

        self.transform = transform
        print(f"Total number of crops: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label, crop_x, crop_y = self.data[idx]

        # Load 16-bit image as is
        image = Image.open(path)

        if image.mode == "I;16":
            # Process 16-bit image as a numpy array
            img_array = np.array(image, dtype=np.float32)
            # Normalize to range 0-1 (divide by 65535)
            img_array = img_array / 65535.0

            # Crop to 256×256
            cropped = img_array[crop_y : crop_y + 256, crop_x : crop_x + 256]

            # Convert to Tensor (CHW format: 1×256×256)
            image_tensor = torch.FloatTensor(cropped).unsqueeze(0)

        else:
            # If not 16-bit, convert to grayscale
            image = image.convert("L")
            image = image.crop((crop_x, crop_y, crop_x + 256, crop_y + 256))

            if self.transform:
                image_tensor = self.transform(image)
            else:
                # Alternative to ToTensor
                image_tensor = torch.FloatTensor(np.array(image)).unsqueeze(0) / 255.0

        # Normalize (mean 0.5, std 0.5 to range -1 to 1)
        image_tensor = (image_tensor - 0.5) / 0.5

        return image_tensor, label


def get_transforms():
    # For 16-bit data, normalization is already handled in CropDataset, so only simple transformations are needed
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
        ]
    )

    val_transform = transforms.Compose(
        [
            # No transformations during validation
        ]
    )

    return train_transform, val_transform


def train_model(model, train_loader, val_loader, epochs=20, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    best_acc = 0

    print(f"{'Epoch':>5} {'Train Loss':>10} {'Train Acc':>10} {'Val Acc':>10}")
    print("-" * 40)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_loss = train_loss / len(train_loader)

        print(f"{epoch + 1:5d} {avg_loss:10.4f} {train_acc:9.1f}% {val_acc:9.1f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(), "../focus_classifier_results/modlesbest_model.pth"
            )

    print(f"\nBest Accuracy: {best_acc:.1f}%")
    model.load_state_dict(
        torch.load("../focus_classifier_results/modlesbest_model.pth")
    )
    return model


def predict_image(model, image_path, device="cpu"):
    model.eval()

    image = Image.open(image_path)

    if image.mode == "I;16":
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array / 65535.0  # Normalize to 0-1
    else:
        image = image.convert("L")
        img_array = np.array(image, dtype=np.float32) / 255.0

    votes = []
    for row in range(8):
        for col in range(8):
            crop_x, crop_y = col * 256, row * 256

            crop_array = img_array[crop_y : crop_y + 256, crop_x : crop_x + 256]
            crop_tensor = (
                torch.FloatTensor(crop_array).unsqueeze(0).unsqueeze(0)
            )  # 1×1×256×256

            crop_tensor = (crop_tensor - 0.5) / 0.5
            crop_tensor = crop_tensor.to(device)

            with torch.no_grad():
                output = model(crop_tensor)
                prob = torch.softmax(output, dim=1)
                pred = torch.argmax(output, dim=1).item()
                conf = torch.max(prob).item()
                votes.append((pred, conf))

    # Majority vote
    predictions = [v[0] for v in votes]
    confidences = [v[1] for v in votes]

    final_pred = max(set(predictions), key=predictions.count)
    avg_conf = np.mean(confidences)

    class_names = ["blurred", "focused"]
    return class_names[final_pred], avg_conf


def get_focus_check_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    focused_dir = "../data/focused"
    blurred_dir = "../data/blurred"

    image_paths = []
    labels = []

    for img_name in os.listdir(focused_dir):
        if img_name.lower().endswith(".tif"):
            image_paths.append(os.path.join(focused_dir, img_name))
            labels.append(1)

    for img_name in os.listdir(blurred_dir):
        if img_name.lower().endswith(".tif"):
            image_paths.append(os.path.join(blurred_dir, img_name))
            labels.append(0)

    print(f"Total images: {len(image_paths)}")
    focused_count = sum(labels)
    blurred_count = len(labels) - focused_count
    print(f"focused: {focused_count}, blurred: {blurred_count}")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Training images: {len(train_paths)}, Validation images: {len(val_paths)}")

    train_transform, val_transform = get_transforms()

    train_dataset = CropDataset(train_paths, train_labels, train_transform)
    val_dataset = CropDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = FocusClassifier()
    model = train_model(model, train_loader, val_loader, epochs=20, device=device)

    torch.save(
        model.state_dict(), "../focus_classifier_results/modles/focus_classifier.pth"
    )

    for val_path in val_paths:
        result, confidence = predict_image(model, val_path, device)
        print("\nTest Prediction:")
        print(f"Image: {val_path}")
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence:.3f}")


def load_and_predict(image_path, model_path=FOCUS_CHECK["model_path"]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FocusClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    _, transform = get_transforms()

    return predict_image(model, image_path, device)
