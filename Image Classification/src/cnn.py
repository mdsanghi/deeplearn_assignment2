from pathlib import Path
import pickle
from sys import path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


DATASET_DIR = Path(__file__).resolve().parent.parent / "cifar-100-python"


def load_cifar100(split="train", dataset_dir=DATASET_DIR):
    """Load one CIFAR-100 split from the extracted cifar-100-python folder."""
    split_path = dataset_dir / split
    meta_path = dataset_dir / "meta"
    
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    
    with open(split_path, "rb") as file:
        split_data = pickle.load(file, encoding="bytes")

    with open(meta_path, "rb") as file:
        meta_data = pickle.load(file, encoding="bytes")

    images = split_data[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    fine_labels = np.array(split_data[b"fine_labels"])
    fine_label_names = [label.decode("utf-8") for label in meta_data[b"fine_label_names"]]
    filenames = [name.decode("utf-8") for name in split_data[b"filenames"]]

    return images, fine_labels, fine_label_names, filenames


def show_sample_images(split="train", num_samples=12):
    """Display a grid of sample images with their fine-label names."""
    images, labels, label_names, filenames = load_cifar100(split=split)

    num_samples = min(num_samples, len(images))
    cols = 4
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for index in range(num_samples):
        axes[index].imshow(images[index])
        axes[index].set_title(f"{label_names[labels[index]]}\n{filenames[index]}", fontsize=9)
        axes[index].axis("off")

    for index in range(num_samples, len(axes)):
        axes[index].axis("off")

    fig.suptitle(f"CIFAR-100 sample images from '{split}'", fontsize=14)
    plt.tight_layout()
    plt.show()


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def preprocess_data(images, labels):
    """Preprocess images and labels for training."""
    # Normalize images to [0, 1]
    images = images.astype(np.float32) / 255.0
    # Convert to torch tensors and transpose to (N, C, H, W)
    images = torch.tensor(images).permute(0, 3, 1, 2)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


def print_train_data_features():
    """Read and print all features of the train data."""
    print("="*80)
    print("TRAIN DATA FEATURES ANALYSIS")
    print("="*80)
    
    # Load raw train data
    train_images, train_labels, label_names, filenames = load_cifar100(split="train")
    
    print("\n1. DATASET SHAPE AND SIZE:")
    print(f"   Number of training samples: {len(train_images)}")
    print(f"   Image shape (raw): {train_images.shape}")
    print(f"   Labels shape: {train_labels.shape}")
    print(f"   Number of unique classes: {len(label_names)}")
    print(f"   Image dimensions (H x W x C): {train_images[0].shape}")
    
    print("\n2. DATA TYPES:")
    print(f"   Images dtype: {train_images.dtype}")
    print(f"   Labels dtype: {train_labels.dtype}")
    print(f"   Labels data: {type(train_labels)}")
    
    print("\n3. IMAGE PIXEL VALUE STATISTICS:")
    print(f"   Min pixel value: {train_images.min()}")
    print(f"   Max pixel value: {train_images.max()}")
    print(f"   Mean pixel value: {train_images.mean():.4f}")
    print(f"   Std pixel value: {train_images.std():.4f}")
    
    print("\n4. LABEL DISTRIBUTION:")
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    print(f"   Unique labels: {len(unique_labels)}")
    print(f"   Min samples per class: {counts.min()}")
    print(f"   Max samples per class: {counts.max()}")
    print(f"   Mean samples per class: {counts.mean():.2f}")
    print(f"   Std samples per class: {counts.std():.2f}")
    
    print("\n5. CLASS NAMES:")
    for i, name in enumerate(label_names[:20]):
        print(f"   Class {i}: {name}")
    print(f"   ... and {len(label_names) - 20} more classes")
    
    print("\n6. MEMORY USAGE:")
    images_memory = train_images.nbytes / (1024**2)
    labels_memory = train_labels.nbytes / (1024**2)
    total_memory = images_memory + labels_memory
    print(f"   Images memory: {images_memory:.2f} MB")
    print(f"   Labels memory: {labels_memory:.2f} MB")
    print(f"   Total memory: {total_memory:.2f} MB")
    
    print("\n7. SAMPLE LABELS AND FILENAMES:")
    for i in range(min(10, len(train_labels))):
        print(f"   Sample {i}: Label={train_labels[i]} ({label_names[train_labels[i]]}), Filename={filenames[i]}")
    
    print("\n8. CHANNEL-WISE STATISTICS:")
    for c in range(3):
        channel_data = train_images[:, :, :, c]
        print(f"   Channel {c} - Min: {channel_data.min()}, Max: {channel_data.max()}, Mean: {channel_data.mean():.4f}")
    
    print("="*80 + "\n")



def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')


def evaluate_model(model, test_loader, device='cpu'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Uncommet this to understand data configuration and features before training - start
    # Print train data features and analysis
    # print_train_data_features()
    
    # Show sample images
    # print("Showing train images...")
    # show_sample_images(split="train", num_samples=12)
    
    # print("Showing test images...")
    # show_sample_images(split="test", num_samples=12)
    # Uncommet this to understand data configuration and features before training - ends
    
    # Load and preprocess training data from train path
    train_path = DATASET_DIR / "train"
    print(f"Loading training data from: {train_path}")
    train_images, train_labels, _, _ = load_cifar100(split="train")
    train_images, train_labels = preprocess_data(train_images, train_labels)
    
    # Load and preprocess test data from test path
    test_path = DATASET_DIR / "test"
    print(f"Loading test data from: {test_path}")
    test_images, test_labels, _, _ = load_cifar100(split="test")
    test_images, test_labels = preprocess_data(test_images, test_labels)
    
    # Create data loaders
    batch_size = 64
    train_dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = SimpleCNN(num_classes=100)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=10, device=device)
    
    # Evaluate the model
    print("Evaluating on test set...")
    evaluate_model(model, test_loader, device=device)
    
   