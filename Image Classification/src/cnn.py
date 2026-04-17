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


def print_cnn_architecture():
    """Print detailed description of the CNN architecture."""
    print("\n" + "="*100)
    print("CNN ARCHITECTURE DESCRIPTION FOR CIFAR-100 IMAGE CLASSIFICATION")
    print("="*100)
    
    print("\n📋 OVERVIEW:")
    print("   Model Name: SimpleCNN")
    print("   Task: Image Classification")
    print("   Dataset: CIFAR-100 (32x32 RGB images, 100 classes)")
    print("   Architecture Type: Convolutional Neural Network (CNN)")
    
    print("\n" + "-"*100)
    print("1️⃣  INPUT LAYER")
    print("-"*100)
    print("   Input Shape: (Batch_Size, 3, 32, 32)")
    print("   - Batch Size: Variable (typically 64)")
    print("   - Channels: 3 (RGB color images)")
    print("   - Height: 32 pixels")
    print("   - Width: 32 pixels")
    
    print("\n" + "-"*100)
    print("2️⃣  CONVOLUTIONAL BLOCK 1")
    print("-"*100)
    print("   Layer: Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)")
    print("   - Input: (Batch, 3, 32, 32)")
    print("   - Kernel: 3x3 filters (32 filters)")
    print("   - Padding: 1 (preserves spatial dimensions)")
    print("   - Number of Parameters: 3 * 3 * 3 * 32 + 32 = 896")
    print("   - Output Shape: (Batch, 32, 32, 32)")
    print("   ")
    print("   Activation: ReLU (Rectified Linear Unit)")
    print("   - Formula: f(x) = max(0, x)")
    print("   - Purpose: Introduces non-linearity, enables learning complex patterns")
    print("   ")
    print("   Pooling: MaxPool2d(kernel_size=2, stride=2)")
    print("   - Operation: Takes maximum value from each 2x2 region")
    print("   - Downsampling: (Batch, 32, 32, 32) → (Batch, 32, 16, 16)")
    print("   - Purpose: Reduces spatial dimensions, extracts dominant features, reduces computation")
    
    print("\n" + "-"*100)
    print("3️⃣  CONVOLUTIONAL BLOCK 2")
    print("-"*100)
    print("   Layer: Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)")
    print("   - Input: (Batch, 32, 16, 16)")
    print("   - Kernel: 3x3 filters (64 filters)")
    print("   - Padding: 1 (preserves spatial dimensions)")
    print("   - Number of Parameters: 3 * 3 * 32 * 64 + 64 = 18,496")
    print("   - Output Shape: (Batch, 64, 16, 16)")
    print("   ")
    print("   Activation: ReLU")
    print("   ")
    print("   Pooling: MaxPool2d(kernel_size=2, stride=2)")
    print("   - Downsampling: (Batch, 64, 16, 16) → (Batch, 64, 8, 8)")
    
    print("\n" + "-"*100)
    print("4️⃣  CONVOLUTIONAL BLOCK 3")
    print("-"*100)
    print("   Layer: Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)")
    print("   - Input: (Batch, 64, 8, 8)")
    print("   - Kernel: 3x3 filters (128 filters)")
    print("   - Padding: 1 (preserves spatial dimensions)")
    print("   - Number of Parameters: 3 * 3 * 64 * 128 + 128 = 73,856")
    print("   - Output Shape: (Batch, 128, 8, 8)")
    print("   ")
    print("   Activation: ReLU")
    print("   ")
    print("   Pooling: MaxPool2d(kernel_size=2, stride=2)")
    print("   - Downsampling: (Batch, 128, 8, 8) → (Batch, 128, 4, 4)")
    
    print("\n" + "-"*100)
    print("5️⃣  FLATTENING LAYER")
    print("-"*100)
    print("   Operation: Flatten the 4D tensor to 2D")
    print("   - Input Shape: (Batch, 128, 4, 4)")
    print("   - Flattened: (Batch, 128 * 4 * 4) = (Batch, 2048)")
    print("   - Purpose: Converts spatial features to a flat vector for fully connected layers")
    
    print("\n" + "-"*100)
    print("6️⃣  FULLY CONNECTED LAYER 1 (FC1)")
    print("-"*100)
    print("   Layer: Linear(in_features=2048, out_features=512)")
    print("   - Input: (Batch, 2048)")
    print("   - Output: (Batch, 512)")
    print("   - Number of Parameters: 2048 * 512 + 512 = 1,049,088")
    print("   ")
    print("   Activation: ReLU")
    print("   - Purpose: Non-linear transformation to learn class representations")
    
    print("\n" + "-"*100)
    print("7️⃣  DROPOUT LAYER")
    print("-"*100)
    print("   Layer: Dropout(p=0.5)")
    print("   - Dropout Rate: 50%")
    print("   - Behavior: Randomly sets 50% of neurons to 0 during training")
    print("   - Purpose: Prevents overfitting by reducing co-adaptation of neurons")
    print("   - Note: Disabled during evaluation/testing")
    
    print("\n" + "-"*100)
    print("8️⃣  OUTPUT LAYER (FC2)")
    print("-"*100)
    print("   Layer: Linear(in_features=512, out_features=100)")
    print("   - Input: (Batch, 512)")
    print("   - Output: (Batch, 100)")
    print("   - Number of Parameters: 512 * 100 + 100 = 51,300")
    print("   - Purpose: Produces logits for 100 CIFAR-100 classes")
    
    print("\n" + "-"*100)
    print("📊 ARCHITECTURE SUMMARY TABLE")
    print("-"*100)
    print(f"{'Layer':<20} {'Type':<20} {'Input Shape':<20} {'Output Shape':<20} {'Parameters':<15}")
    print("-"*100)
    print(f"{'Input':<20} {'Input':<20} {'(N, 3, 32, 32)':<20} {'(N, 3, 32, 32)':<20} {'0':<15}")
    print(f"{'Conv1':<20} {'Conv2d':<20} {'(N, 3, 32, 32)':<20} {'(N, 32, 32, 32)':<20} {'896':<15}")
    print(f"{'ReLU1':<20} {'Activation':<20} {'(N, 32, 32, 32)':<20} {'(N, 32, 32, 32)':<20} {'0':<15}")
    print(f"{'MaxPool1':<20} {'Pooling':<20} {'(N, 32, 32, 32)':<20} {'(N, 32, 16, 16)':<20} {'0':<15}")
    print(f"{'Conv2':<20} {'Conv2d':<20} {'(N, 32, 16, 16)':<20} {'(N, 64, 16, 16)':<20} {'18,496':<15}")
    print(f"{'ReLU2':<20} {'Activation':<20} {'(N, 64, 16, 16)':<20} {'(N, 64, 16, 16)':<20} {'0':<15}")
    print(f"{'MaxPool2':<20} {'Pooling':<20} {'(N, 64, 16, 16)':<20} {'(N, 64, 8, 8)':<20} {'0':<15}")
    print(f"{'Conv3':<20} {'Conv2d':<20} {'(N, 64, 8, 8)':<20} {'(N, 128, 8, 8)':<20} {'73,856':<15}")
    print(f"{'ReLU3':<20} {'Activation':<20} {'(N, 128, 8, 8)':<20} {'(N, 128, 8, 8)':<20} {'0':<15}")
    print(f"{'MaxPool3':<20} {'Pooling':<20} {'(N, 128, 8, 8)':<20} {'(N, 128, 4, 4)':<20} {'0':<15}")
    print(f"{'Flatten':<20} {'Reshape':<20} {'(N, 128, 4, 4)':<20} {'(N, 2048)':<20} {'0':<15}")
    print(f"{'FC1':<20} {'Linear':<20} {'(N, 2048)':<20} {'(N, 512)':<20} {'1,049,088':<15}")
    print(f"{'ReLU4':<20} {'Activation':<20} {'(N, 512)':<20} {'(N, 512)':<20} {'0':<15}")
    print(f"{'Dropout':<20} {'Regularization':<20} {'(N, 512)':<20} {'(N, 512)':<20} {'0':<15}")
    print(f"{'FC2 (Output)':<20} {'Linear':<20} {'(N, 512)':<20} {'(N, 100)':<20} {'51,300':<15}")
    print("-"*100)
    print(f"{'TOTAL PARAMETERS':<20} {'':<20} {'':<20} {'':<20} {'1,193,636':<15}")
    print("-"*100)
    
    print("\n🔑 KEY ARCHITECTURAL FEATURES:")
    print("   ✓ Progressive Feature Learning: Filter count increases (3 → 32 → 64 → 128)")
    print("   ✓ Spatial Dimensionality Reduction: 32×32 → 16×16 → 8×8 → 4×4")
    print("   ✓ Kernel Size: 3×3 kernels throughout (computationally efficient)")
    print("   ✓ Non-linearity: ReLU activations enable learning complex patterns")
    print("   ✓ Regularization: Dropout prevents overfitting")
    print("   ✓ Feature Extraction: 3 convolutional blocks extract hierarchical features")
    print("   ✓ Classification: 2 fully connected layers for class prediction")
    
    print("\n🎯 DESIGN RATIONALE:")
    print("   • 3×3 convolutions: Standard size, computationally efficient, captures local patterns")
    print("   • Progressive depth: Learns from simple (edges) to complex (objects) features")
    print("   • Max pooling: Reduces spatial dimensions while preserving important features")
    print("   • 512 hidden units: Balances model capacity with computational efficiency")
    print("   • 50% dropout: Prevents overfitting on 50,000 training samples")
    print("   • Total params: ~1.2M - suitable for CIFAR-100 classification")
    
    print("\n" + "="*100 + "\n")



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
    
    # Print CNN architecture description
    print_cnn_architecture()
    
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
    
   