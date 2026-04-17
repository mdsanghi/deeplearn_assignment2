from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np


DATASET_DIR = Path(__file__).resolve().parent.parent / "cifar-100-python"


def load_cifar100(split="train", dataset_dir=DATASET_DIR):
    """Load one CIFAR-100 split from the extracted cifar-100-python folder."""
    split_path = dataset_dir / split
    meta_path = dataset_dir / "meta"

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


if __name__ == "__main__":
    show_sample_images(split="train", num_samples=12)
