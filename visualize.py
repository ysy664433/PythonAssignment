import os
import random

import matplotlib.pyplot as plt
import torch

from data_loader import EMNISTDataset, label_to_char


MEAN = 0.1307
STD = 0.3081


def denormalize(image_tensor: torch.Tensor) -> torch.Tensor:
    image = image_tensor * STD + MEAN
    return image.clamp(0.0, 1.0)


def add_gaussian_noise(image: torch.Tensor, sigma: float = 0.2) -> torch.Tensor:
    noise = torch.randn_like(image) * sigma
    return (image + noise).clamp(0.0, 1.0)


def add_salt_pepper_noise(image: torch.Tensor, prob: float = 0.08) -> torch.Tensor:
    noisy = image.clone()
    random_map = torch.rand_like(noisy)
    noisy[random_map < prob / 2] = 0.0
    noisy[random_map > 1 - prob / 2] = 1.0
    return noisy


def collect_typical_samples(dataset: EMNISTDataset):
    samples = {}
    for image, label in dataset:
        if label not in samples:
            samples[label] = image
        if len(samples) == 26:
            break
    return samples


def plot_typical_samples(samples: dict, save_path: str):
    fig, axes = plt.subplots(4, 7, figsize=(14, 8))
    axes = axes.flatten()

    for idx in range(26):
        image = denormalize(samples[idx]).squeeze(0)
        axes[idx].imshow(image, cmap='gray')
        axes[idx].set_title(f"{idx} -> {label_to_char(idx)}", fontsize=10)
        axes[idx].axis('off')

    for idx in range(26, len(axes)):
        axes[idx].axis('off')

    fig.suptitle('EMNIST Typical Samples (0-25 mapped to A-Z)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_noisy_samples(dataset: EMNISTDataset, save_path: str, n: int = 8):
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.4 * n))

    for row in range(n):
        image, label = dataset[row]
        base = denormalize(image)
        gaussian = add_gaussian_noise(base)
        salt_pepper = add_salt_pepper_noise(base)

        axes[row, 0].imshow(base.squeeze(0), cmap='gray')
        axes[row, 0].set_title(f"Original ({label_to_char(label)})", fontsize=9)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(gaussian.squeeze(0), cmap='gray')
        axes[row, 1].set_title('Gaussian Noise', fontsize=9)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(salt_pepper.squeeze(0), cmap='gray')
        axes[row, 2].set_title('Salt & Pepper Noise', fontsize=9)
        axes[row, 2].axis('off')

    fig.suptitle('Noisy Sample Visualization', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main(output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)

    train_dataset = EMNISTDataset(train=True)

    typical_samples = collect_typical_samples(train_dataset)
    typical_path = os.path.join(output_dir, 'typical_samples.png')
    noisy_path = os.path.join(output_dir, 'noisy_samples.png')

    plot_typical_samples(typical_samples, typical_path)
    plot_noisy_samples(train_dataset, noisy_path)

    print(f"Saved: {typical_path}")
    print(f"Saved: {noisy_path}")


if __name__ == '__main__':
    main()
