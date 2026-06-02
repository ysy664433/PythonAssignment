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


def plot_noisy_samples(clean_dataset, gaussian_dataset, salt_dataset, save_path: str, n: int = 8):
    cols = 3
    fig, axes = plt.subplots(n, cols, figsize=(10, 2.5 * n))

    for row in range(n):
        clean_img, label = clean_dataset[row]
        gaussian_img, _ = gaussian_dataset[row]
        salt_img, _ = salt_dataset[row]

        clean_img = denormalize(clean_img)
        gaussian_img = denormalize(gaussian_img)
        salt_img = denormalize(salt_img)

        axes[row, 0].imshow(clean_img.squeeze(0), cmap='gray')
        axes[row, 0].set_title(f"Original ({label_to_char(label)})", fontsize=9)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(gaussian_img.squeeze(0), cmap='gray')
        axes[row, 1].set_title("Gaussian Noise", fontsize=9)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(salt_img.squeeze(0), cmap='gray')
        axes[row, 2].set_title("SaltPepper Noise", fontsize=9)
        axes[row, 2].axis('off')

    fig.suptitle('Original vs Gaussian Noise vs SaltPepper Noise', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def main(output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    from datetime import datetime

    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    clean_dataset = EMNISTDataset(
        train=True,
        augment=False,
        add_gaussian=False,
        add_salt_pepper=False
    )

    gaussian_dataset = EMNISTDataset(
        train=True,
        augment=False,
        add_gaussian=True,
        add_salt_pepper=False
    )

    salt_dataset = EMNISTDataset(
        train=True,
        augment=False,
        add_gaussian=False,
        add_salt_pepper=True
    )

    typical_samples = collect_typical_samples(clean_dataset)

    typical_path = os.path.join(output_dir, f'typical_samples_{time_str}.png')
    noisy_path = os.path.join(output_dir, f'noise_comparison_{time_str}.png')

    plot_typical_samples(typical_samples, typical_path)
    plot_noisy_samples(clean_dataset, gaussian_dataset, salt_dataset, noisy_path)

    print(f"Saved: {typical_path}")
    print(f"Saved: {noisy_path}")


if __name__ == '__main__':
    main()
