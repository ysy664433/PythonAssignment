import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

LETTER_LABELS = [chr(ord('A') + i) for i in range(26)]


def _fix_emnist_orientation(image):
    image = torch.rot90(image, k=-1, dims=(1, 2))
    image = torch.flip(image, dims=(2,))
    return image

class EMNISTDataset(Dataset):
    def __init__(self, split='letters', train=True):
        self.data = datasets.EMNIST(
            root='./data',
            split=split,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(_fix_emnist_orientation),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        label = label - 1
        return image, label


def label_to_char(label: int) -> str:
    if not 0 <= label < len(LETTER_LABELS):
        raise ValueError(f"Label out of range: {label}")
    return LETTER_LABELS[label]

def get_dataloader(batch_size=64):
    train_dataset = EMNISTDataset(train=True)
    test_dataset = EMNISTDataset(train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader