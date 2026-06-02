import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

LETTER_LABELS = [chr(ord('A') + i) for i in range(26)]

def _fix_emnist_orientation(image):
    image = torch.rot90(image, k=-1, dims=(1, 2))
    image = torch.flip(image, dims=(2,))
    return image

#加上两种噪声
class AddGaussianNoise:
    def __init__(self, sigma=0.15):
        self.sigma = sigma

    def __call__(self, x):
        noise = torch.randn_like(x) * self.sigma
        return (x + noise).clamp(0.0, 1.0)

class AddSaltPepperNoise:
    def __init__(self, prob=0.08):
        self.prob = prob

    def __call__(self, x):
        noisy = x.clone()
        random_map = torch.rand_like(noisy)
        noisy[random_map < self.prob / 2] = 0.0
        noisy[random_map > 1 - self.prob / 2] = 1.0
        return noisy

class EMNISTDataset(Dataset):
    def __init__(self, split='letters', train=True, augment=False,add_gaussian= False, add_salt_pepper=False):
        self.augment = augment
        self.add_gaussian = add_gaussian
        self.add_salt_pepper = add_salt_pepper
        transform_list = []
        if train and augment:
            transform_list.append(
                transforms.RandomAffine(
                    degrees=15,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=10,
                )
            )
            
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Lambda(_fix_emnist_orientation))

        if train and add_gaussian:
            transform_list.append(AddGaussianNoise(sigma=0.15))

        if train and add_salt_pepper:
            transform_list.append(AddSaltPepperNoise(prob=0.08))
       
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        
        self.data = datasets.EMNIST(
            root='./data',
            split=split,
            train=train,
            download=True,
            transform=transforms.Compose(transform_list)
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


def get_dataloader(
    batch_size=64,
    augment=False,
    add_gaussian=False,
    add_salt_pepper=False,
    val_ratio=0.15,
    test_gaussian=False,      
    test_salt_pepper=False    
):
    full_train_dataset = EMNISTDataset(
        train=True,
        augment=augment,
        add_gaussian=add_gaussian,
        add_salt_pepper=add_salt_pepper
    )

    # 验证集保持干净
    val_full = EMNISTDataset(train=True, augment=False, add_gaussian=False, add_salt_pepper=False)
    
    total_size = len(full_train_dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(42))
    train_dataset = torch.utils.data.Subset(full_train_dataset, indices[:train_size])
    val_dataset   = torch.utils.data.Subset(val_full, indices[train_size:])

    # 测试集可以灵活选择噪声类型
    test_dataset = EMNISTDataset(
        train=False,
        augment=False,
        add_gaussian=test_gaussian,
        add_salt_pepper=test_salt_pepper
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader