import os
import torch
import torch.nn as nn
import torch.optim as optim
from cnn_model import SimpleCNN
from data_loader import get_dataloader

def train_model(epochs=5, lr=0.001, device='cpu', save_path='checkpoints/simplecnn_emnist.pt'):
    train_loader, test_loader = get_dataloader()
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'num_classes': 26,
        },
        save_path,
    )
    print(f"Model saved to: {save_path}")

    return model, test_loader