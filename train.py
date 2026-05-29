import os
import torch
import torch.nn as nn
import torch.optim as optim
from cnn_model import SimpleCNN
from data_loader import get_dataloader
# 新增：绘图库
import matplotlib.pyplot as plt

def train_model(epochs=30, lr=0.001, device='cpu', save_path='checkpoints/simplecnn_emnist.pt'):
    train_loader, test_loader = get_dataloader()
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #记录loss
    loss_history = []

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
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        # 保存每一轮 loss
        loss_history.append(avg_loss)
        scheduler.step()
       


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'num_classes': 26,
            'loss_history': loss_history,  # 把 loss 历史也存进模型
        },
        save_path,
    )
    print(f"Model saved to: {save_path}")
    
    # 画 loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Training Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve During Training")
    plt.legend()
    plt.grid(True)

    #保存每一次Loss
    from datetime import datetime
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs", exist_ok=True)
    loss_curve_path = os.path.join("outputs", f"loss_curve_{time_str}.png")

    plt.savefig(loss_curve_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Loss 曲线已保存为：{loss_curve_path}")

    return model, test_loader
