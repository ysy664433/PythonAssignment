import os
import torch
import torch.nn as nn
import torch.optim as optim
from cnn_model import SimpleCNN
from cnn_model import SimpleAlexNet
from cnn_model import SimpleResNet
from data_loader import get_dataloader
# 新增：绘图库
import matplotlib.pyplot as plt
from datetime import datetime

def train_model(
    epochs=20,
    lr=0.001,
    device='cpu',
    save_dir="checkpoints",
    augment=False,
    add_gaussian=False,
    add_salt_pepper=False,
    test_gaussian=False,
    test_salt_pepper=False,
    run_name=None
):
    # 加载数据集
    train_loader, val_loader, test_loader = get_dataloader(
        augment=augment,
        add_gaussian=add_gaussian,
        add_salt_pepper=add_salt_pepper,
        test_gaussian=test_gaussian,
        test_salt_pepper=test_salt_pepper
    )

    if run_name is None:
        run_name = f"aug{int(augment)}_gauss{int(add_gaussian)}_sp{int(add_salt_pepper)}_testG{int(test_gaussian)}_testSP{int(test_salt_pepper)}"

    os.makedirs(save_dir, exist_ok=True)

    # 模型列表
    list_of_models = ["cnn", "resnet", "alexnet"]

    trained_models = {}
    for model_type in list_of_models:
        # 初始化模型
        if model_type == "cnn":
            model = SimpleCNN(num_classes=26)
        elif model_type == "resnet":
            model = SimpleResNet(num_classes=26)
        elif model_type == "alexnet":
            model = SimpleAlexNet(num_classes=26)
        else:
            raise ValueError("model_type must be 'cnn', 'resnet', or 'alexnet'")
        
        save_path = os.path.join(save_dir, f"{run_name}_{model_type}.pt")
        if os.path.exists(save_path):
            print(f"[{model_type}] 已存在 checkpoint，跳过训练: {save_path}")
            trained_models[model_type] = save_path
            continue
        
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        loss_history = []
        val_loss_history = []

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
            loss_history.append(avg_loss)

            # 验证集 Loss
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
            avg_val_loss = val_loss / len(val_loader.dataset)
            val_loss_history.append(avg_val_loss)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"[{model_type}] Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

            scheduler.step()
        
        # 保存模型
        torch.save({
            'model_type': model_type,
            'num_classes': 26,
            'model_state_dict': model.state_dict(),
            'loss_history': loss_history,
            'val_loss_history': val_loss_history,
            'augment': augment,
            'add_gaussian': add_gaussian,
            'add_salt_pepper': add_salt_pepper,
            'test_gaussian': test_gaussian,
            'test_salt_pepper': test_salt_pepper,
            'epochs': epochs,
            'lr': lr,
            'run_name': run_name,
        }, save_path)

        trained_models[model_type] = save_path
        
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, label="Train Loss")
        plt.plot(val_loss_history, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{model_type} Training vs Validation Loss")
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(save_dir, f"{run_name}_{model_type}_loss_curve.png"))
        plt.close()
    return trained_models, test_loader
