import os
import csv
import torch
from cnn_model import SimpleCNN, SimpleAlexNet, SimpleResNet
from data_loader import get_dataloader

def build_model(model_type, num_classes=26):
    """根据模型类型返回实例"""
    if model_type == "cnn":
        return SimpleCNN(num_classes=num_classes)
    elif model_type == "resnet":
        return SimpleResNet(num_classes=num_classes)
    elif model_type == "alexnet":
        return SimpleAlexNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def evaluate(model_path, device=None, result_csv="results/eval_results.csv"):
    """自动识别模型类型，加载 checkpoint 并评估准确率"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载 checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_type = checkpoint.get("model_type", "cnn")  # 默认 cnn
    num_classes = checkpoint.get("num_classes", 26)
    
    # 构建模型
    model = build_model(model_type, num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 加载测试集，保持训练时的噪声设置
    _, _, test_loader = get_dataloader(
        augment=False,
        add_gaussian=False,
        add_salt_pepper=False,
        test_gaussian=checkpoint.get("test_gaussian", False),
        test_salt_pepper=checkpoint.get("test_salt_pepper", False)
    )

    # 计算准确率
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total if total > 0 else 0.0

    # 保存到 CSV
    result_dir = os.path.dirname(result_csv)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
        file_exists = os.path.exists(result_csv)
    with open(result_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "model_path", "model_type", "run_name",
                "augment", "train_gaussian", "train_salt_pepper",
                "test_gaussian", "test_salt_pepper",
                "epochs", "lr", "accuracy"
            ])
        writer.writerow([
            model_path,
            model_type,
            checkpoint.get("run_name", ""),
            checkpoint.get("augment", False),
            checkpoint.get("add_gaussian", False),
            checkpoint.get("add_salt_pepper", False),
            checkpoint.get("test_gaussian", False),
            checkpoint.get("test_salt_pepper", False),
            checkpoint.get("epochs", ""),
            checkpoint.get("lr", ""),
            round(accuracy, 4)
        ])

    print(f"[{model_type}] Test Accuracy: {accuracy:.2f}%")
    print(f"结果已保存到: {result_csv}")
    return accuracy