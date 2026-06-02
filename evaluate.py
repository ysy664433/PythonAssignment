import argparse
import os
import torch
from train import train_model
from cnn_model import SimpleCNN
from data_loader import get_dataloader

def evaluate(model_path='checkpoints/simplecnn_emnist.pt', force_retrain=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if force_retrain or not os.path.exists(model_path):
        model, test_loader = train_model(device=device, save_path=model_path)
    else:
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model = SimpleCNN(num_classes=checkpoint.get('num_classes', 26)).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        _, test_loader = get_dataloader()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='checkpoints/simplecnn_emnist.pt')
    parser.add_argument('--retrain', action='store_true', help='Force retraining before evaluation')
    args = parser.parse_args()

    evaluate(model_path=args.model_path, force_retrain=args.retrain)