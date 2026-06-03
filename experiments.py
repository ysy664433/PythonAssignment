import os
import csv
import torch

from train import train_model
from evaluate import evaluate

# --------------------------
# 实验配置：多个数据增强/噪声组合
# --------------------------
EXPERIMENTS = [
    # run_name, augment, add_gaussian, add_salt_pepper, test_gaussian, test_salt_pepper
    ("baseline", False, False, False, False, False),
    ("augment_only", True, False, False, False, False),
    ("gaussian_train", False, True, False, False, False),
    ("saltpepper_train", False, False, True, False, False),
    ("all_train_aug", True, True, True, False, False),
    ("gaussian_robust", False, False, False, True, False),
    ("saltpepper_robust", False, False, False, False, True),
    ("all_train_all_test", True, True, True, True, True),
]

# --------------------------
# 主函数
# --------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    results = []

    for run_name, augment, add_gaussian, add_salt_pepper, test_gaussian, test_salt_pepper in EXPERIMENTS:

        print(f"\n===== Training experiment: {run_name} =====")

        # --------------------------
        # 训练：会返回每个模型的 checkpoint 路径
        # --------------------------
        trained_models, _ = train_model(
            epochs=30,
            lr=0.001,
            device=device,
            save_dir="checkpoints",
            augment=augment,
            add_gaussian=add_gaussian,
            add_salt_pepper=add_salt_pepper,
            test_gaussian=test_gaussian,
            test_salt_pepper=test_salt_pepper,
            run_name=run_name,
        )

        # --------------------------
        # 对每个模型进行评估
        # --------------------------
        for model_type, model_path in trained_models.items():

            print(f"\n--- Evaluating {model_type} ({run_name}) ---")

            accuracy = evaluate(
                model_path=model_path,
                device=device,
                result_csv="outputs/eval_results.csv"
            )

            results.append({
                "run_name": run_name,
                "model_type": model_type,
                "augment": augment,
                "add_gaussian": add_gaussian,
                "add_salt_pepper": add_salt_pepper,
                "test_gaussian": test_gaussian,
                "test_salt_pepper": test_salt_pepper,
                "accuracy": accuracy,
                "model_path": model_path,
            })

    # --------------------------
    # 汇总实验结果 CSV
    # --------------------------
    csv_path = os.path.join("outputs", "experiment_results_summary.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "model_type",
                "augment",
                "add_gaussian",
                "add_salt_pepper",
                "test_gaussian",
                "test_salt_pepper",
                "accuracy",
                "model_path",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"\n实验结果汇总已保存到：{csv_path}")


# --------------------------
# 执行
# --------------------------
if __name__ == "__main__":
    main()