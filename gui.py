import os
import tkinter as tk
from tkinter import messagebox, ttk

import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms

from cnn_model import SimpleCNN
from data_loader import label_to_char


CANVAS_SIZE = 280
MODEL_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "simplecnn_emnist.pt")


class LetterRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("字母识别")
        self.root.resizable(False, False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(MODEL_PATH)

        self._build_ui()
        self._reset_drawing()

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=16)
        outer.grid(row=0, column=0, sticky="nsew")

        title = ttk.Label(outer, text="使用训练好的模型实时识别", font=("Microsoft YaHei", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        tips = ttk.Label(
            outer,
            text="在黑底画布上用白色粗线书写单个大写字母。模型会输出最可能的字母和前三个候选。",
            wraplength=560,
            justify="left",
        )
        tips.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 12))

        self.canvas = tk.Canvas(outer, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", highlightthickness=1, highlightbackground="#666")
        self.canvas.grid(row=2, column=0, rowspan=4, padx=(0, 16))
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<ButtonPress-1>", self._paint)

        controls = ttk.Frame(outer)
        controls.grid(row=2, column=1, sticky="nw")

        ttk.Button(controls, text="识别", command=self.predict_letter).grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(controls, text="清空", command=self.clear_canvas).grid(row=1, column=0, sticky="ew", pady=(0, 8))

        result_box = ttk.LabelFrame(outer, text="识别结果", padding=12)
        result_box.grid(row=3, column=1, sticky="nsew", pady=(16, 0))

        self.result_label = ttk.Label(result_box, text="暂无结果", font=("Microsoft YaHei", 18, "bold"))
        self.result_label.grid(row=0, column=0, sticky="w")

        self.detail_label = ttk.Label(result_box, text="先在左侧画一个字母，再点击识别。", wraplength=260, justify="left")
        self.detail_label.grid(row=1, column=0, sticky="w", pady=(10, 0))

        self.status_label = ttk.Label(outer, text=f"模型: {MODEL_PATH}")
        self.status_label.grid(row=4, column=1, sticky="w", pady=(16, 0))

        note = ttk.Label(outer, text="如果识别不准，优先写大一点、居中一点、一次只写一个字母。", foreground="#666")
        note.grid(row=5, column=0, columnspan=2, sticky="w", pady=(12, 0))

    def _reset_drawing(self):
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        self.canvas.delete("all")

    def _paint(self, event):
        x, y = event.x, event.y
        radius = 12

        self.canvas.create_line(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill="white",
            width=18,
            capstyle=tk.ROUND,
            smooth=True,
        )
        self.draw.line(
            (self.last_x or x, self.last_y or y, x, y),
            fill=255,
            width=18,
        )
        self.last_x = x
        self.last_y = y

    def clear_canvas(self):
        self._reset_drawing()
        self.result_label.config(text="暂无结果")
        self.detail_label.config(text="画布已清空。")

    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            messagebox.showerror("模型不存在", f"找不到模型文件：\n{model_path}\n请先运行 evaluate.py --retrain 训练一次。")
            raise FileNotFoundError(model_path)

        checkpoint = torch.load(model_path, map_location=self.device)
        model = SimpleCNN(num_classes=checkpoint.get("num_classes", 26)).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def _preprocess_drawing(self):
        bbox = self.image.getbbox()
        if bbox is None:
            return None

        cropped = self.image.crop(bbox)

        width, height = cropped.size
        if width == 0 or height == 0:
            return None

        if width > height:
            new_width = 20
            new_height = max(1, round(height * 20 / width))
        else:
            new_height = 20
            new_width = max(1, round(width * 20 / height))

        resized = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (28, 28), 0)
        offset_x = (28 - new_width) // 2
        offset_y = (28 - new_height) // 2
        canvas.paste(resized, (offset_x, offset_y))

        tensor = transforms.ToTensor()(canvas)
        tensor = transforms.Normalize((0.1307,), (0.3081,))(tensor)
        return tensor.unsqueeze(0).to(self.device)

    def predict_letter(self):
        input_tensor = self._preprocess_drawing()
        if input_tensor is None:
            messagebox.showinfo("提示", "画布还是空的，请先写一个字母。")
            return

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0)
            top_probabilities, top_indices = torch.topk(probabilities, k=3)

        top_index = int(top_indices[0].item())
        top_letter = label_to_char(top_index)
        top_confidence = float(top_probabilities[0].item()) * 100

        candidates = []
        for prob, idx in zip(top_probabilities, top_indices):
            candidates.append(f"{label_to_char(int(idx.item()))}: {float(prob.item()) * 100:.1f}%")

        self.result_label.config(text=f"{top_letter}  ({top_confidence:.1f}%)")
        self.detail_label.config(text="前三候选：" + "，".join(candidates))


def main():
    root = tk.Tk()
    app = LetterRecognizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
