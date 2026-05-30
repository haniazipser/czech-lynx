import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from models.base import BaseModel


class GradCAM:

    def __init__(self, model: BaseModel, device: torch.device):
        self.model = model
        self.device = device
        self._gradients = None
        self._activations = None

        model.target_layer.register_forward_hook(self._save_activations)
        model.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def compute(self, img_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.eval()
        img_tensor = img_tensor.to(self.device)

        logits = self.model(img_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def overlay(self, original_img: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
        cam_resized = Image.fromarray(np.uint8(255 * cam)).resize(
            original_img.size, resample=Image.BILINEAR
        )
        heatmap_rgb = np.uint8(cm.jet(np.array(cam_resized) / 255.0)[:, :, :3] * 255)
        overlay = np.uint8(alpha * heatmap_rgb + (1 - alpha) * np.array(original_img))
        return Image.fromarray(overlay)


def run_gradcam_analysis(
    model: BaseModel,
    dataset,
    device: torch.device,
    val_transform,
    n_correct: int = 2,
    n_incorrect: int = 2,
    save_path: str = "gradcam_analysis.png",
):
    gradcam = GradCAM(model, device)
    model.eval()
    inv_map = {v: k for k, v in dataset.label_map.items()}

    correct_indices, incorrect_indices = [], []

    for idx in range(len(dataset)):
        if len(correct_indices) >= n_correct and len(incorrect_indices) >= n_incorrect:
            break
        img_pil = Image.open(dataset.data_root / dataset.df.iloc[idx]["path"]).convert("RGB")
        true_label = dataset.df.iloc[idx]["identity"]
        img_tensor = val_transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_idx = model(img_tensor).argmax(1).item()
        pred_label = inv_map.get(pred_idx, "?")

        if pred_label == true_label and len(correct_indices) < n_correct:
            correct_indices.append(idx)
        elif pred_label != true_label and len(incorrect_indices) < n_incorrect:
            incorrect_indices.append(idx)


    indices = correct_indices + incorrect_indices
    num_samples = len(indices)

    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))
    fig.suptitle("GradCAM Analysis — Correct (top) vs Incorrect (bottom)", fontsize=12)

    for row, idx in enumerate(indices):
        img_pil = Image.open(dataset.data_root / dataset.df.iloc[idx]["path"]).convert("RGB")
        true_label = dataset.df.iloc[idx]["identity"]
        img_tensor = val_transform(img_pil).unsqueeze(0)

        cam = gradcam.compute(img_tensor)
        pred_label = inv_map.get(model(img_tensor.to(device)).argmax(1).item(), "?")
        overlay_img = gradcam.overlay(img_pil.resize((224, 224)), cam)
        color = "green" if pred_label == true_label else "red"

        axes[row, 0].imshow(img_pil.resize((224, 224)))
        axes[row, 0].set_title(f"GT: {true_label}", fontsize=7)
        axes[row, 1].imshow(cam, cmap="jet")
        axes[row, 1].set_title("GradCAM", fontsize=7)
        axes[row, 2].imshow(overlay_img)
        axes[row, 2].set_title(f"Pred: {pred_label}", fontsize=7, color=color)

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")