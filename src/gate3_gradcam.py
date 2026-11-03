from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class HAM10000Dataset(Dataset):
    def __init__(self, csv_path: Path, transform: transforms.Compose) -> None:
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str, str, str]:
        row = self.df.iloc[index]
        image_path = str(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, int(row["label"]), str(row["image_id"]), str(row["dx"]), image_path


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module: nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradients(
        self,
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor],
        grad_output: tuple[torch.Tensor],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, images: torch.Tensor, class_indexes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)
        selected_logits = logits.gather(1, class_indexes.view(-1, 1)).sum()
        selected_logits.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cams = (weights * self.activations).sum(dim=1)
        cams = F.relu(cams)
        cams = F.interpolate(cams.unsqueeze(1), size=images.shape[-2:], mode="bilinear", align_corners=False)
        cams = cams.squeeze(1)

        flat = cams.flatten(start_dim=1)
        min_values = flat.min(dim=1).values.view(-1, 1, 1)
        max_values = flat.max(dim=1).values.view(-1, 1, 1)
        cams = (cams - min_values) / (max_values - min_values + 1e-8)
        return logits.detach(), cams.detach()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate 3: Grad-CAM explanation consistency check.")
    parser.add_argument("--data-dir", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate3_gradcam"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cam-percentile", type=float, default=80.0)
    parser.add_argument("--iou-threshold", type=float, default=0.10)
    parser.add_argument("--centroid-threshold", type=float, default=0.35)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--save-examples", type=int, default=24)
    return parser.parse_args()


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ]
    )


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def target_layer_for(model: nn.Module, model_name: str) -> nn.Module:
    if model_name == "efficientnet_b0":
        return model.features[-1]
    if model_name == "resnet50":
        return model.layer4[-1]
    if model_name == "densenet121":
        return model.features[-1]
    raise ValueError(f"Unsupported model: {model_name}")


def load_rgb(image_path: str, image_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    return np.asarray(image).astype(np.float32) / 255.0


def pseudo_lesion_mask(image_path: str, image_size: int) -> np.ndarray:
    rgb = load_rgb(image_path, image_size)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 1e-6)

    y_grid, x_grid = np.mgrid[0:image_size, 0:image_size]
    center = (image_size - 1) / 2.0
    radius = np.sqrt((x_grid - center) ** 2 + (y_grid - center) ** 2)
    center_prior = 1.0 - np.clip(radius / (np.sqrt(2.0) * center), 0.0, 1.0)

    score = (0.70 * (1.0 - gray)) + (0.20 * saturation) + (0.10 * center_prior)
    threshold = np.percentile(score, 72.0)
    mask = score >= threshold

    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((7, 7)))
    mask = ndimage.binary_fill_holes(mask)

    labels, count = ndimage.label(mask)
    if count > 0:
        component_scores = []
        for component_idx in range(1, count + 1):
            component = labels == component_idx
            area = float(component.sum())
            points = np.argwhere(component)
            centroid_y, centroid_x = points.mean(axis=0)
            centroid_radius = np.sqrt((centroid_x - center) ** 2 + (centroid_y - center) ** 2)
            centrality = 1.0 - np.clip(centroid_radius / (np.sqrt(2.0) * center), 0.0, 1.0)
            touches_border = (
                component[0, :].any()
                or component[-1, :].any()
                or component[:, 0].any()
                or component[:, -1].any()
            )
            border_penalty = 0.10 if touches_border else 1.0
            component_scores.append(area * (0.25 + centrality) * border_penalty)

        selected = int(np.argmax(component_scores)) + 1
        mask = labels == selected

    min_pixels = int(image_size * image_size * 0.01)
    if int(mask.sum()) < min_pixels:
        threshold = np.percentile(score, 85.0)
        mask = score >= threshold

    return mask.astype(bool)


def real_lesion_mask(mask_dir: Path, image_id: str, image_size: int) -> np.ndarray | None:
    candidates = [
        mask_dir / f"{image_id}_segmentation.png",
        mask_dir / f"{image_id}.png",
        mask_dir / f"{image_id}_segmentation.jpg",
        mask_dir / f"{image_id}.jpg",
    ]

    for mask_path in candidates:
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L").resize((image_size, image_size))
            mask_np = np.asarray(mask)
            return mask_np > 127

    return None


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    points = np.argwhere(mask)
    if len(points) == 0:
        return None
    y, x = points.mean(axis=0)
    return float(x), float(y)


def centroid_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    centroid_a = mask_centroid(mask_a)
    centroid_b = mask_centroid(mask_b)
    if centroid_a is None or centroid_b is None:
        return 1.0

    dx = centroid_a[0] - centroid_b[0]
    dy = centroid_a[1] - centroid_b[1]
    diagonal = np.sqrt(mask_a.shape[0] ** 2 + mask_a.shape[1] ** 2)
    return float(np.sqrt(dx * dx + dy * dy) / diagonal)


def iou_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def cam_to_mask(cam: np.ndarray, percentile: float) -> np.ndarray:
    threshold = np.percentile(cam, percentile)
    mask = cam >= threshold
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    return mask.astype(bool)


def save_overlay(
    image_path: str,
    cam: np.ndarray,
    lesion_mask: np.ndarray,
    cam_mask: np.ndarray,
    output_path: Path,
    image_size: int,
    title: str,
) -> None:
    rgb = load_rgb(image_path, image_size)

    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(rgb)
    plt.axis("off")
    plt.title("Input")

    plt.subplot(1, 3, 2)
    plt.imshow(rgb)
    plt.imshow(cam, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.title("Grad-CAM")

    plt.subplot(1, 3, 3)
    plt.imshow(rgb)
    plt.contour(lesion_mask, levels=[0.5], colors="lime", linewidths=1.5)
    plt.contour(cam_mask, levels=[0.5], colors="red", linewidths=1.5)
    plt.axis("off")
    plt.title("Mask check")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    baseline_dir = (data_dir / args.baseline_dir).resolve()
    output_dir = (data_dir / args.output_dir).resolve()
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.mask_dir.resolve() if args.mask_dir is not None else None

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = baseline_dir / "models" / "best_baseline.pt"
    checkpoint_path = checkpoint_path.resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint["model_name"]
    image_size = int(checkpoint.get("image_size", args.image_size))

    model = build_model(model_name, len(LABELS))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    target_layer = target_layer_for(model, model_name)
    gradcam = GradCAM(model, target_layer)

    transform = build_eval_transform(image_size)
    split_path = baseline_dir / "splits" / f"{args.split}.csv"
    dataset = HAM10000Dataset(split_path, transform)

    if args.max_images is not None:
        dataset.df = dataset.df.head(args.max_images).copy()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Gate 3 Grad-CAM consistency using {model_name} on {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {args.split}, images: {len(dataset)}")
    if mask_dir is None:
        print("Pseudo-mask mode: using image-processing lesion masks because official masks are not present.")
    else:
        print(f"Real-mask mode: using lesion masks from {mask_dir}")

    rows = []
    saved_examples = 0

    for images, labels, image_ids, true_labels, image_paths in tqdm(loader, leave=False):
        images = images.to(device)
        logits_for_pred = model(images)
        probabilities = torch.softmax(logits_for_pred, dim=1)
        pred_indexes = torch.argmax(probabilities, dim=1)
        confidences = torch.max(probabilities, dim=1).values

        _, cams = gradcam(images, pred_indexes)
        cams_np = cams.cpu().numpy()

        for i in range(len(labels)):
            cam = cams_np[i]
            image_path = image_paths[i]
            lesion_mask = None
            mask_source = "pseudo_image_processing"
            if mask_dir is not None:
                lesion_mask = real_lesion_mask(mask_dir, image_ids[i], image_size)
                if lesion_mask is not None:
                    mask_source = "official_segmentation_mask"

            if lesion_mask is None:
                lesion_mask = pseudo_lesion_mask(image_path, image_size)
            cam_mask = cam_to_mask(cam, args.cam_percentile)
            iou = iou_score(cam_mask, lesion_mask)
            distance = centroid_distance(cam_mask, lesion_mask)
            consistency_score = 0.5 * iou + 0.5 * (1.0 - distance)
            gate_pass = iou >= args.iou_threshold and distance <= args.centroid_threshold

            pred_idx = int(pred_indexes[i].detach().cpu())
            row = {
                "image_id": image_ids[i],
                "true_label": true_labels[i],
                "true_index": int(labels[i]),
                "pred_label": LABELS[pred_idx],
                "pred_index": pred_idx,
                "confidence": float(confidences[i].detach().cpu()),
                "correct": int(pred_idx == int(labels[i])),
                "cam_lesion_iou": iou,
                "centroid_distance": distance,
                "consistency_score": consistency_score,
                "gate3_status": "pass" if gate_pass else "refer",
                "mask_source": mask_source,
            }
            rows.append(row)

            if saved_examples < args.save_examples:
                title = (
                    f"{image_ids[i]} true={true_labels[i]} pred={LABELS[pred_idx]} "
                    f"IoU={iou:.3f} dist={distance:.3f} {row['gate3_status']}"
                )
                save_overlay(
                    image_path,
                    cam,
                    lesion_mask,
                    cam_mask,
                    examples_dir / f"{saved_examples + 1:03d}_{image_ids[i]}_{row['gate3_status']}.png",
                    image_size,
                    title,
                )
                saved_examples += 1

    gradcam.remove_hooks()

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / f"{args.split}_gradcam_scores.csv", index=False)

    passed = results[results["gate3_status"] == "pass"]
    referred = results[results["gate3_status"] == "refer"]
    summary = {
        "model": model_name,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "total_count": int(len(results)),
        "pass_count": int(len(passed)),
        "referred_count": int(len(referred)),
        "pass_rate": float(len(passed) / len(results)) if len(results) else 0.0,
        "referral_rate": float(len(referred) / len(results)) if len(results) else 0.0,
        "iou_threshold": float(args.iou_threshold),
        "centroid_threshold": float(args.centroid_threshold),
        "cam_percentile": float(args.cam_percentile),
        "avg_iou": float(results["cam_lesion_iou"].mean()),
        "median_iou": float(results["cam_lesion_iou"].median()),
        "avg_centroid_distance": float(results["centroid_distance"].mean()),
        "median_centroid_distance": float(results["centroid_distance"].median()),
        "pass_accuracy": float(passed["correct"].mean()) if len(passed) else 0.0,
        "referred_accuracy": float(referred["correct"].mean()) if len(referred) else 0.0,
        "mask_source": "official_segmentation_mask" if mask_dir is not None else "pseudo_image_processing",
        "mask_dir": str(mask_dir) if mask_dir is not None else None,
        "note": "Official lesion segmentation masks were used." if mask_dir is not None else "Official lesion segmentation masks are not present. Replace pseudo masks with ISIC masks for final thesis-grade Gate 3 evaluation.",
    }

    (output_dir / "gate3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Gate 3 complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
