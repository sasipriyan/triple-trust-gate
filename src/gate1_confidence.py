from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class HAM10000Dataset(Dataset):
    def __init__(self, csv_path: Path, transform: transforms.Compose) -> None:
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str, str]:
        row = self.df.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        return image, int(row["label"]), str(row["image_id"]), str(row["dx"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate 1: confidence and uncertainty evaluation.")
    parser.add_argument("--data-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate1_confidence"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--target-accepted-accuracy", type=float, default=0.90)
    parser.add_argument("--mc-samples", type=int, default=1)
    return parser.parse_args()


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
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


def enable_mc_dropout(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -np.sum(probs * np.log(probs + eps), axis=1)


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, pd.DataFrame]:
    rows = []
    ece = 0.0
    total = len(y_true)

    for bin_idx in range(n_bins):
        lower = bin_idx / n_bins
        upper = (bin_idx + 1) / n_bins
        if bin_idx == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        count = int(mask.sum())
        if count == 0:
            accuracy = 0.0
            avg_confidence = 0.0
            gap = 0.0
        else:
            accuracy = float((y_true[mask] == y_pred[mask]).mean())
            avg_confidence = float(confidence[mask].mean())
            gap = abs(accuracy - avg_confidence)
            ece += (count / total) * gap

        rows.append(
            {
                "bin": bin_idx + 1,
                "lower": lower,
                "upper": upper,
                "count": count,
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "gap": gap,
            }
        )

    return float(ece), pd.DataFrame(rows)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mc_samples: int,
) -> pd.DataFrame:
    rows = []
    model.eval()

    for images, labels, image_ids, true_labels in tqdm(loader, leave=False):
        images = images.to(device)

        sample_probs = []
        with torch.no_grad():
            for _ in range(mc_samples):
                if mc_samples > 1:
                    enable_mc_dropout(model)
                else:
                    model.eval()

                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                sample_probs.append(probs.detach().cpu().numpy())

        probs_np = np.mean(sample_probs, axis=0)
        pred_idx = probs_np.argmax(axis=1)
        confidence = probs_np.max(axis=1)
        entropy = predictive_entropy(probs_np)

        if mc_samples > 1:
            stacked = np.stack(sample_probs, axis=0)
            uncertainty = stacked.max(axis=2).std(axis=0)
        else:
            uncertainty = np.zeros_like(confidence)

        for i in range(len(labels)):
            rows.append(
                {
                    "image_id": image_ids[i],
                    "true_label": true_labels[i],
                    "true_index": int(labels[i]),
                    "pred_label": LABELS[int(pred_idx[i])],
                    "pred_index": int(pred_idx[i]),
                    "confidence": float(confidence[i]),
                    "predictive_entropy": float(entropy[i]),
                    "mc_confidence_std": float(uncertainty[i]),
                    "correct": int(pred_idx[i] == int(labels[i])),
                }
            )

    return pd.DataFrame(rows)


def metrics_at_threshold(df: pd.DataFrame, threshold: float) -> dict[str, float]:
    accepted = df["confidence"] >= threshold
    accepted_df = df[accepted]

    if len(accepted_df) == 0:
        return {
            "threshold": threshold,
            "accepted_count": 0,
            "referred_count": int(len(df)),
            "coverage": 0.0,
            "referral_rate": 1.0,
            "accepted_accuracy": 0.0,
            "accepted_precision_macro": 0.0,
            "accepted_recall_macro": 0.0,
            "accepted_f1_macro": 0.0,
        }

    y_true = accepted_df["true_index"].to_numpy()
    y_pred = accepted_df["pred_index"].to_numpy()

    return {
        "threshold": threshold,
        "accepted_count": int(len(accepted_df)),
        "referred_count": int(len(df) - len(accepted_df)),
        "coverage": float(len(accepted_df) / len(df)),
        "referral_rate": float(1.0 - len(accepted_df) / len(df)),
        "accepted_accuracy": float(accuracy_score(y_true, y_pred)),
        "accepted_precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "accepted_recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "accepted_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def choose_threshold(df: pd.DataFrame, target_accepted_accuracy: float) -> tuple[float, pd.DataFrame]:
    thresholds = np.round(np.arange(0.0, 1.001, 0.01), 2)
    rows = [metrics_at_threshold(df, float(threshold)) for threshold in thresholds]
    table = pd.DataFrame(rows)

    candidates = table[table["accepted_accuracy"] >= target_accepted_accuracy]
    candidates = candidates[candidates["accepted_count"] > 0]

    if len(candidates) > 0:
        best = candidates.sort_values(["coverage", "accepted_accuracy"], ascending=[False, False]).iloc[0]
    else:
        best = table.sort_values(["accepted_accuracy", "coverage"], ascending=[False, False]).iloc[0]

    return float(best["threshold"]), table


def save_reliability_plot(calibration_df: pd.DataFrame, ece: float, output_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.bar(
        calibration_df["avg_confidence"],
        calibration_df["accuracy"],
        width=0.08,
        alpha=0.7,
        edgecolor="black",
        label="Observed accuracy",
    )
    plt.xlabel("Average confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Gate 1 Reliability Diagram, ECE={ece:.4f}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    baseline_dir = (data_dir / args.baseline_dir).resolve()
    output_dir = (data_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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

    transform = build_eval_transform(image_size)
    val_ds = HAM10000Dataset(baseline_dir / "splits" / "val.csv", transform)
    test_ds = HAM10000Dataset(baseline_dir / "splits" / "test.csv", transform)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Gate 1 evaluation using {model_name} on {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"MC samples: {args.mc_samples}")

    val_predictions = collect_predictions(model, val_loader, device, args.mc_samples)
    if args.threshold is None:
        threshold, threshold_table = choose_threshold(val_predictions, args.target_accepted_accuracy)
    else:
        threshold = args.threshold
        threshold_table = pd.DataFrame([metrics_at_threshold(val_predictions, threshold)])

    test_predictions = collect_predictions(model, test_loader, device, args.mc_samples)
    test_predictions["gate1_status"] = np.where(test_predictions["confidence"] >= threshold, "pass", "refer")

    val_predictions.to_csv(output_dir / "val_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    threshold_table.to_csv(output_dir / "threshold_search.csv", index=False)

    y_true = test_predictions["true_index"].to_numpy()
    y_pred = test_predictions["pred_index"].to_numpy()
    confidence = test_predictions["confidence"].to_numpy()
    ece, calibration_df = expected_calibration_error(y_true, y_pred, confidence)
    calibration_df.to_csv(output_dir / "calibration_bins.csv", index=False)
    save_reliability_plot(calibration_df, ece, output_dir / "reliability_diagram.png")

    accepted_df = test_predictions[test_predictions["gate1_status"] == "pass"]
    referred_df = test_predictions[test_predictions["gate1_status"] == "refer"]
    summary = metrics_at_threshold(test_predictions, threshold)
    summary.update(
        {
            "model": model_name,
            "device": str(device),
            "checkpoint": str(checkpoint_path),
            "mc_samples": int(args.mc_samples),
            "target_accepted_accuracy": float(args.target_accepted_accuracy),
            "test_ece": ece,
            "overall_test_accuracy_before_gate": float(accuracy_score(y_true, y_pred)),
            "overall_test_f1_macro_before_gate": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "accepted_correct": int(accepted_df["correct"].sum()),
            "accepted_incorrect": int(len(accepted_df) - accepted_df["correct"].sum()),
            "referred_correct": int(referred_df["correct"].sum()),
            "referred_incorrect": int(len(referred_df) - referred_df["correct"].sum()),
        }
    )

    (output_dir / "gate1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Gate 1 complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
