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


class FeatureExtractor(nn.Module):
    def __init__(self, model_name: str, checkpoint: dict) -> None:
        super().__init__()
        self.model_name = model_name
        self.model = build_model(model_name, len(LABELS))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model_name == "efficientnet_b0":
            features = self.model.features(images)
            features = self.model.avgpool(features)
            embeddings = torch.flatten(features, 1)
            logits = self.model.classifier(embeddings)
            return embeddings, logits

        if self.model_name == "resnet50":
            x = self.model.conv1(images)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            embeddings = torch.flatten(x, 1)
            logits = self.model.fc(embeddings)
            return embeddings, logits

        if self.model_name == "densenet121":
            features = self.model.features(images)
            features = F.relu(features, inplace=False)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            embeddings = torch.flatten(features, 1)
            logits = self.model.classifier(embeddings)
            return embeddings, logits

        raise ValueError(f"Unsupported model: {self.model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate 2: feature-space known/OOD detection.")
    parser.add_argument("--data-dir", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate2_ood"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-known-pass-rate", type=float, default=0.95)
    parser.add_argument("--threshold", type=float, default=None)
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


def extract_embeddings(
    extractor: FeatureExtractor,
    loader: DataLoader,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    all_embeddings = []
    extractor.eval()

    with torch.no_grad():
        for images, labels, image_ids, true_labels in tqdm(loader, leave=False):
            images = images.to(device)
            embeddings, logits = extractor(images)
            probabilities = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1).values

            embeddings_np = embeddings.detach().cpu().numpy()
            all_embeddings.append(embeddings_np)

            for i in range(len(labels)):
                rows.append(
                    {
                        "image_id": image_ids[i],
                        "true_label": true_labels[i],
                        "true_index": int(labels[i]),
                        "pred_label": LABELS[int(pred_idx[i])],
                        "pred_index": int(pred_idx[i]),
                        "confidence": float(confidence[i]),
                        "correct": int(pred_idx[i] == int(labels[i])),
                    }
                )

    return pd.DataFrame(rows), np.vstack(all_embeddings)


def l2_normalize(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return values / norm


def build_centroids(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    normalized = l2_normalize(embeddings)
    centroids = []

    for class_idx in range(len(LABELS)):
        class_embeddings = normalized[labels == class_idx]
        if len(class_embeddings) == 0:
            raise ValueError(f"No embeddings found for class {LABELS[class_idx]}")
        centroid = class_embeddings.mean(axis=0, keepdims=True)
        centroid = l2_normalize(centroid)[0]
        centroids.append(centroid)

    return np.vstack(centroids)


def add_distance_scores(df: pd.DataFrame, embeddings: np.ndarray, centroids: np.ndarray) -> pd.DataFrame:
    normalized = l2_normalize(embeddings)
    cosine_similarity = normalized @ centroids.T
    cosine_distance = 1.0 - cosine_similarity

    nearest_idx = cosine_distance.argmin(axis=1)
    min_distance = cosine_distance.min(axis=1)
    predicted_distance = cosine_distance[np.arange(len(df)), df["pred_index"].to_numpy()]
    true_class_distance = cosine_distance[np.arange(len(df)), df["true_index"].to_numpy()]

    scored = df.copy()
    scored["nearest_centroid_label"] = [LABELS[int(idx)] for idx in nearest_idx]
    scored["nearest_centroid_index"] = nearest_idx.astype(int)
    scored["min_centroid_distance"] = min_distance.astype(float)
    scored["predicted_class_distance"] = predicted_distance.astype(float)
    scored["true_class_distance"] = true_class_distance.astype(float)
    scored["nearest_matches_prediction"] = (scored["nearest_centroid_index"] == scored["pred_index"]).astype(int)
    return scored


def choose_threshold(distances: np.ndarray, target_known_pass_rate: float) -> float:
    target_known_pass_rate = min(max(target_known_pass_rate, 0.0), 1.0)
    return float(np.quantile(distances, target_known_pass_rate))


def summarize_gate(scored: pd.DataFrame, threshold: float, split_name: str) -> dict[str, float | int | str]:
    known_mask = scored["min_centroid_distance"] <= threshold
    known_df = scored[known_mask]
    referred_df = scored[~known_mask]

    summary: dict[str, float | int | str] = {
        "split": split_name,
        "threshold": float(threshold),
        "total_count": int(len(scored)),
        "known_count": int(len(known_df)),
        "referred_count": int(len(referred_df)),
        "known_pass_rate": float(len(known_df) / len(scored)),
        "referral_rate": float(len(referred_df) / len(scored)),
        "avg_min_centroid_distance": float(scored["min_centroid_distance"].mean()),
        "median_min_centroid_distance": float(scored["min_centroid_distance"].median()),
        "p95_min_centroid_distance": float(scored["min_centroid_distance"].quantile(0.95)),
    }

    if len(known_df) > 0:
        summary["known_accuracy"] = float(known_df["correct"].mean())
    else:
        summary["known_accuracy"] = 0.0

    if len(referred_df) > 0:
        summary["referred_accuracy"] = float(referred_df["correct"].mean())
    else:
        summary["referred_accuracy"] = 0.0

    return summary


def save_distance_plot(val_scored: pd.DataFrame, test_scored: pd.DataFrame, threshold: float, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(val_scored["min_centroid_distance"], bins=40, alpha=0.55, label="Validation known", density=True)
    plt.hist(test_scored["min_centroid_distance"], bins=40, alpha=0.55, label="Test known", density=True)
    plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold={threshold:.4f}")
    plt.xlabel("Minimum cosine distance to class centroid")
    plt.ylabel("Density")
    plt.title("Gate 2 Feature-Space Distance Distribution")
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

    transform = build_eval_transform(image_size)
    train_ds = HAM10000Dataset(baseline_dir / "splits" / "train.csv", transform)
    val_ds = HAM10000Dataset(baseline_dir / "splits" / "val.csv", transform)
    test_ds = HAM10000Dataset(baseline_dir / "splits" / "test.csv", transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    extractor = FeatureExtractor(model_name, checkpoint).to(device)

    print(f"Gate 2 feature-space OOD detection using {model_name} on {device}")
    print(f"Checkpoint: {checkpoint_path}")

    train_df, train_embeddings = extract_embeddings(extractor, train_loader, device)
    centroids = build_centroids(train_embeddings, train_df["true_index"].to_numpy())

    val_df, val_embeddings = extract_embeddings(extractor, val_loader, device)
    test_df, test_embeddings = extract_embeddings(extractor, test_loader, device)

    train_scored = add_distance_scores(train_df, train_embeddings, centroids)
    val_scored = add_distance_scores(val_df, val_embeddings, centroids)
    test_scored = add_distance_scores(test_df, test_embeddings, centroids)

    if args.threshold is None:
        threshold = choose_threshold(val_scored["min_centroid_distance"].to_numpy(), args.target_known_pass_rate)
    else:
        threshold = float(args.threshold)

    for scored in (train_scored, val_scored, test_scored):
        scored["gate2_status"] = np.where(scored["min_centroid_distance"] <= threshold, "known", "refer")

    train_scored.to_csv(output_dir / "train_ood_scores.csv", index=False)
    val_scored.to_csv(output_dir / "val_ood_scores.csv", index=False)
    test_scored.to_csv(output_dir / "test_ood_scores.csv", index=False)
    np.savez(output_dir / "class_centroids.npz", centroids=centroids, labels=np.array(LABELS))

    summary = {
        "model": model_name,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "score": "minimum cosine distance to class centroid",
        "target_known_pass_rate": float(args.target_known_pass_rate),
        "threshold": float(threshold),
        "note": "This run uses HAM10000 known-class data only. External OOD data is still needed for AUROC/FPR95.",
        "train": summarize_gate(train_scored, threshold, "train"),
        "validation": summarize_gate(val_scored, threshold, "validation"),
        "test": summarize_gate(test_scored, threshold, "test"),
    }

    (output_dir / "gate2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_distance_plot(val_scored, test_scored, threshold, output_dir / "distance_distribution.png")

    print("Gate 2 complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
