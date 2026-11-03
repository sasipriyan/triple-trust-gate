from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from gate1_confidence import enable_mc_dropout, predictive_entropy
from gate2_ood import FeatureExtractor, add_distance_scores
from gate3_gradcam import (
    GradCAM,
    LABELS,
    build_eval_transform,
    build_model,
    cam_to_mask,
    centroid_distance,
    iou_score,
    pseudo_lesion_mask,
    real_lesion_mask,
    save_overlay,
    target_layer_for,
)


DEFAULT_LABEL_NAMES = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one image through the Triple Trust-Gate pipeline.")
    parser.add_argument("--image", type=Path, required=True, help="Path to a skin lesion image.")
    parser.add_argument("--data-dir", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/baseline/models/best_baseline.pt"))
    parser.add_argument("--gate1-summary", type=Path, default=Path("outputs/gate1_mc_dropout/gate1_summary.json"))
    parser.add_argument("--gate2-summary", type=Path, default=Path("outputs/gate2_ood/gate2_summary.json"))
    parser.add_argument("--gate2-centroids", type=Path, default=Path("outputs/gate2_ood/class_centroids.npz"))
    parser.add_argument("--gate3-summary", type=Path, default=Path("outputs/gate3_gradcam/gate3_summary.json"))
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("external/isic_segmentation/dataverse_files/HAM10000_segmentations_lesion_tschandl"),
    )
    parser.add_argument("--class-mapping", type=Path, default=Path("outputs/baseline/class_mapping.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/single_image_predictions"))
    parser.add_argument("--mc-samples", type=int, default=5)
    return parser.parse_args()


def resolve_path(data_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else (data_dir / path).resolve()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_label_names(path: Path) -> dict[str, str]:
    if not path.exists():
        return DEFAULT_LABEL_NAMES
    mapping = load_json(path)
    return mapping.get("label_names", DEFAULT_LABEL_NAMES)


def image_tensor(image_path: Path, transform) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def gate1_predict(
    model: nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    mc_samples: int,
) -> dict:
    sample_probs = []
    with torch.no_grad():
        for _ in range(mc_samples):
            if mc_samples > 1:
                enable_mc_dropout(model)
            else:
                model.eval()
            logits = model(tensor.to(device))
            probs = torch.softmax(logits, dim=1)
            sample_probs.append(probs.detach().cpu().numpy())

    probs_np = np.mean(sample_probs, axis=0)
    pred_index = int(probs_np.argmax(axis=1)[0])
    confidence = float(probs_np.max(axis=1)[0])
    entropy = float(predictive_entropy(probs_np)[0])

    if mc_samples > 1:
        stacked = np.stack(sample_probs, axis=0)
        confidence_std = float(stacked.max(axis=2).std(axis=0)[0])
    else:
        confidence_std = 0.0

    return {
        "pred_index": pred_index,
        "pred_label": LABELS[pred_index],
        "confidence": confidence,
        "predictive_entropy": entropy,
        "mc_confidence_std": confidence_std,
    }


def gate2_score(
    extractor: FeatureExtractor,
    tensor: torch.Tensor,
    centroids: np.ndarray,
    pred_index: int,
    pred_label: str,
    confidence: float,
    device: torch.device,
) -> dict:
    extractor.eval()
    with torch.no_grad():
        embeddings, _logits = extractor(tensor.to(device))

    # Reuse the same scoring helper used in the dataset-level Gate 2 run.
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "image_id": "single_image",
                "true_label": "unknown",
                "true_index": pred_index,
                "pred_label": pred_label,
                "pred_index": pred_index,
                "confidence": confidence,
                "correct": 0,
            }
        ]
    )
    scored = add_distance_scores(df, embeddings.detach().cpu().numpy(), centroids).iloc[0]
    return {
        "nearest_centroid_label": str(scored["nearest_centroid_label"]),
        "min_centroid_distance": float(scored["min_centroid_distance"]),
        "predicted_class_distance": float(scored["predicted_class_distance"]),
        "nearest_matches_prediction": bool(int(scored["nearest_matches_prediction"])),
    }


def gate3_score(
    model: nn.Module,
    tensor: torch.Tensor,
    image_path: Path,
    pred_index: int,
    image_size: int,
    cam_percentile: float,
    mask_dir: Path | None,
    output_path: Path,
    device: torch.device,
    title: str,
) -> dict:
    gradcam = GradCAM(model, target_layer_for(model, "efficientnet_b0"))
    class_indexes = torch.tensor([pred_index], dtype=torch.long, device=device)
    _logits, cams = gradcam(tensor.to(device), class_indexes)
    gradcam.remove_hooks()

    cam = cams[0].detach().cpu().numpy()
    lesion_mask = None
    mask_source = "pseudo_image_processing"
    if mask_dir is not None and mask_dir.exists():
        lesion_mask = real_lesion_mask(mask_dir, image_path.stem, image_size)
        if lesion_mask is not None:
            mask_source = "official_segmentation_mask"

    if lesion_mask is None:
        lesion_mask = pseudo_lesion_mask(str(image_path), image_size)

    cam_mask = cam_to_mask(cam, cam_percentile)

    iou = iou_score(cam_mask, lesion_mask)
    distance = centroid_distance(cam_mask, lesion_mask)
    consistency_score = 0.5 * iou + 0.5 * (1.0 - distance)

    save_overlay(str(image_path), cam, lesion_mask, cam_mask, output_path, image_size, title)

    return {
        "cam_lesion_iou": float(iou),
        "centroid_distance": float(distance),
        "consistency_score": float(consistency_score),
        "mask_source": mask_source,
        "gradcam_overlay": str(output_path),
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    image_path = resolve_path(data_dir, args.image)
    output_dir = resolve_path(data_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = resolve_path(data_dir, args.checkpoint)
    gate1_summary_path = resolve_path(data_dir, args.gate1_summary)
    gate2_summary_path = resolve_path(data_dir, args.gate2_summary)
    gate2_centroids_path = resolve_path(data_dir, args.gate2_centroids)
    gate3_summary_path = resolve_path(data_dir, args.gate3_summary)
    mask_dir = resolve_path(data_dir, args.mask_dir)
    class_mapping_path = resolve_path(data_dir, args.class_mapping)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint["model_name"]
    image_size = int(checkpoint.get("image_size", 224))

    if model_name != "efficientnet_b0":
        raise ValueError("Single-image Gate 3 currently expects the EfficientNet-B0 baseline.")

    model = build_model(model_name, len(LABELS))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    extractor = FeatureExtractor(model_name, checkpoint).to(device)
    transform = build_eval_transform(image_size)
    tensor = image_tensor(image_path, transform)

    gate1_summary = load_json(gate1_summary_path)
    gate2_summary = load_json(gate2_summary_path)
    gate3_summary = load_json(gate3_summary_path)
    label_names = load_label_names(class_mapping_path)
    centroids = np.load(gate2_centroids_path)["centroids"]

    gate1 = gate1_predict(model, tensor, device, args.mc_samples)
    gate1_threshold = float(gate1_summary["threshold"])
    gate1_status = "pass" if gate1["confidence"] >= gate1_threshold else "refer"

    gate2 = gate2_score(
        extractor,
        tensor,
        centroids,
        gate1["pred_index"],
        gate1["pred_label"],
        gate1["confidence"],
        device,
    )
    gate2_threshold = float(gate2_summary["threshold"])
    gate2_status = "known" if gate2["min_centroid_distance"] <= gate2_threshold else "refer"

    safe_stem = image_path.stem.replace(" ", "_")
    overlay_path = output_dir / f"{safe_stem}_gradcam.png"
    gate3_title = f"{image_path.name} pred={gate1['pred_label']} confidence={gate1['confidence']:.3f}"
    gate3 = gate3_score(
        model,
        tensor,
        image_path,
        gate1["pred_index"],
        image_size,
        float(gate3_summary["cam_percentile"]),
        mask_dir if mask_dir.exists() else None,
        overlay_path,
        device,
        gate3_title,
    )
    gate3_status = (
        "pass"
        if gate3["cam_lesion_iou"] >= float(gate3_summary["iou_threshold"])
        and gate3["centroid_distance"] <= float(gate3_summary["centroid_threshold"])
        else "refer"
    )

    gate1_result = "PASS" if gate1_status == "pass" else "FAIL"
    gate2_result = "PASS" if gate2_status == "known" else "FAIL"
    gate3_result = "PASS" if gate3_status == "pass" else "FAIL"

    all_pass = gate1_status == "pass" and gate2_status == "known" and gate3_status == "pass"
    final_decision = "reliable_diagnosis" if all_pass else "refer_to_specialist"

    failure_reasons = []
    if gate1_status != "pass":
        failure_reasons.append("low_confidence")
    if gate2_status != "known":
        failure_reasons.append("feature_space_outlier")
    if gate3_status != "pass":
        failure_reasons.append("explanation_mismatch")

    result = {
        "image": str(image_path),
        "prediction": {
            "class_code": gate1["pred_label"],
            "disease_name": label_names.get(gate1["pred_label"], gate1["pred_label"]),
            "confidence": gate1["confidence"],
        },
        "gates": {
            "gate_1_confidence": {
                "result": gate1_result,
                "confidence": gate1["confidence"],
                "threshold": gate1_threshold,
            },
            "gate_2_ood_detection": {
                "result": gate2_result,
                "distance": gate2["min_centroid_distance"],
                "threshold": gate2_threshold,
                "nearest_class": gate2["nearest_centroid_label"],
            },
            "gate_3_explanation": {
                "result": gate3_result,
                "gradcam_lesion_iou": gate3["cam_lesion_iou"],
                "threshold": float(gate3_summary["iou_threshold"]),
                "mask_source": gate3["mask_source"],
            },
        },
        "final_decision": final_decision,
        "failure_reasons": failure_reasons if failure_reasons else ["none"],
        "gradcam_overlay": gate3["gradcam_overlay"],
    }

    result_path = output_dir / f"{safe_stem}_prediction.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nSaved JSON: {result_path}")
    print(f"Saved Grad-CAM overlay: {overlay_path}")


if __name__ == "__main__":
    main()
