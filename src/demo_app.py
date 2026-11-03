from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "outputs" / "demo_uploads"
PREDICTION_DIR = ROOT / "outputs" / "single_image_predictions"


def run_prediction(image_path: Path) -> dict:
    command = [
        sys.executable,
        str(ROOT / "src" / "predict_single_image.py"),
        "--image",
        str(image_path),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)

    result_path = PREDICTION_DIR / f"{image_path.stem}_prediction.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Prediction JSON was not created: {result_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def gate_badge(status: str) -> str:
    if status.upper() == "PASS":
        return f":green[{status}]"
    return f":orange[{status}]"


st.set_page_config(page_title="Triple Trust-Gate Demo", layout="wide")

st.title("Triple Trust-Gate Skin Lesion Demo")
st.caption("Research prototype. Not a medical diagnostic tool.")

uploaded = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    image_path = UPLOAD_DIR / uploaded.name
    image_path.write_bytes(uploaded.getbuffer())

    left, right = st.columns([1, 1])
    with left:
        st.image(str(image_path), caption="Uploaded image", width=350)

    if st.button("Run Triple Trust-Gate", type="primary"):
        with st.spinner("Running prediction and trust gates..."):
            try:
                result = run_prediction(image_path)
            except Exception as exc:
                st.error(str(exc))
                st.stop()

        prediction = result["prediction"]
        gates = result.get("gates", {})
        gate1 = gates.get("gate_1_confidence", result.get("gate1_confidence"))
        gate2 = gates.get("gate_2_ood_detection", result.get("gate2_ood"))
        gate3 = gates.get("gate_3_explanation", result.get("gate3_explanation"))

        if not all([gate1, gate2, gate3]):
            st.error("The prediction result is missing one or more gate outputs.")
            st.stop()

        with right:
            st.subheader("Prediction")
            st.metric("Disease", prediction["disease_name"])
            st.metric("Class label", prediction.get("class_code", prediction.get("label", "Unknown")))
            st.metric("Confidence", f"{prediction['confidence'] * 100:.2f}%")
            st.metric("Final decision", result["final_decision"].replace("_", " "))

        st.subheader("Gate Results")
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Gate 1 Confidence:** {gate_badge(gate1.get('result', gate1.get('status', 'FAIL')))}")
        col1.write(f"Confidence: `{gate1['confidence']:.2f}`")
        col1.write(f"Threshold: `{gate1['threshold']:.2f}`")

        col2.markdown(f"**Gate 2 OOD:** {gate_badge(gate2.get('result', gate2.get('status', 'FAIL')))}")
        col2.write(f"Distance: `{gate2.get('distance', gate2.get('min_centroid_distance', 0.0)):.4f}`")
        col2.write(f"Nearest known class: `{gate2.get('nearest_class', gate2.get('nearest_centroid_label', 'Unknown'))}`")

        col3.markdown(f"**Gate 3 Explanation:** {gate_badge(gate3.get('result', gate3.get('status', 'FAIL')))}")
        gate3_iou = gate3.get("gradcam_lesion_iou", gate3.get("cam_lesion_iou", 0.0))
        col3.write(f"IoU: `{gate3_iou:.4f}`")
        col3.write(f"Mask source: `{gate3['mask_source']}`")

        if result["failure_reasons"] != ["none"]:
            st.warning("Referral reasons: " + ", ".join(result["failure_reasons"]))
        else:
            st.success("All gates passed.")

        st.subheader("Grad-CAM Explanation")
        st.image(result.get("gradcam_overlay", gate3.get("gradcam_overlay")), use_container_width=True)

        with st.expander("Full JSON result"):
            st.json(result)
