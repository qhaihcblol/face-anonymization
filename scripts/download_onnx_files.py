"""Download ONNX files from Kaggle dataset into ai_core/*/onnx directories.

Run from project root:
    python scripts/download_onnx_files.py
"""

from pathlib import Path
import shutil

import kagglehub

DATASET = "quanghijr/video-anonymization-onnx"

PROJECT_ROOT = Path.cwd()
AI_CORE = PROJECT_ROOT / "ai_core"

ONNX_TARGETS = {
    "gfpgan_1.4.onnx": AI_CORE / "face_restoration" / "onnx",
    "bisenet_resnet_34.onnx": AI_CORE / "face_parsing" / "onnx",
    "blendswap_256.onnx": AI_CORE / "face_swapping" / "onnx",
    "retinaface_best.onnx": AI_CORE / "face_detection" / "onnx",
    "wavlm_encoder.onnx": AI_CORE / "voice_anonymization" / "onnx",
    "hifigan_vocoder.onnx": AI_CORE / "voice_anonymization" / "onnx",
}


def main() -> None:
    dataset_path = Path(kagglehub.dataset_download(DATASET))
    print(f"Dataset path: {dataset_path}")

    copied = 0

    for filename, target_dir in ONNX_TARGETS.items():
        matches = list(dataset_path.rglob(filename))

        if not matches:
            print(f"[MISSING] {filename}")
            continue

        source = matches[0]
        target_dir.mkdir(parents=True, exist_ok=True)

        destination = target_dir / filename
        shutil.copy2(source, destination)

        copied += 1
        print(f"[OK] {filename} -> {destination.relative_to(PROJECT_ROOT)}")

        # Copy external-weights sidecars (e.g. model.onnx.data) sitting next to the .onnx.
        for sidecar in source.parent.glob(filename + ".*"):
            shutil.copy2(sidecar, target_dir / sidecar.name)
            print(f"[OK] {sidecar.name} -> {(target_dir / sidecar.name).relative_to(PROJECT_ROOT)}")

    print(f"\nCopied {copied}/{len(ONNX_TARGETS)} ONNX files.")


if __name__ == "__main__":
    main()
