from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

WAVLM_LAYER = 6
FEATURE_DIM = 1024


class WavLMEncoderONNX(nn.Module):
    """
    Export WavLM encoder used by kNN-VC.

    Input:
        wav: float32 tensor, shape (1, n_samples), 16 kHz mono audio

    Output:
        features: float32 tensor, shape (1, n_frames, 1024)
    """

    def __init__(self, wavlm: nn.Module, layer: int = WAVLM_LAYER):
        super().__init__()
        self.wavlm = wavlm
        self.layer = layer

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        features = self.wavlm.extract_features(
            wav,
            output_layer=self.layer,
            ret_layer_results=False,
        )[0]
        return features


class HiFiGANVocoderONNX(nn.Module):
    """
    Export prematched HiFi-GAN vocoder used by kNN-VC.

    Input:
        features: float32 tensor, shape (1, n_frames, 1024)

    Output:
        wav: float32 tensor, shape (1, n_samples)
    """

    def __init__(self, hifigan: nn.Module):
        super().__init__()
        self.hifigan = hifigan

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        wav = self.hifigan(features)

        # kNN-VC HiFi-GAN may return (B, 1, N)
        if wav.dim() == 3 and wav.shape[1] == 1:
            wav = wav.squeeze(1)

        return wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export kNN-VC WavLM + HiFi-GAN to ONNX")

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ai_core/voice_anonymization/onnx"),
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
    )
    parser.add_argument(
        "--dummy-seconds",
        type=float,
        default=1.0,
        help="Dummy audio length for tracing. Default: 1 second.",
    )
    parser.add_argument(
        "--prematched",
        action="store_true",
        default=True,
        help="Use prematched HiFi-GAN from kNN-VC.",
    )

    return parser.parse_args()


def export_onnx(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict,
    opset: int,
) -> None:
    model.eval()

    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            # IMPORTANT:
            # Do not use torch.export / dynamo exporter for WavLM.
            # WavLM attention contains permute + view patterns that break
            # symbolic-shape decomposition in the new exporter.
            dynamo=False,
        )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    print("Loading official kNN-VC from torch.hub...")
    knn_vc = torch.hub.load(
        "bshall/knn-vc",
        "knn_vc",
        prematched=args.prematched,
        pretrained=True,
        trust_repo=True,
        device="cpu",
    )

    encoder = WavLMEncoderONNX(knn_vc.wavlm).to(device).eval()
    vocoder = HiFiGANVocoderONNX(knn_vc.hifigan).to(device).eval()

    n_samples = int(16000 * args.dummy_seconds)
    dummy_wav = torch.randn(1, n_samples, dtype=torch.float32, device=device)

    print("Running sanity check...")
    with torch.no_grad():
        dummy_features = encoder(dummy_wav)
        dummy_out_wav = vocoder(dummy_features)

    print(f"WavLM encoder: {tuple(dummy_wav.shape)} -> {tuple(dummy_features.shape)}")
    print(
        f"HiFi-GAN vocoder: {tuple(dummy_features.shape)} -> {tuple(dummy_out_wav.shape)}"
    )

    if dummy_features.dim() != 3 or dummy_features.shape[-1] != FEATURE_DIM:
        raise RuntimeError(
            f"Unexpected WavLM feature shape: {tuple(dummy_features.shape)}. "
            f"Expected (1, T, {FEATURE_DIM})."
        )

    encoder_path = args.out_dir / "wavlm_encoder.onnx"
    vocoder_path = args.out_dir / "hifigan_vocoder.onnx"

    print(f"Exporting WavLM encoder to: {encoder_path}")
    export_onnx(
        model=encoder,
        inputs=(dummy_wav,),
        output_path=encoder_path,
        input_names=["wav"],
        output_names=["features"],
        dynamic_axes={
            "wav": {1: "n_samples"},
            "features": {1: "n_frames"},
        },
        opset=args.opset,
    )

    print(f"Exporting HiFi-GAN vocoder to: {vocoder_path}")
    export_onnx(
        model=vocoder,
        inputs=(dummy_features,),
        output_path=vocoder_path,
        input_names=["features"],
        output_names=["wav"],
        dynamic_axes={
            "features": {1: "n_frames"},
            "wav": {1: "n_samples"},
        },
        opset=args.opset,
    )

    print("Done.")
    print(f"Saved: {encoder_path}")
    print(f"Saved: {vocoder_path}")


if __name__ == "__main__":
    main()
