#!/usr/bin/env python3
"""Compare the actual ncnn-omni audio frontend with Qwen3-ASR preprocessing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from transformers import WhisperFeatureExtractor

from qwen3_asr_reference import enable_local_qwen_asr


SOURCE_ROOT = Path(__file__).resolve().parents[3]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


enable_local_qwen_asr()

from qwen_asr.inference.utils import normalize_audio_input  # noqa: E402


PCM_MAX_ABS = 1e-7
MEL_MAX_ABS = 5e-5
MEL_MEAN_ABS = 5e-6
MEL_MIN_COSINE = 0.99999999


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float | bool]:
    shape_match = reference.shape == actual.shape
    if not shape_match:
        return {"shape_match": False}
    ref = reference.astype(np.float64, copy=False).reshape(-1)
    got = actual.astype(np.float64, copy=False).reshape(-1)
    delta = np.abs(ref - got)
    ref_norm = float(np.linalg.norm(ref))
    got_norm = float(np.linalg.norm(got))
    if ref_norm == 0.0 and got_norm == 0.0:
        cosine = 1.0
    elif ref_norm == 0.0 or got_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.dot(ref, got) / (ref_norm * got_norm))
    return {
        "shape_match": True,
        "finite": bool(np.isfinite(ref).all() and np.isfinite(got).all()),
        "max_abs_error": float(delta.max(initial=0.0)),
        "mean_abs_error": float(delta.mean()) if delta.size else 0.0,
        "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "cosine_similarity": cosine,
        "reference_min": float(ref.min()) if ref.size else 0.0,
        "reference_max": float(ref.max()) if ref.size else 0.0,
        "ncnn_min": float(got.min()) if got.size else 0.0,
        "ncnn_max": float(got.max()) if got.size else 0.0,
    }


def compare_one(
    audio_path: Path,
    dump_binary: Path,
    feature_extractor: WhisperFeatureExtractor,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="qwen3_asr_frontend_") as directory:
        output = Path(directory)
        subprocess.run(
            [str(dump_binary), "--audio", str(audio_path), "--output", str(output)],
            check=True,
        )
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        ncnn_pcm = np.fromfile(output / "normalized_pcm.f32", dtype=np.float32)
        ncnn_frontend_pcm = np.fromfile(output / "frontend_pcm.f32", dtype=np.float32)
        mel_shape = tuple(metadata["input_features_shape"])
        ncnn_mel = np.fromfile(output / "input_features.f32", dtype=np.float32).reshape(mel_shape)

    # The official example decodes with SoundFile and passes (waveform, sr) to
    # Qwen3ASRModel. Using that same public normalization path avoids a second
    # decoder implementation becoming part of the numeric reference.
    decoded, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    reference_pcm = normalize_audio_input((decoded, sample_rate)).astype(np.float32, copy=False)
    pad = (-reference_pcm.size) % feature_extractor.hop_length
    reference_frontend_pcm = np.pad(reference_pcm, (0, pad)).astype(np.float32, copy=False)

    # Preserve the untouched upstream observation in the report. For a partial
    # final hop, Qwen currently returns floor Mel frames and a ceil-length mask,
    # which later fails in the Audio Encoder's split_with_sizes call.
    raw_processed = feature_extractor(
        [reference_pcm],
        padding=True,
        truncation=False,
        return_attention_mask=True,
        return_tensors="np",
        sampling_rate=16000,
    )
    raw_reference_mel = np.asarray(raw_processed["input_features"][0], dtype=np.float32)
    raw_feature_mask = np.asarray(raw_processed["attention_mask"][0])

    processed = feature_extractor(
        [reference_frontend_pcm],
        padding=True,
        truncation=False,
        return_attention_mask=True,
        return_tensors="np",
        sampling_rate=16000,
    )
    reference_mel = np.asarray(processed["input_features"][0], dtype=np.float32)
    feature_mask = np.asarray(processed["attention_mask"][0])

    pcm = metrics(reference_pcm, ncnn_pcm)
    frontend_pcm = metrics(reference_frontend_pcm, ncnn_frontend_pcm)
    mel = metrics(reference_mel, ncnn_mel)
    processor_feature_length = int(feature_mask.sum())
    processor_consistent = processor_feature_length == reference_mel.shape[1]
    pcm_passed = bool(
        pcm.get("shape_match")
        and pcm.get("finite")
        and pcm.get("max_abs_error", float("inf")) <= PCM_MAX_ABS
    )
    frontend_pcm_passed = bool(
        frontend_pcm.get("shape_match")
        and frontend_pcm.get("finite")
        and frontend_pcm.get("max_abs_error", float("inf")) <= PCM_MAX_ABS
    )
    mel_passed = bool(
        mel.get("shape_match")
        and mel.get("finite")
        and mel.get("max_abs_error", float("inf")) <= MEL_MAX_ABS
        and mel.get("mean_abs_error", float("inf")) <= MEL_MEAN_ABS
        and mel.get("cosine_similarity", 0.0) >= MEL_MIN_COSINE
    )
    return {
        "audio": str(audio_path.resolve()),
        "input": {
            "sample_rate": metadata["sample_rate"],
            "channels": metadata["channels"],
            "samples": int(reference_pcm.size),
            "hop_aligned": bool(reference_pcm.size % feature_extractor.hop_length == 0),
            "compatibility_padding_samples": int(pad),
            "frontend_samples": int(reference_frontend_pcm.size),
        },
        "untouched_qwen_processor": {
            "input_features_shape": list(raw_reference_mel.shape),
            "feature_attention_mask_shape": list(raw_feature_mask.shape),
            "feature_attention_mask_sum": int(raw_feature_mask.sum()),
            "mask_matches_mel_frames": bool(int(raw_feature_mask.sum()) == raw_reference_mel.shape[1]),
        },
        "canonical_qwen_processor": {
            "input_features_shape": list(reference_mel.shape),
            "feature_attention_mask_shape": list(feature_mask.shape),
            "feature_attention_mask_sum": processor_feature_length,
            "mask_matches_mel_frames": processor_consistent,
        },
        "ncnn_frontend": {
            "input_features_shape": list(ncnn_mel.shape),
            "pcm_dtype": str(ncnn_pcm.dtype),
            "input_features_dtype": str(ncnn_mel.dtype),
        },
        "pcm": pcm,
        "canonical_frontend_pcm": frontend_pcm,
        "log_mel": mel,
        "passed": pcm_passed and frontend_pcm_passed and mel_passed and processor_consistent,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-binary", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--audio", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.assets)
    cases = [compare_one(path, args.dump_binary.resolve(), feature_extractor) for path in args.audio]
    report = {
        "purpose": "Qwen3-ASR normalized PCM and full Log-Mel parity",
        "reference": (
            "qwen_asr normalize_audio_input + documented final-hop zero padding + "
            "Transformers WhisperFeatureExtractor torch path"
        ),
        "scope": "mono 16 kHz PCM16/float32 WAV accepted by ncnn-omni v1",
        "parameters": {
            "sampling_rate": feature_extractor.sampling_rate,
            "mel_bins": feature_extractor.feature_size,
            "n_fft": feature_extractor.n_fft,
            "hop_length": feature_extractor.hop_length,
            "dither": feature_extractor.dither,
            "padding": True,
            "truncation": False,
            "compatibility_padding": "right zero pad to a 160-sample multiple; 0..159 samples",
        },
        "tolerances": {
            "pcm_max_abs": PCM_MAX_ABS,
            "mel_max_abs": MEL_MAX_ABS,
            "mel_mean_abs": MEL_MEAN_ABS,
            "mel_min_cosine": MEL_MIN_COSINE,
        },
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for case in cases:
        mel = case["log_mel"]
        print(
            f"{'PASS' if case['passed'] else 'FAIL'} {case['audio']} "
            f"shape={case['ncnn_frontend']['input_features_shape']} "
            f"max_abs={mel.get('max_abs_error', 'shape-mismatch')} "
            f"mean_abs={mel.get('mean_abs_error', 'shape-mismatch')} "
            f"cosine={mel.get('cosine_similarity', 'shape-mismatch')}"
        )
    print(f"report: {args.report.resolve()}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
