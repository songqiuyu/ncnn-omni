#!/usr/bin/env python3
"""Compare ncnn-omni greedy token IDs with the original Transformers model.

The script is intentionally a development tool, not a runtime dependency.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from qwen3_asr_reference import enable_local_qwen_asr


def ncnn_tokens(args: argparse.Namespace) -> list[int]:
    command = [
        str(args.binary),
        "--model", str(args.ncnn_model),
        "--assets", str(args.transformers_model),
        "--audio", str(args.audio),
        "--max-new-tokens", str(args.max_new_tokens),
        "--threads", str(args.threads),
    ]
    if args.language:
        command += ["--language", args.language]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    print(completed.stdout, end="")
    match = re.search(r"^token_ids:(.*)$", completed.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError("ncnn CLI did not print token_ids")
    return [int(value) for value in match.group(1).split()]


def reference_tokens(args: argparse.Namespace) -> list[int]:
    enable_local_qwen_asr()
    source_root = args.qwen_asr_source.resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from qwen_asr import Qwen3ASRModel

    waveform, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if waveform.ndim != 1 or sample_rate != 16000:
        raise ValueError("parity input must be mono 16 kHz")
    # Apply the same documented compatibility canonicalization as ncnn-omni.
    # Without it, upstream Qwen3-ASR has ceil(mask) vs floor(Mel) lengths for
    # a partial final hop and fails inside split_with_sizes.
    waveform = np.pad(waveform, (0, (-len(waveform)) % 160))
    asr = Qwen3ASRModel.from_pretrained(
        str(args.transformers_model),
        dtype=torch.float32,
        device_map="cpu",
        max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens,
    )
    prompt = asr._build_text_prompt("", args.language or None)
    inputs = asr.processor(
        text=[prompt], audio=[waveform], return_tensors="pt", padding=True
    ).to(asr.model.device).to(asr.model.dtype)
    output = asr.model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    ids = output.sequences[0, inputs["input_ids"].shape[1] :].tolist()
    while ids and ids[-1] in (151643, 151645):
        ids.pop()
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--ncnn-model", type=Path, required=True)
    parser.add_argument("--transformers-model", type=Path, required=True)
    parser.add_argument(
        "--qwen-asr-source",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Qwen3-ASR Python source checkout",
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--language", default="")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    actual = ncnn_tokens(args)
    expected = reference_tokens(args)
    print(f"Transformers token_ids: {expected}")
    if actual != expected:
        for index, (left, right) in enumerate(zip(actual, expected)):
            if left != right:
                raise SystemExit(f"FAIL: first mismatch at {index}: ncnn={left}, torch={right}")
        raise SystemExit(f"FAIL: token count differs: ncnn={len(actual)}, torch={len(expected)}")
    print(f"PASS: {len(actual)} generated tokens are exactly equal")


if __name__ == "__main__":
    main()
