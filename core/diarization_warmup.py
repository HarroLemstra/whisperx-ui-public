from __future__ import annotations

import argparse

from pyannote.audio import Pipeline
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Warm up pyannote diarization pipeline cache.")
    parser.add_argument("--model", required=True, help="Diarization model id, e.g. pyannote/speaker-diarization-3.1")
    parser.add_argument("--hf_token", required=True, help="Hugging Face token with access to gated models")
    parser.add_argument("--device", default="cpu", help="Device to initialize the pipeline on")
    args = parser.parse_args()

    pipeline = Pipeline.from_pretrained(args.model, use_auth_token=args.hf_token)
    pipeline.to(torch.device(args.device))
    print("ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
