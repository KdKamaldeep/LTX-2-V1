#!/usr/bin/env python3
import os
import re
import json
import time
import argparse
from pathlib import Path

import torch

from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number


def slugify(text: str, max_len: int = 80) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] if text else "untitled"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Batch generate videos with LTX-2 two-stage pipeline (load once).")

    ap.add_argument("--input-json", required=True, help="Path to JSON: {settings:{}, prompts:[{title,prompt},...]}")
    ap.add_argument("--out-dir", default="/workspace/out_batch", help="Output directory.")

    ap.add_argument("--checkpoint-path", required=True)
    ap.add_argument("--distilled-lora-path", required=True)
    ap.add_argument("--distilled-lora-strength", type=float, default=0.8)
    ap.add_argument("--spatial-upsampler-path", required=True)
    ap.add_argument("--gemma-root", required=True)

    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1216)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--negative-prompt", default="text, watermark, logo, blurry, low quality, distorted, glitch, jitter")
    ap.add_argument("--enable-fp8", action="store_true")

    args = ap.parse_args()

    # New allocator env var (your warning said PYTORCH_CUDA_ALLOC_CONF is deprecated)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = load_json(Path(args.input_json))
    settings = payload.get("settings", {})
    prompts = payload.get("prompts", [])
    if not prompts:
        raise SystemExit("No prompts[] found in JSON.")

    # Settings (fallbacks)
    num_frames = int(settings.get("num_frames", 241))
    frame_rate = float(settings.get("frame_rate", 24))
    cfg_scale = float(settings.get("cfg_guidance_scale", 3.5))
    steps = int(settings.get("num_inference_steps", 30))

    # Tiling config + chunk count (same as CLI main)
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    # Build distilled LoRA list (this pipeline requires list[LoraPathStrengthAndSDOps])
    distilled_lora = [LoraPathStrengthAndSDOps(args.distilled_lora_path, args.distilled_lora_strength, None)]

    print("=" * 80)
    print("Loading TI2VidTwoStagesPipeline ONCE...")
    print("=" * 80)

    pipe = TI2VidTwoStagesPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=[],
        fp8transformer=bool(args.enable_fp8),
    )

    print(f"Batch settings: frames={num_frames}, fps={frame_rate}, steps={steps}, cfg={cfg_scale}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Out dir: {out_dir}")

    for i, item in enumerate(prompts, start=1):
        title = item.get("title", f"prompt_{i}")
        prompt = (item.get("prompt") or "").strip()
        if not prompt:
            print(f"[{i}/{len(prompts)}] SKIP empty prompt: {title}")
            continue

        out_name = f"{i:02d}_{slugify(title)}.mp4"
        out_path = out_dir / out_name

        print("-" * 80)
        print(f"[{i}/{len(prompts)}] Generating: {title}")
        print(f"Output: {out_path}")

        t0 = time.time()

        with torch.inference_mode():
            video_iter, audio = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                seed=args.seed + i,
                height=args.height,
                width=args.width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=steps,
                cfg_guidance_scale=cfg_scale,
                images=[],
                tiling_config=tiling_config,
            )

            # Save MP4 (exactly how ti2vid_two_stages.py main() does it)
            encode_video(
                video=video_iter,
                fps=frame_rate,
                audio=audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=str(out_path),
                video_chunks_number=video_chunks_number,
            )

        dt = time.time() - t0
        print(f"Done in {dt:.1f}s -> {out_path}")

    print("=" * 80)
    print("All done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
