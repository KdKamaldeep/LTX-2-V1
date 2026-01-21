#!/usr/bin/env python3
"""
Batch video generation script for LTX-2 story JSON files.

Processes JSON story files created by generate_ltx2_prompts.py and generates
videos for each scene prompt using TI2VidTwoStagesPipeline.

Requirements:
    pip install openai

Usage:
    python batch_generate_story_videos.py --json story.json --checkpoint-path path/to/checkpoint.safetensors ...
"""

import json
import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict

from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.constants import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_SEED,
    DEFAULT_2_STAGE_HEIGHT,
    DEFAULT_2_STAGE_WIDTH,
    DEFAULT_NUM_FRAMES,
    DEFAULT_FRAME_RATE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_CFG_GUIDANCE_SCALE,
    AUDIO_SAMPLE_RATE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resolve_path(path: str) -> str:
    """Resolve and expand a file path."""
    return str(Path(path).expanduser().resolve().as_posix())


def load_story_json(json_path: str) -> Dict:
    """Load and validate story JSON file."""
    json_path = resolve_path(json_path)
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Story JSON file not found: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if "prompts" not in data:
        raise ValueError("JSON file must contain a 'prompts' array")
    
    if not isinstance(data["prompts"], list) or len(data["prompts"]) == 0:
        raise ValueError("'prompts' must be a non-empty array")
    
    for i, prompt in enumerate(data["prompts"]):
        if "title" not in prompt or "prompt" not in prompt:
            raise ValueError(f"Prompt {i+1} is missing 'title' or 'prompt' field")
    
    return data


def parse_lora_args(lora_args: List[str]) -> List[LoraPathStrengthAndSDOps]:
    """Parse LoRA arguments into LoraPathStrengthAndSDOps objects."""
    if not lora_args:
        return []
    
    loras = []
    for lora_spec in lora_args:
        parts = lora_spec.split(":")
        if len(parts) == 1:
            path = parts[0]
            strength = 1.0
        elif len(parts) == 2:
            path, strength_str = parts
            strength = float(strength_str)
        else:
            raise ValueError(f"Invalid LoRA specification: {lora_spec}. Use format 'path' or 'path:strength'")
        
        resolved_path = resolve_path(path)
        loras.append(LoraPathStrengthAndSDOps(resolved_path, strength, LTXV_LORA_COMFY_RENAMING_MAP))
    
    return loras


def generate_video_for_scene(
    pipeline: TI2VidTwoStagesPipeline,
    scene: Dict,
    scene_num: int,
    output_dir: str,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    num_inference_steps: int,
    cfg_guidance_scale: float,
    negative_prompt: str,
    enhance_prompt: bool = False,
) -> str:
    """Generate a video for a single scene prompt."""
    title = scene["title"]
    prompt = scene["prompt"]
    
    # Sanitize title for filename
    safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).strip()
    safe_title = safe_title.replace(" ", "_")
    output_filename = f"scene_{scene_num:02d}_{safe_title}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    logger.info(f"Generating Scene {scene_num}: {title}")
    logger.info(f"Prompt: {prompt[:100]}...")
    logger.info(f"Output: {output_path}")
    
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
    
    # Use scene-specific seed (base seed + scene number for variation)
    scene_seed = seed + scene_num - 1
    
    # Generate video
    video, audio = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=scene_seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        num_inference_steps=num_inference_steps,
        cfg_guidance_scale=cfg_guidance_scale,
        images=[],  # No image conditioning
        tiling_config=tiling_config,
        enhance_prompt=enhance_prompt,
    )
    
    # Encode and save video
    encode_video(
        video=video,
        fps=frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=output_path,
        video_chunks_number=video_chunks_number,
    )
    
    logger.info(f"✅ Completed Scene {scene_num}: {output_filename}")
    return output_path


def main():
    """Main function to batch generate videos from story JSON."""
    parser = argparse.ArgumentParser(
        description="Batch generate videos from LTX-2 story JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_generate_story_videos.py \\
      --json story.json \\
      --checkpoint-path models/checkpoint.safetensors \\
      --distilled-lora models/distilled_lora.safetensors:0.8 \\
      --spatial-upsampler-path models/upsampler.safetensors \\
      --gemma-root models/gemma

  python batch_generate_story_videos.py \\
      --json story.json \\
      --checkpoint-path models/checkpoint.safetensors \\
      --distilled-lora models/distilled_lora.safetensors:0.8 \\
      --spatial-upsampler-path models/upsampler.safetensors \\
      --gemma-root models/gemma \\
      --output-dir videos \\
      --seed 42 \\
      --num-frames 121 \\
      --height 1024 \\
      --width 1024
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to story JSON file created by generate_ltx2_prompts.py"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to LTX-2 model checkpoint (.safetensors file)"
    )
    parser.add_argument(
        "--distilled-lora",
        type=str,
        required=True,
        help="Distilled LoRA path with optional strength (format: 'path' or 'path:strength')"
    )
    parser.add_argument(
        "--spatial-upsampler-path",
        type=str,
        required=True,
        help="Path to spatial upsampler model (.safetensors file)"
    )
    parser.add_argument(
        "--gemma-root",
        type=str,
        required=True,
        help="Path to Gemma text encoder root directory"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_videos",
        help=f"Output directory for generated videos (default: generated_videos)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base random seed (each scene uses seed + scene_number) (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_2_STAGE_HEIGHT,
        help=f"Video height in pixels, divisible by 64 (default: {DEFAULT_2_STAGE_HEIGHT})"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_2_STAGE_WIDTH,
        help=f"Video width in pixels, divisible by 64 (default: {DEFAULT_2_STAGE_WIDTH})"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help=f"Number of frames per video, (8n + 1) format (default: {DEFAULT_NUM_FRAMES})"
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=DEFAULT_FRAME_RATE,
        help=f"Frame rate (fps) (default: {DEFAULT_FRAME_RATE})"
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
        help=f"Number of denoising steps (default: {DEFAULT_NUM_INFERENCE_STEPS})"
    )
    parser.add_argument(
        "--cfg-guidance-scale",
        type=float,
        default=DEFAULT_CFG_GUIDANCE_SCALE,
        help=f"CFG guidance scale (default: {DEFAULT_CFG_GUIDANCE_SCALE})"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt (default: comprehensive negative prompt)"
    )
    parser.add_argument(
        "--lora",
        type=str,
        action="append",
        default=[],
        help="Additional LoRA models (format: 'path' or 'path:strength'). Can be specified multiple times."
    )
    parser.add_argument(
        "--enable-fp8",
        action="store_true",
        help="Enable FP8 mode to reduce memory footprint"
    )
    parser.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="Enable prompt enhancement"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scenes that already have generated videos"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    json_path = resolve_path(args.json)
    checkpoint_path = resolve_path(args.checkpoint_path)
    spatial_upsampler_path = resolve_path(args.spatial_upsampler_path)
    gemma_root = resolve_path(args.gemma_root)
    output_dir = resolve_path(args.output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load story JSON
    logger.info(f"Loading story from: {json_path}")
    story_data = load_story_json(json_path)
    scenes = story_data["prompts"]
    logger.info(f"Found {len(scenes)} scenes to generate")
    
    # Parse LoRAs
    loras = parse_lora_args(args.lora)
    
    # Parse distilled LoRA
    distilled_lora = parse_lora_args([args.distilled_lora])
    if not distilled_lora:
        raise ValueError("--distilled-lora is required")
    
    # Initialize pipeline (once, reuse for all scenes)
    logger.info("Initializing LTX-2 pipeline...")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Distilled LoRA: {args.distilled_lora}")
    logger.info(f"Spatial Upsampler: {spatial_upsampler_path}")
    logger.info(f"Gemma Root: {gemma_root}")
    logger.info(f"FP8 Enabled: {args.enable_fp8}")
    
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=loras,
        fp8transformer=args.enable_fp8,
    )
    
    logger.info("Pipeline initialized successfully!")
    logger.info("=" * 60)
    
    # Generate videos for each scene
    generated_videos = []
    for i, scene in enumerate(scenes, 1):
        title = scene["title"]
        
        # Check if video already exists
        safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).strip()
        safe_title = safe_title.replace(" ", "_")
        output_filename = f"scene_{i:02d}_{safe_title}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        if args.skip_existing and os.path.exists(output_path):
            logger.info(f"⏭️  Skipping Scene {i} (already exists): {output_filename}")
            generated_videos.append(output_path)
            continue
        
        try:
            video_path = generate_video_for_scene(
                pipeline=pipeline,
                scene=scene,
                scene_num=i,
                output_dir=output_dir,
                seed=args.seed,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                frame_rate=args.frame_rate,
                num_inference_steps=args.num_inference_steps,
                cfg_guidance_scale=args.cfg_guidance_scale,
                negative_prompt=args.negative_prompt,
                enhance_prompt=args.enhance_prompt,
            )
            generated_videos.append(video_path)
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"❌ Error generating Scene {i} ({title}): {str(e)}")
            logger.error(f"Continuing with next scene...")
            logger.info("=" * 60)
    
    # Summary
    logger.info("=" * 60)
    logger.info("BATCH GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total scenes processed: {len(scenes)}")
    logger.info(f"Videos generated: {len(generated_videos)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("Generated videos:")
    for i, video_path in enumerate(generated_videos, 1):
        logger.info(f"  {i}. {os.path.basename(video_path)}")
    
    # Save video list to JSON for potential concatenation
    video_list_path = os.path.join(output_dir, "video_list.json")
    with open(video_list_path, "w", encoding="utf-8") as f:
        json.dump({
            "story_json": json_path,
            "generated_videos": generated_videos,
            "scenes": [{"title": s["title"], "video": v} for s, v in zip(scenes, generated_videos)]
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"\nVideo list saved to: {video_list_path}")


if __name__ == "__main__":
    main()
