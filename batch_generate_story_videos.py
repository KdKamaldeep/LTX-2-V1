#!/usr/bin/env python3
"""
Batch video generation script for LTX-2 story JSON files.

Processes JSON story files created by generate_ltx2_prompts.py and generates
videos for each scene prompt using the CLI command directly.

Usage:
    python batch_generate_story_videos.py --json story.json --checkpoint-path path/to/checkpoint.safetensors ...
"""

import json
import os
import argparse
import logging
import subprocess
import shlex
import tempfile
import shutil
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resolve_path(path: str) -> str:
    """Resolve and expand a file path."""
    return str(Path(path).expanduser().resolve())


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


def stitch_videos_with_ffmpeg(
    video_paths: list[str],
    output_path: str,
) -> str:
    """Stitch multiple videos together using ffmpeg concat demuxer."""
    if not video_paths:
        raise ValueError("No videos to stitch")
    
    # Filter out None values (failed generations)
    valid_videos = [v for v in video_paths if v is not None and os.path.exists(v)]
    if not valid_videos:
        raise ValueError("No valid videos to stitch")
    
    if len(valid_videos) == 1:
        logger.info(f"Only one video, copying to {output_path}")
        shutil.copy2(valid_videos[0], output_path)
        return output_path
    
    logger.info(f"Stitching {len(valid_videos)} videos into {output_path}")
    
    # Create a temporary file list for ffmpeg concat demuxer
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_file = f.name
        for video_path in valid_videos:
            # Use absolute paths and escape single quotes
            abs_path = os.path.abspath(video_path).replace("'", "'\\''")
            f.write(f"file '{abs_path}'\n")
    
    try:
        # Use ffmpeg concat demuxer for seamless concatenation
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",  # Copy streams without re-encoding (fast)
            output_path,
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        
        logger.info(f"✅ Successfully stitched videos to {output_path}")
        if result.stderr:
            logger.debug(f"ffmpeg output: {result.stderr}")
        
        return output_path
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout or str(e)
        logger.error(f"❌ Error stitching videos: {error_msg}")
        raise
    finally:
        # Clean up temp file
        if os.path.exists(concat_file):
            os.unlink(concat_file)


def generate_video_for_scene(
    scene: Dict,
    scene_num: int,
    output_path: str,
    checkpoint_path: str,
    distilled_lora: str,
    spatial_upsampler_path: str,
    gemma_root: str,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    num_inference_steps: int,
    cfg_guidance_scale: float,
    negative_prompt: str,
    enable_fp8: bool = False,
    loras: list[str] = None,
) -> str:
    """Generate a video for a single scene prompt using CLI command."""
    title = scene["title"]
    prompt = scene["prompt"]
    
    logger.info(f"Generating Scene {scene_num}: {title}")
    logger.info(f"Prompt: {prompt[:100]}...")
    logger.info(f"Output: {output_path}")
    
    # Use scene-specific seed (base seed + scene number for variation)
    scene_seed = seed + scene_num - 1
    
    # Build the CLI command
    cmd = [
        "python", "-m", "ltx_pipelines.ti2vid_two_stages",
        "--checkpoint-path", checkpoint_path,
        "--distilled-lora", distilled_lora,
        "--spatial-upsampler-path", spatial_upsampler_path,
        "--gemma-root", gemma_root,
        "--height", str(height),
        "--width", str(width),
        "--num-frames", str(num_frames),
        "--frame-rate", str(frame_rate),
        "--num-inference-steps", str(num_inference_steps),
        "--cfg-guidance-scale", str(cfg_guidance_scale),
        "--prompt", prompt,
        "--negative-prompt", negative_prompt,
        "--seed", str(scene_seed),
        "--output-path", output_path,
    ]
    
    if enable_fp8:
        cmd.append("--enable-fp8")
    
    if loras:
        for lora in loras:
            cmd.extend(["--lora", lora])
    
    # Set environment variable for CUDA allocator
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Run the command
    logger.info(f"Running command: {' '.join(shlex.quote(str(arg)) for arg in cmd[:10])}...")
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"✅ Completed Scene {scene_num}: {os.path.basename(output_path)}")
        if result.stdout:
            logger.debug(f"Command output: {result.stdout}")
        return output_path
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout or str(e)
        logger.error(f"❌ Error generating Scene {scene_num}: {error_msg}")
        raise


def main():
    """Main function to batch generate videos from story JSON."""
    parser = argparse.ArgumentParser(
        description="Batch generate videos from LTX-2 story JSON files using CLI commands",
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
      --num-frames 241 \\
      --height 704 \\
      --width 1216 \\
      --frame-rate 24 \\
      --num-inference-steps 35 \\
      --cfg-guidance-scale 4.0 \\
      --enable-fp8
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
        help="Output directory for generated videos (default: generated_videos)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (each scene uses seed + scene_number) (default: 42)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=704,
        help="Video height in pixels, divisible by 64 (default: 704)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1216,
        help="Video width in pixels, divisible by 64 (default: 1216)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=241,
        help="Number of frames per video, (8n + 1) format (default: 241)"
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=24.0,
        help="Frame rate (fps) (default: 24.0)"
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=35,
        help="Number of denoising steps (default: 35)"
    )
    parser.add_argument(
        "--cfg-guidance-scale",
        type=float,
        default=4.0,
        help="CFG guidance scale (default: 4.0)"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="text, watermark, logo, blurry, low quality, distorted, glitch, jitter",
        help="Negative prompt"
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
        "--skip-existing",
        action="store_true",
        help="Skip scenes that already have generated videos"
    )
    parser.add_argument(
        "--stitch-videos",
        action="store_true",
        help="Stitch all generated videos into a single output video using ffmpeg"
    )
    parser.add_argument(
        "--stitch-output",
        type=str,
        default=None,
        help="Output path for stitched video (default: {output_dir}/stitched_story.mp4)"
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
    
    logger.info("=" * 60)
    logger.info("BATCH GENERATION SETTINGS")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Distilled LoRA: {args.distilled_lora}")
    logger.info(f"Spatial Upsampler: {spatial_upsampler_path}")
    logger.info(f"Gemma Root: {gemma_root}")
    logger.info(f"FP8 Enabled: {args.enable_fp8}")
    logger.info(f"Resolution: {args.width}x{args.height}")
    logger.info(f"Frames: {args.num_frames}, FPS: {args.frame_rate}")
    logger.info(f"Steps: {args.num_inference_steps}, CFG: {args.cfg_guidance_scale}")
    logger.info(f"Output directory: {output_dir}")
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
                scene=scene,
                scene_num=i,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
                distilled_lora=args.distilled_lora,
                spatial_upsampler_path=spatial_upsampler_path,
                gemma_root=gemma_root,
                seed=args.seed,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                frame_rate=args.frame_rate,
                num_inference_steps=args.num_inference_steps,
                cfg_guidance_scale=args.cfg_guidance_scale,
                negative_prompt=args.negative_prompt,
                enable_fp8=args.enable_fp8,
                loras=args.lora if args.lora else None,
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
    
    # Stitch videos if requested
    if args.stitch_videos:
        logger.info("=" * 60)
        logger.info("STITCHING VIDEOS")
        logger.info("=" * 60)
        
        if args.stitch_output:
            stitch_output = resolve_path(args.stitch_output)
        else:
            stitch_output = os.path.join(output_dir, "stitched_story.mp4")
        
        try:
            stitched_path = stitch_videos_with_ffmpeg(generated_videos, stitch_output)
            logger.info(f"✅ Stitched video saved to: {stitched_path}")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"❌ Error stitching videos: {str(e)}")
            logger.error("Individual scene videos are still available in the output directory")


if __name__ == "__main__":
    main()
