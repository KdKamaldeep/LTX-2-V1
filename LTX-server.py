import os
import uuid
import json
import time
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# -----------------------------
# Config
# -----------------------------
MODEL_DIR = os.getenv("LTX2_MODEL_DIR", "/workspace/LTX-2/models/ltx2")
GEMMA_ROOT = os.getenv("LTX2_GEMMA_ROOT", "/workspace/LTX-2/models/gemma")
OUTPUT_DIR = os.getenv("LTX2_OUTPUT_DIR", "/workspace/ltx2_outputs")
CHECKPOINT_PATH = os.getenv("LTX2_CHECKPOINT_PATH", f"{MODEL_DIR}/ltx-2-19b-dev-fp8.safetensors")
SPATIAL_UPSAMPLER_PATH = os.getenv("LTX2_SPATIAL_UPSAMPLER_PATH", f"{MODEL_DIR}/ltx-2-spatial-upscaler-x2-1.0.safetensors")

# If your two-stage script requires distilled-lora, set this to the file path
DISTILLED_LORA_PATH = os.getenv("LTX2_DISTILLED_LORA_PATH", "")  # e.g. /workspace/LTX-2/models/ltx2/ltx-2-19b-distilled-lora-384.safetensors
DISTILLED_LORA_STRENGTH = float(os.getenv("LTX2_DISTILLED_LORA_STRENGTH", "0.8"))

# Mode: "cli" is guaranteed. "inproc" requires you to implement generate_inproc().
MODE = os.getenv("LTX2_MODE", "cli").lower()  # "cli" or "inproc"

# Perf stability
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Data Models
# -----------------------------
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="Text prompt for T2V")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    seed: Optional[int] = Field(None, description="Random seed")
    width: int = Field(704, ge=256, le=2048, description="Width divisible by 32 recommended")
    height: int = Field(1216, ge=256, le=2048, description="Height divisible by 32 recommended")
    num_frames: int = Field(121, ge=9, le=401, description="Frames: (8n + 1) recommended")
    frame_rate: int = Field(24, ge=1, le=60)
    steps: int = Field(30, ge=1, le=100)
    cfg: float = Field(5.0, ge=0.0, le=20.0)
    enhance_prompt: bool = Field(False)
    enable_fp8: bool = Field(True)

class GenerateResponse(BaseModel):
    job_id: str
    status: str
    output_path: Optional[str] = None
    error: Optional[str] = None

@dataclass
class Job:
    job_id: str
    status: str  # queued | running | done | error
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    request: Optional[Dict[str, Any]] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    log_path: Optional[str] = None

JOBS: Dict[str, Job] = {}

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="LTX-2 Text-to-Video Server",
    description="FastAPI server for LTX-2 T2V generation with Swagger at /docs",
    version="1.0.0",
)

# -----------------------------
# Helper functions
# -----------------------------
def _job_paths(job_id: str) -> Dict[str, str]:
    out_mp4 = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
    out_log = os.path.join(OUTPUT_DIR, f"{job_id}.log")
    out_json = os.path.join(OUTPUT_DIR, f"{job_id}.json")
    return {"mp4": out_mp4, "log": out_log, "json": out_json}

def _validate_paths():
    for p in [CHECKPOINT_PATH, SPATIAL_UPSAMPLER_PATH, GEMMA_ROOT]:
        if not os.path.exists(p):
            raise RuntimeError(f"Missing required path: {p}")

def generate_cli(req: GenerateRequest, out_path: str, log_path: str):
    """
    Guaranteed mode: call ltx_pipelines CLI.
    Note: this may reload models per request depending on your environment.
    """
    cmd = [
        "python", "-m", "ltx_pipelines.ti2vid_two_stages",
        "--checkpoint-path", CHECKPOINT_PATH,
        "--spatial-upsampler-path", SPATIAL_UPSAMPLER_PATH,
        "--gemma-root", GEMMA_ROOT,
        "--prompt", req.prompt,
        "--output-path", out_path,
        "--height", str(req.height),
        "--width", str(req.width),
        "--num-frames", str(req.num_frames),
        "--frame-rate", str(req.frame_rate),
        "--num-inference-steps", str(req.steps),
        "--cfg-guidance-scale", str(req.cfg),
    ]

    if req.seed is not None:
        cmd += ["--seed", str(req.seed)]
    if req.enhance_prompt:
        cmd += ["--enhance-prompt"]
    if req.enable_fp8:
        cmd += ["--enable-fp8"]
    if req.negative_prompt:
        cmd += ["--negative-prompt", req.negative_prompt]

    # Your CLI requires distilled-lora in your current version:
    if DISTILLED_LORA_PATH:
        cmd += ["--distilled-lora", DISTILLED_LORA_PATH, str(DISTILLED_LORA_STRENGTH)]
    else:
        # If your version requires it, it will error with "required: --distilled-lora".
        # You can set LTX2_DISTILLED_LORA_PATH env var to fix.
        pass

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("COMMAND:\n" + " ".join(cmd) + "\n\n")
        f.flush()

        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=os.environ.copy())
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ltx_pipelines exited with code {rc}. See log: {log_path}")

def generate_inproc(req: GenerateRequest, out_path: str, log_path: str):
    """
    Preferred mode: load models once and generate in-process.
    You MUST adapt this function to match the actual callable API in your installed ltx_pipelines.
    """
    raise NotImplementedError(
        "INPROC mode not wired. Set LTX2_MODE=cli or implement generate_inproc() "
        "using your ltx_pipelines callable pipeline."
    )

def run_job(job: Job):
    paths = _job_paths(job.job_id)
    job.status = "running"
    job.started_at = time.time()

    try:
        # Save request snapshot
        with open(paths["json"], "w", encoding="utf-8") as f:
            json.dump(job.request, f, indent=2)

        if MODE == "inproc":
            generate_inproc(GenerateRequest(**job.request), paths["mp4"], paths["log"])
        else:
            generate_cli(GenerateRequest(**job.request), paths["mp4"], paths["log"])

        job.status = "done"
        job.output_path = paths["mp4"]
        job.log_path = paths["log"]

    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.log_path = paths["log"]
    finally:
        job.finished_at = time.time()

# -----------------------------
# API endpoints
# -----------------------------
@app.on_event("startup")
def startup():
    _validate_paths()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": MODE,
        "checkpoint": CHECKPOINT_PATH,
        "upscaler": SPATIAL_UPSAMPLER_PATH,
        "gemma_root": GEMMA_ROOT,
        "has_distilled_lora": bool(DISTILLED_LORA_PATH),
        "output_dir": OUTPUT_DIR,
    }

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, background: BackgroundTasks):
    job_id = uuid.uuid4().hex
    job = Job(
        job_id=job_id,
        status="queued",
        created_at=time.time(),
        request=req.model_dump(),
    )
    JOBS[job_id] = job

    background.add_task(run_job, job)

    return GenerateResponse(job_id=job_id, status=job.status)

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return asdict(job)

@app.get("/download/{job_id}")
def download(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done" or not job.output_path or not os.path.exists(job.output_path):
        raise HTTPException(status_code=400, detail=f"Job not ready. status={job.status}")
    return FileResponse(job.output_path, media_type="video/mp4", filename=os.path.basename(job.output_path))

@app.get("/logs/{job_id}")
def logs(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.log_path or not os.path.exists(job.log_path):
        raise HTTPException(status_code=404, detail="Log not found yet")
    return FileResponse(job.log_path, media_type="text/plain", filename=os.path.basename(job.log_path))
