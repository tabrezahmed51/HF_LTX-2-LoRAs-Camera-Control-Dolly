print("[boot] importing modules...", flush=True)
import sys
import os
import gc
import threading
from pathlib import Path
import uuid
import tempfile
import time
import random
import numpy as np
import torch
import gradio as gr
import spaces
from typing import Iterable, Optional
from PIL import Image

# Gradio Theme Imports
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

# Add packages to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "packages" / "ltx-pipelines" / "src"))
sys.path.insert(0, str(current_dir / "packages" / "ltx-core" / "src"))

import flash_attn_interface
from huggingface_hub import hf_hub_download, snapshot_download

# LTX Imports
from ltx_pipelines.distilled import DistilledPipeline
import rife_ifnet
import esrgan_rrdbnet
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video_
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio_
from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
from ltx_pipelines.utils import ModelLedger  # noqa: F401 (re-exported for pipeline internals)
from ltx_pipelines.utils.helpers import generate_enhanced_prompt
from ltx_pipelines.utils.constants import (
    DEFAULT_SEED,
    DEFAULT_1_STAGE_HEIGHT,
    DEFAULT_1_STAGE_WIDTH,
    DEFAULT_NUM_FRAMES,
    DEFAULT_FRAME_RATE,
    DEFAULT_LORA_STRENGTH,
)

# -----------------------------------------------------------------------------
# Attention backend fix for ZeroGPU A10G (Ampere, sm86):
# FlashAttention-3 kernels are Hopper-only (sm90); running them here fails with
# "CUDA error ... no kernel image is available for execution on the device".
# Route ALL attention through PyTorch SDPA, which ships efficient Ampere kernels.
# -----------------------------------------------------------------------------
from ltx_core.model.transformer.attention import AttentionFunction, PytorchAttention

_pytorch_attention = PytorchAttention()

def _sdpa_attention_call(self, q, k, v, heads, mask=None):
    return _pytorch_attention(q, k, v, heads, mask)

AttentionFunction.__call__ = _sdpa_attention_call
print("Attention backend: PyTorch SDPA (FlashAttention3 unavailable on Ampere)")

# -----------------------------------------------------------------------------
# 1. Theme — "LTX Studio" (ui-ux-pro-max design system: OLED dark,
#    AI-violet primary #7C3AED + generation-pink accent #EC4899, Inter type)
# -----------------------------------------------------------------------------
from gradio.themes import Base
from gradio.themes.utils import sizes as _sizes

class StudioTheme(Base):
    def __init__(
        self,
        *,
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
        font=(fonts.GoogleFont("Inter"), "system-ui", "sans-serif"),
        font_mono=(fonts.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"),
        **kwargs,
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            font=font,
            font_mono=font_mono,
            **kwargs,
        )
        desired = {
            "body_background_fill": "#0B0F1A",
            "body_background_fill_dark": "#0B0F1A",
            "block_background_fill": "#111827",
            "block_background_fill_dark": "#111827",
            "block_border_width": "1px",
            "block_border_color": "#1E293B",
            "block_radius": "16px",
            "block_title_text_color": "#C7D2FE",
            "block_label_background_fill": "#131C31",
            "block_label_background_fill_dark": "#131C31",
            "block_label_text_color": "#A5B4FC",
            "border_color_primary": "#1E293B",
            "background_fill_primary": "#0B0F1A",
            "background_fill_secondary": "#111827",
            "background_fill_secondary_dark": "#111827",
            "input_background_fill": "#0F172A",
            "input_background_fill_dark": "#0F172A",
            "input_border_color": "#1E293B",
            "button_primary_background_fill": "linear-gradient(135deg,#7C3AED,#EC4899)",
            "button_primary_background_fill_hover": "linear-gradient(135deg,#8B5CF6,#F472B6)",
            "button_primary_background_fill_dark": "linear-gradient(135deg,#7C3AED,#EC4899)",
            "button_primary_text_color": "#FFFFFF",
            "button_primary_border_color": "transparent",
            "button_secondary_background_fill": "#1E293B",
            "button_secondary_background_fill_dark": "#1E293B",
            "button_secondary_text_color": "#E2E8F0",
            "button_secondary_text_color_dark": "#E2E8F0",
            "slider_color": "#7C3AED",
            "color_accent_soft": "#7C3AED",
            "text_color": "#E2E8F0",
            "text_color_dark": "#E2E8F0",
            "text_color_subdued": "#94A3B8",
            "text_color_subdued_dark": "#94A3B8",
        }
        for _k, _v in desired.items():   # never let one bad key kill boot
            try:
                super().set(**{_k: _v})
            except Exception as _ke:
                print(f"[theme] skip {_k}: {_ke}", flush=True)

try:
    studio_theme = StudioTheme()
    print("[boot] StudioTheme constructed", flush=True)
except Exception as _te:
    import traceback as _tb
    print("[boot] StudioTheme FAILED:", repr(_te), flush=True)
    _tb.print_exc()
    from gradio.themes import Soft as _Soft
    studio_theme = _Soft(primary_hue="violet")

# -----------------------------------------------------------------------------
# 2. Configuration & Adapters
# -----------------------------------------------------------------------------

MAX_SEED = np.iinfo(np.int32).max

# HuggingFace Hub defaults
DEFAULT_REPO_ID = "Lightricks/LTX-2"
DEFAULT_GEMMA_REPO_ID = "unsloth/gemma-3-12b-it-qat-bnb-4bit"
DEFAULT_CHECKPOINT_FILENAME = "ltx-2-19b-dev.safetensors"
DEFAULT_DISTILLED_LORA_FILENAME = "ltx-2-19b-distilled-lora-384.safetensors"
DEFAULT_SPATIAL_UPSAMPLER_FILENAME = "ltx-2-spatial-upscaler-x2-1.0.safetensors"

# New Adapter Definitions
ADAPTER_SPECS = {
    "Camera-Control-Dolly-Left": {
        "repo": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left",
        "weights": "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
        "adapter_name": "camera-control-dolly-left"
    },
    "Camera-Control-Dolly-Right": {
        "repo": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right",
        "weights": "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
        "adapter_name": "camera-control-dolly-right"
    },
    "Camera-Control-Dolly-In": {
        "repo": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In",
        "weights": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
        "adapter_name": "camera-control-dolly-in"
    },
    "Camera-Control-Dolly-Out": {
        "repo": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out",
        "weights": "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
        "adapter_name": "camera-control-dolly-out"
    },

# ---------------------------------------others ---------------------------------------
    
     "Camera-Control-Jib-Down": {
        "repo": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down",
        "weights": "ltx-2-19b-lora-camera-control-jib-down.safetensors",
        "adapter_name": "camera-control-jib-down"
    },
     "Camera-Control-Jib-Up": {
        "repo": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up",
        "weights": "ltx-2-19b-lora-camera-control-jib-up.safetensors",
        "adapter_name": "camera-control-jib-up"
    },
     "Camera-Control-Static": {
        "repo": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static",
        "weights": "ltx-2-19b-lora-camera-control-static.safetensors",
        "adapter_name": "camera-control-static"
    },
}

# -----------------------------------------------------------------------------
# 2b. Skill Packs — official Lightricks LTX-2-19B IC-LoRA stack (free upgrade layer)
#     Experimental without reference-guide wiring; strength kept conservative.
# -----------------------------------------------------------------------------
STYLE_PACKS = {
    "None": None,
    "Pose Control": {
        "repo": "Lightricks/LTX-2-19b-IC-LoRA-Pose-Control",
        "weights": "ltx-2-19b-ic-lora-pose-control.safetensors",
        "strength": 0.9,
    },
    "Union Control": {
        "repo": "Lightricks/LTX-2-19b-IC-LoRA-Union-Control",
        "weights": "ltx-2-19b-ic-lora-union-control-ref0.5.safetensors",
        "strength": 0.9,
    },
    "Detailer": {
        "repo": "Lightricks/LTX-2-19b-IC-LoRA-Detailer",
        "weights": "ltx-2-19b-ic-lora-detailer.safetensors",
        "strength": 0.9,
    },
}

# ---------------- Free vault: private HF Dataset as storage ------------------
import json as _json
from huggingface_hub import HfApi

VAULT_REPO = os.environ.get("VAULT_REPO", f"{os.environ.get('SPACE_AUTHOR','tabrezahmed51')}/ltx2-space-vault")
_hf_api = HfApi(token=os.environ.get("HF_TOKEN"))

def _vault_ok():
    tok = os.environ.get("HF_TOKEN")
    return bool(tok)

def vault_ensure():
    if not _vault_ok():
        return False
    try:
        _hf_api.create_repo(VAULT_REPO, repo_type="dataset", private=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"[vault] ensure failed: {e}")
        return False

def vault_upload_render(mp4_path: str, meta: dict):
    """Save rendered mp4 + metadata line into the private dataset (best-effort)."""
    if not _vault_ok():
        return None
    try:
        vault_ensure()
        import datetime as _dt
        day = _dt.datetime.utcnow().strftime("%Y-%m-%d")
        uid = uuid.uuid4().hex[:10]
        path_in_repo = f"runs/{day}/{uid}.mp4"
        _hf_api.upload_file(path_or_fileobj=mp4_path, path_in_repo=path_in_repo,
                            repo_id=VAULT_REPO, repo_type="dataset")
        url = f"https://huggingface.co/datasets/{VAULT_REPO}/resolve/main/{path_in_repo}"
        meta = {**meta, "url": url, "ts": _dt.datetime.utcnow().isoformat(timespec="seconds")}
        line = _json.dumps(meta, ensure_ascii=False)

        _VAULT_WRITE_LOCK.acquire()
        try:
            try:
                from huggingface_hub import hf_hub_download as _dl
                old = _dl(VAULT_REPO, "runs/metadata.jsonl", repo_type="dataset")
                content = open(old, "r", encoding="utf-8").read()
            except Exception:
                content = ""
            import tempfile as _tf
            tmp = _tf.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
            tmp.write((content + line + "\n")[-200000:])
            tmp.close()
            _hf_api.upload_file(path_or_fileobj=tmp.name, path_in_repo="runs/metadata.jsonl",
                                repo_id=VAULT_REPO, repo_type="dataset")
            os.unlink(tmp.name)
            print(f"[vault] saved {path_in_repo}")
        finally:
            _VAULT_WRITE_LOCK.release()
        return url
    except Exception as e:
        print(f"[vault] upload failed (non-fatal): {e}")
        return None

def vault_recent(n: int = 12):
    """Return list of recent run dicts (newest first)."""
    if not _vault_ok():
        return []
    try:
        from huggingface_hub import hf_hub_download
        f = hf_hub_download(VAULT_REPO, "runs/metadata.jsonl", repo_type="dataset")
        rows = [ _json.loads(l) for l in open(f, encoding='utf-8') if l.strip() ]
        return list(reversed(rows[-n:]))
    except Exception as e:
        print(f"[vault] read failed: {e}")
        return []

def vault_save_preset(name: str, values: dict):
    if not _vault_ok():
        return False, "HF_TOKEN secret missing"
    try:
        vault_ensure()
        from huggingface_hub import hf_hub_download
        try:
            f = hf_hub_download(VAULT_REPO, "presets.json", repo_type="dataset")
            data = _json.load(open(f, encoding="utf-8"))
        except Exception:
            data = {}
        data[name] = values
        import tempfile as _tf
        tmp = _tf.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        _json.dump(data, tmp.open("w", encoding="utf-8")); tmp.close()
        # rewrite properly (tmp handle quirk-safe)
        _hf_api.upload_file(path_or_fileobj=tmp.name, path_in_repo="presets.json",
                            repo_id=VAULT_REPO, repo_type="dataset")
        os.unlink(tmp.name)
        return True, f"saved '{name}'"
    except Exception as e:
        return False, str(e)

def vault_list_presets():
    if not _vault_ok():
        return []
    try:
        from huggingface_hub import hf_hub_download
        f = hf_hub_download(VAULT_REPO, "presets.json", repo_type="dataset")
        return list(_json.load(open(f, encoding="utf-8")).keys())
    except Exception:
        return []

def vault_get_preset(name: str):
    try:
        from huggingface_hub import hf_hub_download
        f = hf_hub_download(VAULT_REPO, "presets.json", repo_type="dataset")
        return _json.load(open(f, encoding="utf-8")).get(name)
    except Exception:
        return None

# -----------------------------------------------------------------------------
# 3. Identity Preservation Prompt Engineering
#
# The input image conditions frame 0 directly (latent conditioning @ strength
# 1.0), so the face is pixel-anchored at the start. Identity drift happens when
# TEXT contradicts the image - mainly because prompt enhancement invents new
# facial attributes ("a man in his late 40s, graying hair..."). These blocks:
#   1) constrain Gemma to describe ONLY motion/camera/ambience and refer to
#      people generically ("the person", never re-describing their face);
#   2) append a strict identity-lock directive to every prompt reaching the
#      video model, forbidding any facial change across frames.
# -----------------------------------------------------------------------------

# Injected INTO the user prompt before Gemma enhancement, so the rewritten
# prompt inherits the constraints instead of hallucinating appearance details.
ENHANCEMENT_IDENTITY_ANCHOR = (
    "IMPORTANT IDENTITY RULE: This is an animation of the provided image. Refer to every person "
    "only as 'the person', 'the woman/man' or 'the subject' - NEVER describe, list, invent or alter "
    "any facial features, face shape, eyes, nose, lips, jawline, skin tone, texture, age, ethnicity, "
    "hairstyle, facial hair, makeup, glasses, body type, clothing, accessories, moles, scars or tattoos. "
    "Their appearance already exists in the image and must remain exactly as-is. Describe ONLY: camera "
    "movement, body and head motion, gesture, gaze direction, breathing and blinking, hair and fabric "
    "motion caused by movement, environmental motion, lighting dynamics, depth and atmosphere."
)

# Appended to the FINAL prompt (enhanced or raw) sent to the video model.
IDENTITY_LOCK_SUFFIX = (
    " IDENTITY LOCK (highest priority): The person(s) captured in the source image are the single "
    "ground-truth reference for this shot. Render their exact same face in every single frame with zero "
    "deviation: identical bone structure and face shape, eye color and eyelid shape, eyebrow shape, nose "
    "shape, lip shape and volume, jawline and chin, ears, skin tone, skin texture and pores, apparent age, "
    "ethnicity, complexion, hairstyle, hairline and hair color, facial hair, makeup, glasses or jewelry, "
    "and every mole, freckle, scar or tattoo exactly where they are in the image. Preserve wardrobe, "
    "colors, proportions and accessories unchanged. Animate only natural, physically plausible micro-motion "
    "(blinks, breathing, subtle muscle shifts, head and eye tracking) coherent with the described camera move. "
    "Strictly forbidden: face morphing or warping over time, identity drift between frames, beautification "
    "or skin smoothing, changing age, gender, ethnicity or weight, restyling hair or clothes, generating a "
    "lookalike or different actor, adding, removing or swapping people, extra limbs, distorted features, "
    "uncanny or plastic skin. First frame must be visually identical to the input image."
)

def _build_motion_prompt(prompt: str, has_image: bool) -> str:
    """Anchor the user prompt for safe Gemma enhancement."""
    if has_image:
        return f"{prompt}\n\n{ENHANCEMENT_IDENTITY_ANCHOR}"
    return prompt

def _finalize_prompt(prompt: str) -> str:
    """Append the identity-lock directive to the prompt used for generation."""
    return f"{prompt.rstrip()}{IDENTITY_LOCK_SUFFIX}"

# -----------------------------------------------------------------------------
# 4. Startup (CPU-safe downloads only) + Lazy CUDA Model Loading
#
# ZeroGPU containers boot WITHOUT a GPU: any .to("cuda") / CUDA tensor op at
# module level raises "RuntimeError: No CUDA GPUs are available". So we only
# download weights here (plain file I/O), and defer every CUDA load to the
# first @spaces.GPU call.
# -----------------------------------------------------------------------------

print("=" * 80, flush=True)
print("Downloading LTX-2 checkpoints (CPU-safe, no GPU required)...", flush=True)
print("=" * 80, flush=True)

_checkpoint_path = hf_hub_download(repo_id=DEFAULT_REPO_ID, filename=DEFAULT_CHECKPOINT_FILENAME)
print(f"Checkpoint ready: {_checkpoint_path}")

snapshot_download(repo_id=DEFAULT_GEMMA_REPO_ID)
print(f"Gemma model cached: {DEFAULT_GEMMA_REPO_ID}")

_distilled_lora_path = hf_hub_download(repo_id=DEFAULT_REPO_ID, filename=DEFAULT_DISTILLED_LORA_FILENAME)
print(f"Distilled LoRA ready: {_distilled_lora_path}")

_spatial_upsampler_path = hf_hub_download(repo_id=DEFAULT_REPO_ID, filename=DEFAULT_SPATIAL_UPSAMPLER_FILENAME)
print(f"Spatial upsampler ready: {_spatial_upsampler_path}")

# NOTE: full adapter/pack prefetch (~11GB) exceeded ephemeral disk on
# rescheduled pods and crashed boots. They now download on first use inside
# the GPU call (small cost), while the tiny RIFE/ESRGAN/TTS assets below stay
# prefetched (<600MB combined).

# --- post-processing assets (CPU phase, quota-free) ---
try:
    _flownet_path = hf_hub_download(repo_id="MonsterMMORPG/Practical-RIFE",
                                    filename="train_log/flownet.pkl")
    # strict pairing check at boot — feature self-disables on mismatch
    rife_ifnet.load_rife(_flownet_path, device="cpu")
    print(f"RIFE weights ready + verified: {_flownet_path}")
except Exception as _e:
    _flownet_path = None
    print(f"[warn] RIFE prefetch/verify failed (feature disabled): {_e}")

try:
    _esrgan_path = hf_hub_download(repo_id="ai-forever/Real-ESRGAN",
                                   filename="RealESRGAN_x2.pth")
    esrgan_rrdbnet.load_esrgan(_esrgan_path, device="cpu")  # strict pairing check
    print(f"ESRGAN x2 weights ready: {_esrgan_path}")
except Exception as _e:
    _esrgan_path = None
    print(f"[warn] ESRGAN prefetch failed (feature disabled): {_e}")

try:
    hf_hub_download(repo_id="facebook/mms-tts-eng", filename="config.json")
    print("mms-tts-eng reachable")
except Exception as _e:
    print(f"[warn] mms-tts probe failed: {_e}")

print("=" * 80)
print("All checkpoints downloaded. CUDA models will lazy-load on first GPU call.")
print("=" * 80)

_TEMP_FILES = []   # fitted-image scratch files, swept after each run
_TEMP_LOCK = threading.Lock()
_VAULT_WRITE_LOCK = threading.Lock()

def _sweep_temp(keep=None):
    with _TEMP_LOCK:
        for f in list(_TEMP_FILES):
            if f != keep and os.path.exists(f):
                try: os.unlink(f)
                except OSError: pass
                _TEMP_FILES.remove(f)

# -----------------------------------------------------------------------------
# 5. Inference Logic
#
# VRAM strategy for ZeroGPU (24GB A10G):
#   - Build DistilledPipeline with fp8transformer=True (transformer ~20GB fp8
#     instead of ~40GB bf16 - bf16 cannot fit and fails with NVML asserts).
#   - Use the pipeline's OWN ModelLedger (no separate/global copies, no double
#     loading). Prompt enhancement + encoding use pipeline.model_ledger's text
#     encoder, then free it BEFORE the transformer loads for diffusion.
# -----------------------------------------------------------------------------

def encode_text_simple(text_encoder, prompt: str):
    """Simple text encoding without using pipeline_utils."""
    v_context, a_context, _ = text_encoder(prompt)
    return v_context, a_context

# -----------------------------------------------------------------------------
# Aspect-ratio protection: the pipeline's load_image_conditioning() uses
# cover + CENTER-CROP, which decapitates portrait photos when generation runs
# at a mismatched aspect (e.g. 3:4 person photo -> 768x512 landscape).
# We pre-fit the image to the target aspect ourselves: scale to FIT and fill
# the remainder with a blurred, darkened version of the photo (no hard crop,
# no black bars, full body/head always preserved).
# -----------------------------------------------------------------------------
from PIL import Image as PILImage, ImageFilter

def _make_ref_video(image_path: str, width: int, height: int, seconds: float,
                    num_frames: int | None = None) -> str:
    """Build a still reference video for IC-LoRA guide conditioning.
    Frame count must satisfy the LTX latent rule: 1 + 8*x."""
    out = tempfile.NamedTemporaryFile(suffix="_ref.mp4", delete=False)
    out.close()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", image_path,
           "-frames:v", str(max(9, int(num_frames or 9))), "-r", "24",
           "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}",
           "-pix_fmt", "yuv420p", out.name]
    rc = subprocess.run(cmd, capture_output=True).returncode if False else None
    import subprocess as _sp
    if _sp.run(cmd, capture_output=True).returncode != 0 or not os.path.exists(out.name) \
            or os.path.getsize(out.name) == 0:
        raise RuntimeError("reference video build failed")
    with _TEMP_LOCK:
        _TEMP_FILES.append(out.name)
    return out.name

def _fit_image_to_aspect(path: str, target_w: int, target_h: int, threshold: float = 0.06):
    try:
        img = PILImage.open(path).convert("RGB")
    except Exception:
        return path
    src_ar = img.width / float(img.height)
    tgt_ar = target_w / float(target_h)
    if abs(src_ar - tgt_ar) / tgt_ar <= threshold:
        return path  # already compatible - leave untouched

    scale = min(target_w / img.width, target_h / img.height)
    fitted = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), PILImage.LANCZOS)

    bg = img.resize((target_w, target_h), PILImage.LANCZOS).filter(ImageFilter.GaussianBlur(28))
    bg = PILImage.blend(bg, PILImage.new("RGB", (target_w, target_h), (8, 8, 12)), 0.35)

    canvas = bg
    canvas.paste(fitted, ((target_w - fitted.width) // 2, (target_h - fitted.height) // 2))
    out = tempfile.NamedTemporaryFile(suffix="_fitted.jpg", delete=False)
    canvas.save(out, format="JPEG", quality=95)
    out.close()
    _TEMP_FILES.append(out.name)
    print(f"Aspect-fit: {img.width}x{img.height} -> padded canvas {target_w}x{target_h} (no crop)")
    return out.name

# Function to calculate timeout based on the arguments passed to the GPU decorated function
def calc_timeout_duration(*args, **kwargs):
    # ZeroGPU passes every generate_video argument; gpu_timeout sits at index 9
    try:
        return int(args[9])
    except Exception:
        return 300

@spaces.GPU(duration=calc_timeout_duration)
def generate_video(
    input_image,
    prompt: str,
    lora_adapter: str,
    duration: float,
    enhance_prompt: bool = True,
    seed: int = 42,
    randomize_seed: bool = True,
    height: int = DEFAULT_1_STAGE_HEIGHT,
    width: int = DEFAULT_1_STAGE_WIDTH,
    gpu_timeout: int = 300,
    anchor: float = 1.0,
    lora_str: float = 0.8,
    style_pack: str = "None",
    chk_rife: bool = False,
    chk_upscale: bool = False,
    vo_text: str = "",
    vo_lang: str = "English (US)",
    vo_mix: str = "Replace native audio",
    progress=gr.Progress(track_tqdm=True),
):
    gc.collect()
    torch.cuda.empty_cache()

    # defensive coercion (UI labels must never leak into numeric pipeline params)
    def _f(v, dflt):
        try:
            return float(v)
        except (TypeError, ValueError):
            return dflt
    anchor = _f(anchor, 1.0)
    lora_str = _f(lora_str, 0.8)
    duration = _f(duration, 3.0)

    try:
        current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
        frame_rate = 24.0
        num_frames = int(duration * frame_rate) + 1

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            output_path = tmpfile.name

        # Prepare Inputs
        images = []
        if input_image is not None:
            # Protect against center-crop: fit image to target aspect first
            input_image = _fit_image_to_aspect(input_image, int(width), int(height))
            images = [(input_image, 0, 1.0)]

        # ---------------------------
        # Configure LoRAs (downloads happen BEFORE heavy models load)
        # ---------------------------
        active_loras = [
            LoraPathStrengthAndSDOps(
                path=_distilled_lora_path,
                strength=DEFAULT_LORA_STRENGTH,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ]

        if lora_adapter and lora_adapter != "None":
            spec = ADAPTER_SPECS.get(lora_adapter)
            if spec:
                print(f"Loading Adapter: {lora_adapter}")
                adapter_path = hf_hub_download(repo_id=spec["repo"], filename=spec["weights"])
                active_loras.append(
                    LoraPathStrengthAndSDOps(
                        path=adapter_path,
                        strength=float(lora_str),  # guide: 0.9-1.1 subtle .. 1.2+ strong
                        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                    )
                )

        # Skill Pack (official Lightricks IC-LoRA stack) — combined strength guard < 2.0
        pack_used = "None"
        if style_pack and style_pack != "None":
            pack = STYLE_PACKS.get(style_pack)
            camera_active = bool(active_loras) and len(active_loras) > 1  # distilled + camera
            total = float(lora_str if camera_active else 0.0) + float(pack["strength"])
            if pack and total < 2.0:
                print(f"Stacking Skill Pack: {style_pack}")
                pack_path = hf_hub_download(repo_id=pack["repo"], filename=pack["weights"])
                active_loras.append(LoraPathStrengthAndSDOps(
                    path=pack_path, strength=float(pack["strength"]),
                    sd_ops=LTXV_LORA_COMFY_RENAMING_MAP))
                pack_used = style_pack
            elif pack:
                print(f"Skill Pack skipped: combined strength {total:.2f} >= 2.0")

        job_adapter_note = f"{lora_adapter}{('+' + pack_used) if pack_used != 'None' else ''}"

        # =====================================================================
        # Post-processing chain: RIFE → Swin2SR → Voiceover  (each optional,
        # lazy-loaded on GPU, self-releasing; failures degrade gracefully)
        # =====================================================================
        import subprocess as ff
        def _mux_audio(video_no_audio: str, original_video: str) -> str:
            """Copy original audio track back onto a re-encoded silent video."""
            out = tempfile.NamedTemporaryFile(suffix="_aud.mp4", delete=False)
            out.close()
            probe = ff.run(["ffprobe", "-v", "error", "-select_streams", "a",
                            "-show_entries", "stream=codec_type",
                            "-of", "csv=p=0", original_video],
                           capture_output=True)
            if b"audio" not in probe.stdout:
                return original_video
            ff.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_no_audio,
                    "-i", original_video, "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:v", "copy", "-c:a", "aac", "-shortest", out.name],
                   capture_output=True)
            return out.name if os.path.exists(out.name) and os.path.getsize(out.name) else original_video

        post_flags = {"rife": bool(chk_rife), "upscale": bool(chk_upscale),
                      "vo": bool(vo_text and vo_text.strip())}
        print(f"[post] flags: {post_flags}", flush=True)

        def _post_rife(path_in: str) -> str:
            if not _flownet_path:
                print("[post] RIFE skipped: weights unavailable", flush=True); return path_in
            import av as _av
            try:
                rife = rife_ifnet.load_rife(_flownet_path, device="cuda")
            except RuntimeError as le:
                raise RuntimeError(f"RIFE weight pairing failed: {le}") from None
                container = _av.open(path_in)
                stream = container.streams.video[0]
                fps = float(stream.average_rate or 24)
                frames = []
                for fr in container.decode(stream):
                    arr = fr.to_ndarray(format="rgb24")
                    t = torch.from_numpy(arr).to("cuda", torch.float32) / 255.0
                    frames.append(t.permute(2, 0, 1))
                container.close()
                out_frames = []
                for i in range(len(frames)):
                    out_frames.append(frames[i])
                    if i + 1 < len(frames):
                        mid = rife.midpoint(frames[i][None], frames[i + 1][None])[0]
                        out_frames.append(mid)
                del rife; gc.collect(); torch.cuda.empty_cache()

                silent = tempfile.NamedTemporaryFile(suffix="_rife.mp4", delete=False)
                silent.close()
                oc = _av.open(silent.name, "w")
                vstream = oc.add_stream("libx264", rate=int(round(fps * 2)))
                vstream.width, vstream.height = out_frames[0].shape[2], out_frames[0].shape[1]
                vstream.pix_fmt = "yuv420p"
                for t in out_frames:
                    arr = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                    packet = vstream.encode(_av.VideoFrame.from_ndarray(arr, format="rgb24"))
                    if packet: oc.mux(packet)
                while True:
                    packet = vstream.encode()
                    if packet: oc.mux(packet)
                    else: break
                oc.close()
                final = _mux_audio(silent.name, path_in)
                if final != silent.name and os.path.exists(silent.name):
                    os.unlink(silent.name)
                del out_frames; gc.collect(); torch.cuda.empty_cache()
                print(f"[post] RIFE done -> {len(out_frames)} frames @ {fps*2:.0f}fps")
                return final
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("[post] RIFE hit OOM - skipping gracefully")
                    gc.collect(); torch.cuda.empty_cache(); return path_in
                raise

        def _post_upscale(path_in: str) -> str:
            print(f"[dbg] UP enter esrgan={_esrgan_path!r}", flush=True)
            if not _esrgan_path:
                print("[post] upscale skipped: weights unavailable"); return path_in
            try:
                import av as _av
                net = esrgan_rrdbnet.load_esrgan(_esrgan_path, device="cuda")
                container = _av.open(path_in)
                stream = container.streams.video[0]
                fps = float(stream.average_rate or 24)
                frames = [fr.to_ndarray(format="rgb24") for fr in container.decode(stream)]
                container.close()
                outs = []
                for i, arr in enumerate(frames):
                    t = torch.from_numpy(arr).to("cuda", torch.float32) / 255.0
                    res = esrgan_rrdbnet.upscale_tensor(net, t.permute(2, 0, 1)[None])[0]
                    outs.append(res.permute(1, 2, 0).clamp(0, 1).cpu().numpy())
                    if i % 24 == 0:
                        gc.collect(); torch.cuda.empty_cache()
                        print(f"[post] upscale {i+1}/{len(frames)}")
                del net; gc.collect(); torch.cuda.empty_cache()
                silent = tempfile.NamedTemporaryFile(suffix="_up.mp4", delete=False); silent.close()
                oc = _av.open(silent.name, "w")
                vs = oc.add_stream("libx264", rate=int(round(fps)))
                vs.width, vs.height = outs[0].shape[1], outs[0].shape[0]
                vs.pix_fmt = "yuv420p"
                for arr in outs:
                    pk = vs.encode(_av.VideoFrame.from_ndarray(arr, format="rgb24"))
                    if pk: oc.mux(pk)
                while True:
                    pk = vs.encode()
                    if pk: oc.mux(pk)
                    else: break
                oc.close()
                final = _mux_audio(silent.name, path_in)
                if final != silent.name and os.path.exists(silent.name):
                    os.unlink(silent.name)
                print(f"[post] upscaled to {outs[0].shape[1]}x{outs[0].shape[0]} (Real-ESRGAN x2)")
                return final
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("[post] upscale OOM - skipping gracefully")
                    gc.collect(); torch.cuda.empty_cache(); return path_in
                raise

        def _post_voiceover(path_in: str):
            if not vo_text.strip():
                return path_in
            try:
                from transformers import VitsModel, AutoTokenizer
                lang_map = {
                    "English (US)": "facebook/mms-tts-eng", "Hindi": "facebook/mms-tts-hin",
                    "Tamil": "facebook/mms-tts-tam", "Telugu": "facebook/mms-tts-tel",
                    "Bengali": "facebook/mms-tts-ben", "Marathi": "facebook/mms-tts-mar",
                    "Gujarati": "facebook/mms-tts-guj", "Kannada": "facebook/mms-tts-kan",
                    "Malayalam": "facebook/mms-tts-mal", "Urdu": "facebook/mms-tts-urd",
                    "Punjabi": "facebook/mms-tts-pan",
                }
                repo = lang_map.get(vo_lang, "facebook/mms-tts-eng")
                tok = AutoTokenizer.from_pretrained(repo)
                model = VitsModel.from_pretrained(repo).to("cuda").eval()
                inputs = tok(vo_text[:600], return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    wave = model(**inputs).waveform.squeeze().float().cpu().numpy()
                sr = model.config.sampling_rate
                del model; gc.collect(); torch.cuda.empty_cache()

                import scipy.io.wavfile as _wav
                wavf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); wavf.close()
                _wav.write(wavf.name, rate=sr, data=(wave * 32767).astype("int16"))

                out = tempfile.NamedTemporaryFile(suffix="_vo.mp4", delete=False); out.close()
                if vo_mix == "Replace native audio":
                    ff.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path_in,
                            "-i", wavf.name, "-map", "0:v:0", "-map", "1:a:0",
                            "-c:v", "copy", "-c:a", "aac", "-shortest", out.name],
                           capture_output=True)
                else:  # Mix over ducked native
                    ff.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path_in,
                            "-i", wavf.name, "-filter_complex",
                            "[1:a]volume=1.0[v];[0:a]volume=0.35[n];[n][v]amix=inputs=2:duration=first[a]",
                            "-map", "0:v:0", "-map", "[a]", "-c:v", "copy",
                            "-c:a", "aac", "-shortest", out.name], capture_output=True)
                os.unlink(wavf.name)
                ok = os.path.exists(out.name) and os.path.getsize(out.name) > 0
                print(f"[post] voiceover ({vo_lang}, {vo_mix}) {'ok' if ok else 'failed'}")
                return out.name if ok else path_in
            except Exception as e:
                print(f"[post] voiceover failed (non-fatal): {e}")
                return path_in

        def _finalize(output_path: str) -> tuple[str, int, str]:
            meta_post = []
            pth = output_path
            try:
                if post_flags["rife"]:
                    pth = _post_rife(pth); meta_post.append("rife")
            except Exception as e:
                print(f"[post] rife failed non-fatal: {e}", flush=True)
            try:
                if post_flags["upscale"]:
                    pth = _post_upscale(pth); meta_post.append("upscale")
            except Exception as e:
                print(f"[post] upscale failed non-fatal: {e}", flush=True)
            try:
                if post_flags["vo"]:
                    pth = _post_voiceover(pth); meta_post.append(f"vo:{vo_lang}")
            except Exception as e:
                print(f"[post] voiceover outer failure non-fatal: {e}", flush=True)
            vault_url = vault_upload_render(pth, {
                "prompt": final_prompt[:400], "seed": current_seed,
                "adapter": job_adapter_note, "duration": duration,
                "width": width, "height": height, "anchor": anchor,
                "pack": pack_used, "post": ",".join(meta_post),
            }) or ""
            return pth, current_seed, vault_url
        # ================= TRUE IC-GUIDE PATH (Detailer) =================
        # Official IC-LoRA usage: source image becomes a half-res reference
        # video guide; identity/fidelity flows through the guide, not text.
        if pack_used == "Detailer" and images:
            import subprocess as _sp
            from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE as _ASR
            from ltx_pipelines.utils.media_io import encode_video as _encode_video

            print("IC-guide mode: building reference video from source image…")
            ref_video = _make_ref_video(images[0][0], int(width), int(height),
                                        float(duration), num_frames=num_frames)

            ic_pipeline = ICLoraPipeline(
                checkpoint_path=_checkpoint_path,
                spatial_upsampler_path=_spatial_upsampler_path,
                gemma_root=DEFAULT_GEMMA_REPO_ID,
                loras=active_loras,
                device=torch.device("cuda"),
                fp8transformer=True,
            )
            tcfg = TilingConfig.default()

            text_encoder_ic = ic_pipeline.stage_1_model_ledger.text_encoder()
            final_prompt_local = prompt
            if enhance_prompt:
                final_prompt_local = generate_enhanced_prompt(
                    text_encoder=text_encoder_ic,
                    prompt=_build_motion_prompt(prompt, True),
                    image_path=images[0][0], seed=current_seed)
                print(f"Enhanced prompt: {final_prompt_local}")
                del text_encoder_ic
                gc.collect(); torch.cuda.empty_cache()

            # pipeline encodes text internally and frees its encoder before
            # the transformer loads (its built-in VRAM-safe sequence)
            video_t, audio_t = ic_pipeline(
                prompt=_finalize_prompt(final_prompt_local),
                seed=current_seed,
                height=int(height),
                width=int(width),
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=[],                                  # identity flows via guide
                video_conditioning=[(ref_video, 0.9)],
                enhance_prompt=False,                       # we handled enhancement
                tiling_config=tcfg,
            )
            chunks = get_video_chunks_number(num_frames, tcfg)
            with torch.inference_mode():   # tensors were born in inference mode
                _encode_video(video=video_t, fps=frame_rate, audio=audio_t,
                              audio_sample_rate=_ASR, output_path=str(output_path),
                              video_chunks_number=chunks)
            del ic_pipeline
            gc.collect(); torch.cuda.empty_cache()
            _sweep_temp()
            return _finalize(str(output_path))
        # ================= end IC-guide path =================

        # ---------------------------
        # Instantiate Pipeline (fp8 transformer fits 24GB ZeroGPU; bf16 does not)
        # Uses its own internal ModelLedger - no separate/duplicate model copies.
        # ---------------------------
        print("Building DistilledPipeline (fp8 transformer)...")
        pipeline = DistilledPipeline(
            device=torch.device("cuda"),
            checkpoint_path=_checkpoint_path,
            spatial_upsampler_path=_spatial_upsampler_path,
            gemma_root=DEFAULT_GEMMA_REPO_ID,
            loras=active_loras,
            fp8transformer=True,
            local_files_only=True,  # everything pre-downloaded at startup
        )

        # ---------------------------
        # Prompt enhancement + encoding via the pipeline's own text encoder.
        # Encoder is freed BEFORE the transformer loads for diffusion.
        # ---------------------------
        video_context = None
        audio_context = None
        final_prompt = prompt

        if enhance_prompt:
            print("Loading Gemma text encoder for prompt enhancement...")
            text_encoder = pipeline.model_ledger.text_encoder()
            final_prompt = generate_enhanced_prompt(
                text_encoder=text_encoder,
                prompt=_build_motion_prompt(prompt, input_image is not None),
                image_path=input_image if input_image is not None else None,
                seed=current_seed,
            )
            print(f"Enhanced prompt: {final_prompt}")

            with torch.inference_mode():
                video_context, audio_context = encode_text_simple(text_encoder, final_prompt)
            video_context = video_context.detach().cpu()
            audio_context = audio_context.detach().cpu()

            # Free ~8GB before diffusion-stage models load
            del text_encoder
            gc.collect()
            torch.cuda.empty_cache()
            print("Text encoder freed. Loading transformer (fp8)...")

        # Enforce facial-identity preservation on whatever prompt reaches the model
        final_prompt = _finalize_prompt(final_prompt)
        print(f"Final prompt (identity-locked): {final_prompt}")


        # Run Generation (guide-compliant: partial S1 anchor, full S2)
        from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig

        def _run_gen(tcfg):
            pipeline(
                prompt=final_prompt,
                output_path=str(output_path),
                seed=current_seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images,
                tiling_config=tcfg,
                video_context=video_context,
                audio_context=audio_context,
                stage1_anchor_scale=(float(anchor) if anchor and float(anchor) < 1.0 else None),
            )



        try:
            _run_gen(TilingConfig.default())
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print("CUDA OOM during decode/render - retrying with finer VAE tiles (guide tip)")
            gc.collect()
            torch.cuda.empty_cache()
            fine_tiling = TilingConfig(
                spatial_config=SpatialTilingConfig(tile_size_in_pixels=256, tile_overlap_in_pixels=64),
                temporal_config=TemporalTilingConfig(tile_size_in_frames=32, tile_overlap_in_frames=16),
            )
            _run_gen(fine_tiling)

        del video_context, audio_context, pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return _finalize(str(output_path))

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise gr.Error(f"{type(e).__name__}: {e}", print_exception=False)

def generate_video_example(input_image, prompt, lora_adapter, duration):
    # We pass a default 300s timeout for examples
    output, seed, _vault = generate_video(
        input_image=input_image, 
        prompt=prompt, 
        lora_adapter=lora_adapter,
        duration=5.0, 
        enhance_prompt=True, 
        seed=42, 
        randomize_seed=True, 
        height=DEFAULT_1_STAGE_HEIGHT, 
        width=DEFAULT_1_STAGE_WIDTH,
        gpu_timeout=300
    )
    return output

# -----------------------------------------------------------------------------
# 6. UI Components
# -----------------------------------------------------------------------------

def apply_resolution(resolution: str):
    w, h = resolution.split("x")
    return int(w), int(h)

def apply_duration(duration: str):
    duration_s = int(duration[:-1])
    return duration_s

def apply_gpu_duration(val: str):
    return int(val)

class RadioAnimated(gr.HTML):
    def __init__(self, choices, value=None, **kwargs):
        if not choices or len(choices) < 2:
            raise ValueError("RadioAnimated requires at least 2 choices.")
        if value is None:
            value = choices[0]

        uid = uuid.uuid4().hex[:8]
        group_name = f"ra-{uid}"

        inputs_html = "\n".join(
            f"""
            <input class="ra-input" type="radio" name="{group_name}" id="{group_name}-{i}" value="{c}">
            <label class="ra-label" for="{group_name}-{i}">{c}</label>
            """
            for i, c in enumerate(choices)
        )

        html_template = f"""
        <div class="ra-wrap" data-ra="{uid}">
          <div class="ra-inner">
            <div class="ra-highlight"></div>
            {inputs_html}
          </div>
        </div>
        """

        js_on_load = r"""
        (() => {
          const wrap = element.querySelector('.ra-wrap');
          const inner = element.querySelector('.ra-inner');
          const highlight = element.querySelector('.ra-highlight');
          const inputs = Array.from(element.querySelectorAll('.ra-input'));

          if (!inputs.length) return;

          const choices = inputs.map(i => i.value);

          function setHighlightByIndex(idx) {
            const n = choices.length;
            const pct = 100 / n;
            highlight.style.width = `calc(${pct}% - 6px)`;
            highlight.style.transform = `translateX(${idx * 100}%)`;
          }

          function setCheckedByValue(val, shouldTrigger=false) {
            const idx = Math.max(0, choices.indexOf(val));
            inputs.forEach((inp, i) => { inp.checked = (i === idx); });
            setHighlightByIndex(idx);

            props.value = choices[idx];
            if (shouldTrigger) trigger('change', props.value);
          }

          setCheckedByValue(props.value ?? choices[0], false);

          inputs.forEach((inp) => {
            inp.addEventListener('change', () => {
              setCheckedByValue(inp.value, true);
            });
          });
        })();
        """

        super().__init__(
            value=value,
            html_template=html_template,
            js_on_load=js_on_load,
            **kwargs
        )

# -----------------------------------------------------------------------------
# 7. Gradio Application
# -----------------------------------------------------------------------------

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg:#0B0F1A; --panel:#111827; --line:#1E293B;
  --violet:#7C3AED; --indigo:#6366F1; --pink:#EC4899;
  --txt:#E2E8F0; --sub:#94A3B8;
}
.gradio-container, body { background:#0B0F1A !important; color:var(--txt);
  font-family:'Inter',system-ui,sans-serif !important; }

/* ---- Header ---- */
#hdr { display:flex; align-items:center; justify-content:space-between;
  padding:18px 22px; margin-bottom:14px; border-radius:20px;
  background:linear-gradient(120deg,rgba(124,58,237,.16),rgba(236,72,153,.10) 55%,rgba(99,102,241,.14)),
             radial-gradient(80% 140% at 85% -20%, rgba(236,72,153,.18), transparent 60%);
  border:1px solid var(--line); }
.brand { font-size:1.65em; font-weight:800; letter-spacing:.4px;
  background:linear-gradient(90deg,#C4B5FD,#F9A8D4); -webkit-background-clip:text;
  background-clip:text; color:transparent; }
.brand .sub { font-size:.52em; font-weight:500; letter-spacing:.14em;
  text-transform:uppercase; opacity:.66; margin-left:14px; vertical-align:middle; }
#pill { text-align:right; font-weight:600; font-size:.95em; margin:0;
  color:#86EFAC; text-shadow:0 0 10px rgba(134,239,172,.35); }

/* ---- Panels ---- */
#step-column, .block { background:var(--panel) !important;
  border:1px solid var(--line) !important; border-radius:16px !important;
  transition:border-color .25s ease, transform .25s ease, box-shadow .25s ease; }
#step-column:hover { border-color:rgba(124,58,237,.45) !important;
  box-shadow:0 12px 34px rgba(124,58,237,.10); }
.block-title, .block-label span { color:#A5B4FC !important; }

/* ---- Generate CTA ---- */
.button-gradient { width:100%; min-height:52px; font-size:1.05em; font-weight:700;
  letter-spacing:.02em; border-radius:14px !important; cursor:pointer;
  background:linear-gradient(135deg,var(--violet),var(--pink)) !important;
  border:none !important; color:#fff !important;
  box-shadow:0 8px 26px rgba(124,58,237,.35), inset 0 0 0 1px rgba(255,255,255,.08);
  transition:transform .2s ease, box-shadow .25s ease, filter .2s ease; }
.button-gradient:hover { transform:translateY(-1px); filter:brightness(1.08);
  box-shadow:0 12px 38px rgba(236,72,153,.40); }
.button-gradient:active { transform:translateY(0); }

/* ---- Segmented radio (RadioAnimated) ---- */
.ra-inner{ position:relative; display:inline-flex; align-items:center; padding:6px;
  background:#0F172A; border:1px solid var(--line); border-radius:9999px; overflow:hidden; }
.ra-label{ position:relative; z-index:2; padding:8px 18px; font-size:13.5px; font-weight:600;
  color:var(--sub); cursor:pointer; transition:color .2s ease; white-space:nowrap; }
.ra-highlight{ position:absolute; z-index:1; top:6px; left:6px; height:calc(100% - 12px);
  border-radius:9999px; background:linear-gradient(135deg,rgba(124,58,237,.9),rgba(236,72,153,.9));
  box-shadow:0 4px 14px rgba(124,58,237,.45); transition:transform .22s cubic-bezier(.4,0,.2,1), width .22s; }
.ra-input:checked + .ra-label{ color:#fff; }

/* ---- Inputs ---- */
input[type=text], textarea { background:#0F172A !important; color:var(--txt) !important;
  border:1px solid var(--line) !important; border-radius:12px !important; }
input:focus, textarea:focus { border-color:var(--violet) !important;
  box-shadow:0 0 0 2px rgba(124,58,237,.28) !important; outline:none; }

/* ---- Tabs ---- */
.tab-nav button { font-weight:600; color:var(--sub) !important; cursor:pointer;
  border-bottom:2px solid transparent !important; transition:color .2s, border-color .2s; }
.tab-nav button.selected { color:#fff !important;
  border-bottom:2px solid var(--violet) !important; }

/* ---- Tables / gallery ---- */
table { background:#0F172A; }
.jsplot, .gallery-item { border-radius:14px !important; overflow:hidden; }

/* ---- Motion & accessibility ---- */
@media (prefers-reduced-motion: reduce) {
  * { animation-duration:.01ms !important; transition-duration:.01ms !important; }
}
::-webkit-scrollbar { width:10px; } ::-webkit-scrollbar-thumb {
  background:#243049; border-radius:8px; }
::selection { background:rgba(124,58,237,.45); }
footer { visibility:hidden; }
"""

def _space_status_md():
    """Live status pill (auth-aware: private spaces need the token)."""
    import urllib.request as _u
    try:
        req = _u.Request(f"https://huggingface.co/api/spaces/{SPACE_REPO_ID}")
        tok = os.environ.get("HF_TOKEN")
        if tok:
            req.add_header("Authorization", f"Bearer {tok}")
        d = _json.load(_u.urlopen(req, timeout=8))
        stage = d.get("runtime", {}).get("stage", "?")
        icon = {"RUNNING": "🟢 Running", "RUNNING_BUILDING": "🔨 Building",
                "RUNNING_APP_STARTING": "⏳ Starting"}.get(stage, f"⚪ {stage}")
        return icon
    except Exception as e:
        return f"⚪ Status: {type(e).__name__}"

SPACE_REPO_ID = "tabrezahmed51/LTX-2-LoRAs-Camera-Control-Dolly"

DOCS_MD = """
## 📖 Quick Guide

**Basic:** upload a photo (or skip for text-to-video) → write what *happens* → Generate.

**Pro controls**
| Control | What it does |
|---|---|
| 🎛 Camera LoRA | Official camera-control adapters (dolly/jib/static) |
| 🎬 Camera intent | Director-style sentence: move + speed + amplitude |
| 🎯 Stage-1 Anchor | Lower = more motion freedom; higher = tighter first frame |
| 🔗 LoRA strength | 0.6 subtle · 0.8 balanced · 1.0 full |
| 🧠 Enhance | Gemma rewrites your prompt (identity-locked) |

**Identity tips:** portrait photos render portrait · keep shots ≤5s · gentle moves · send images as documents in the Telegram bot.

**Formats:** W×H divisible by 32 · frames = 24s+1 · two-stage outputs are 2× base.

**🧩 Skill Packs:** *Detailer* runs true IC-LoRA guide mode (your image becomes a
half-res reference — maximum fidelity retention). *Pose/Union Control* stack as
adapters until control-map preprocessors are added.

*Powered by LTX-2 19B distilled · ZeroGPU*
"""

with gr.Blocks(theme=studio_theme, css=css, title="LTX-2 Studio") as demo:
    # ---------- Header ----------
    with gr.Row(elem_id="hdr"):
        with gr.Column(scale=10):
            gr.HTML(
                "<div class='brand'>"
                "<svg width='30' height='30' viewBox='0 0 24 24' fill='none' style='vertical-align:-4px;margin-right:10px' aria-hidden='true'>"
                "<defs><linearGradient id='lg' x1='0' y1='0' x2='1' y2='1'>"
                "<stop offset='0' stop-color='#7C3AED'/><stop offset='1' stop-color='#EC4899'/></linearGradient></defs>"
                "<path d='M15 8v8H5a2 2 0 0 1-2-2V10a2 2 0 0 1 2-2h10zm2-2h1a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-1V6zM4 6h11a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2z' fill='url(#lg)'/></svg>"
                "<b>LTX-2 STUDIO</b>"
                "<span class='sub'>Camera-Control · Identity-Lock · Director Timelines</span></div>"
            )
        with gr.Column(scale=2, min_width=140):
            status_pill = gr.Markdown(_space_status_md(), elem_id="pill")

    with gr.Tabs():
        # ================= GENERATE =================
        with gr.Tab("🎬 Generate"):
            with gr.Row(elem_id="col-container"):
                with gr.Column(elem_id="step-column"):
                    input_image = gr.Image(label="Input Image (optional)", type="filepath", height=300)
                    prompt = gr.Textbox(
                        label="Prompt — describe what HAPPENS",
                        value="Make this image come alive with cinematic motion...",
                        lines=3, placeholder="Motion, action, camera, audio...")
                    lora_adapter = gr.Dropdown(
                        label="🎛 Camera LoRA",
                        choices=list(ADAPTER_SPECS.keys()), value="Camera-Control-Dolly-Left")
                    style_pack = gr.Dropdown(
                        label="🧩 Skill Pack (experimental IC-LoRA stack)",
                        choices=list(STYLE_PACKS.keys()), value="None")
                    enhance_prompt = gr.Checkbox(label="Enhance Prompt", value=True, visible=False)
                    generate_btn = gr.Button("Generate Video", variant="primary",
                                             elem_classes="button-gradient")

                with gr.Column(elem_id="step-column"):
                    output_video = gr.Video(label="Generated Video", autoplay=True, height=360)
                    vault_url_box = gr.Textbox(label="Vault link", interactive=False, visible=True)

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**Duration**")
                            radioanimated_duration = RadioAnimated(["3s","5s","10s"], value="3s",
                                                                   elem_id="radioanimated_duration")
                            duration = gr.Number(value=3.0, visible=False)
                        with gr.Column():
                            gr.Markdown("**Resolution**")
                            radioanimated_resolution = RadioAnimated(
                                ["768x512","512x512","512x768"],
                                value=f"{DEFAULT_1_STAGE_WIDTH}x{DEFAULT_1_STAGE_HEIGHT}",
                                elem_id="radioanimated_resolution")
                            width = gr.Number(value=DEFAULT_1_STAGE_WIDTH, visible=False)
                            height = gr.Number(value=DEFAULT_1_STAGE_HEIGHT, visible=False)

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**GPU Duration**")
                            radioanimated_gpu_duration = RadioAnimated(
                                ["90","120","180","240","300"], value="300",
                                elem_id="radioanimated_gpu_duration")
                            gpu_duration_state = gr.Number(value=300, visible=False)
                        with gr.Column():
                            gr.Markdown("**🎯 Anchor / 🔗 LoRA**")
                            radioanimated_anchor = RadioAnimated(
                                ["🔒 Tight","⚖️ Balanced","🌀 Loose"], value="🔒 Tight",
                                elem_id="radioanimated_anchor")
                            anchor_state = gr.Number(value=1.0, visible=False)
                            radioanimated_lstr = RadioAnimated(
                                ["0.6","0.8","1.0"], value="0.8",
                                elem_id="radioanimated_lstr")
                            lora_str_input = gr.Number(value=0.8, visible=False)

                    with gr.Accordion("✨ Post-Processing", open=False):
                        with gr.Row():
                            chk_rife = gr.Checkbox(label="🌀 Smooth Motion (RIFE 2× frames)", value=False)
                            chk_upscale = gr.Checkbox(label="🖼 Enhance Resolution (Swin2SR 2×)", value=False)
                        with gr.Accordion("🎙 Voiceover (MMS-TTS)", open=False):
                            chk_voiceover = gr.Checkbox(label="Enable Voiceover", value=False)
                            vo_text = gr.Textbox(label="Narration text", lines=2,
                                                 placeholder="Script for the voiceover...",
                                                 visible=True)
                            vo_lang = gr.Dropdown(
                                ["English (US)","Hindi","Tamil","Telugu","Bengali",
                                 "Marathi","Gujarati","Kannada","Malayalam","Urdu","Punjabi"],
                                value="English (US)", label="🌐 Language")
                            vo_mix = gr.Radio(
                                ["Replace native audio", "Mix over native audio"],
                                value="Replace native audio", label="Audio mode")

                    with gr.Accordion("Advanced Settings", open=False, visible=False):
                        seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED,
                                         value=DEFAULT_SEED, step=1)
                        randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                    anchor_input = gr.Number(value=1.0, visible=False)

        # ================= GALLERY =================
        with gr.Tab("🖼 Gallery (Vault)"):
            gr.Markdown("Recent renders saved to your private HF Dataset vault.")
            refresh_btn = gr.Button("🔄 Refresh history", size="sm")
            history_df = gr.DataFrame(
                headers=["time", "prompt", "seed", "size", "url"], interactive=False,
                wrap=True, column_widths=[130, 380, 70, 90, 200])
            hist_video = gr.Video(label="Selected render", height=340)

            def load_history():
                try:
                    rows = vault_recent(15)
                    return [[r.get("ts",""), (r.get("prompt") or "")[:80], r.get("seed",""),
                             f"{r.get('width','?')}x{r.get('height','?')}", r.get("url","")] for r in rows]
                except Exception as e:
                    print(f"[gallery] load failed: {e}")
                    return []
            refresh_btn.click(load_history, outputs=[history_df])
            demo.load(lambda: [], outputs=[history_df])  # lazy: no network at page-open

            def pick_row(evt: gr.SelectData):
                try:
                    rows = vault_recent(15)[evt.index[0]]
                    return rows.get("url")
                except Exception:
                    return None
            history_df.select(pick_row, outputs=[hist_video])

        # ================= PRESETS =================
        with gr.Tab("💾 Presets"):
            gr.Markdown("Save / load parameter sets (stored free in your HF vault).")
            preset_name = gr.Textbox(label="Preset name", placeholder="Cinematic Portrait v2")
            with gr.Row():
                save_preset_btn = gr.Button("💾 Save current", size="sm")
                apply_preset_dd = gr.Dropdown(label="Load preset", choices=[], interactive=True)
                load_preset_btn = gr.Button("📥 Apply", size="sm")
            preset_status = gr.Markdown("")

            def do_save(name, dur, res_w, res_h, anch, lst):
                if not name.strip():
                    return "", "⚠️ Give the preset a name"
                ok, msgtxt = vault_save_preset(name.strip(), {
                    "duration": float(dur), "width": int(res_w), "height": int(res_h),
                    "anchor": float(anch), "lstr": float(lst)})
                names = vault_list_presets()
                return gr.update(choices=names), ("✅ " if ok else "❌ ") + msgtxt
            save_preset_btn.click(do_save,
                inputs=[preset_name, duration, width, height, anchor_input, lora_str_input],
                outputs=[apply_preset_dd, preset_status])

            def refresh_presets():
                return gr.update(choices=vault_list_presets())
            apply_preset_dd.focus(refresh_presets, outputs=[apply_preset_dd])

            def do_apply(name):
                p = vault_get_preset(name) or {}
                return (p.get("duration", 3.0), p.get("width", DEFAULT_1_STAGE_WIDTH),
                        p.get("height", DEFAULT_1_STAGE_HEIGHT),
                        p.get("anchor", 1.0), p.get("lstr", 0.8),
                        f"✅ Applied '{name}'")
            load_preset_btn.click(do_apply, inputs=[apply_preset_dd],
                outputs=[duration, width, height, anchor_input, lora_str_input, preset_status])

        # ================= DOCS =================
        with gr.Tab("📖 Guide"):
            gr.Markdown(DOCS_MD)

    # ---------- event wiring ----------
    radioanimated_duration.change(fn=apply_duration, inputs=radioanimated_duration, outputs=[duration])
    radioanimated_resolution.change(fn=apply_resolution, inputs=radioanimated_resolution, outputs=[width, height])
    radioanimated_gpu_duration.change(fn=apply_gpu_duration, inputs=radioanimated_gpu_duration, outputs=[gpu_duration_state])

    def _anchor_label_to_num(v):
        return {"🔒 Tight": 1.0, "⚖️ Balanced": 0.9, "🌀 Loose": 0.8}.get(v, 1.0)
    def _anchor_pair(v):
        num = {"🔒 Tight": 1.0, "⚖️ Balanced": 0.9, "🌀 Loose": 0.8}.get(v, 1.0)
        return num, num
    radioanimated_anchor.change(fn=_anchor_pair, inputs=[radioanimated_anchor],
                                outputs=[anchor_state, anchor_input])
    radioanimated_lstr.change(fn=lambda v: float(v), inputs=radioanimated_lstr, outputs=[lora_str_input])

    def _toggle_vo(v):
        return gr.update(visible=v), gr.update(visible=v), gr.update(visible=v)
    chk_voiceover.change(_toggle_vo, inputs=[chk_voiceover],
                         outputs=[vo_text, vo_lang, vo_mix])

    generate_btn.click(
        fn=generate_video,
        inputs=[input_image, prompt, lora_adapter, duration, enhance_prompt, seed,
                randomize_seed, height, width, gpu_duration_state,
                anchor_input, lora_str_input, style_pack,
                chk_rife, chk_upscale, vo_text, vo_lang, vo_mix],
        outputs=[output_video, seed, vault_url_box]
    )

    gr.Examples(
        examples=[
            ["examples/dolly_left.jpg", "Dolly-left camera move, sliding left in an open space, revealing a stationary car, off-frame left elements enter the shot, cinematic parallax", "Camera-Control-Dolly-Left"],
            ["examples/dolly_in.jpg", "Slow dolly-in toward a face, gentle zoom-in, cinematic parallax.", "Camera-Control-Dolly-In"],
            ["examples/dolly_right.jpg", "Dolly-right camera move in an open space, sliding right, cinematic parallax.", "Camera-Control-Dolly-Right"],
            ["examples/dolly_out.jpg", "Slow dolly-out with gentle zoom-out, cinematic parallax.", "Camera-Control-Dolly-Out"],
            ["examples/Jib_down.jpg", "Jib-down camera move, smooth vertical descent, cinematic reveal.", "Camera-Control-Jib-Down"],
            ["examples/jib-up.jpg", "Jib-up camera move, smooth vertical ascent, cinematic reveal.", "Camera-Control-Jib-Up"],
            ["examples/cam_static.jpg", "Static camera, locked-off shot, no camera movement.", "Camera-Control-Static"],
        ],
        fn=generate_video_example,
        inputs=[input_image, prompt, lora_adapter],
        outputs=[output_video],
        label="Examples",
        cache_examples=False
    )

    gr.Markdown("_Higher duration/resolution increases GPU usage. Vault saves best-effort when HF_TOKEN secret is present._")


if __name__ == "__main__":
    demo.launch(ssr_mode=False, mcp_server=True)