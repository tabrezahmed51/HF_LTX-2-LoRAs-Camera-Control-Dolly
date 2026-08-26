#!/usr/bin/env python3
"""
LTX-2-LoRAs-Camera-Control-Dolly — Telegram Bot <-> HF Space bridge (full UI replica)
=====================================================================================
Mirrors the complete Gradio feature menu of the Space inside Telegram:

  • Input image (photo or uncompressed document) / text-to-video (/t2v)
  • Prompt (caption)
  • Camera Control LoRA dropdown  -> /settings buttons (all 7 + none)
  • Enhance Prompt toggle         -> /settings
  • Duration 3s/5s/10s            -> /settings
  • Resolution presets            -> /settings (+ smart auto-fit)
  • GPU Duration 90..300          -> /settings
  • Advanced: Seed + Randomize    -> /settings, /seed <n>
  • Examples gallery              -> /examples (runs real Space examples)

Diagnostics: /status /live /ping /gpu /queue /cancel /refresh /reboot /logs /buildlogs

Run:
    pip install requests pillow
    python ltx2_telegram_bot.py
"""

import io
import json
import os
import queue
import re
import threading
import time
import traceback
import uuid

import requests

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from huggingface_hub import HfApi, hf_hub_download
    HAS_HUB = True
except ImportError:
    HAS_HUB = False

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Secrets: prefer environment variables (export TG_BOT_TOKEN=... etc).
# Fallbacks below exist for local convenience - ROTATE these tokens and
# remove fallbacks before sharing this file with anyone.
BOT_TOKEN   = os.environ["TG_BOT_TOKEN"]  # required
CHANNEL_ID  = os.environ["TG_CHANNEL_ID"]  # required
HF_TOKEN    = os.environ["HF_TOKEN"]  # required
ADMINS_FILE = os.path.join(os.path.expanduser("~"), ".ltx2_admins.json")
SPACE_URL   = "https://tabrezahmed51-ltx-2-loras-camera-control-dolly.hf.space"
SPACE_REPO  = "tabrezahmed51/LTX-2-LoRAs-Camera-Control-Dolly"
SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".ltx2_bot_settings.json")
VAULT_REPO   = "tabrezahmed51/ltx2-space-vault"

TG_API  = f"https://api.telegram.org/bot{BOT_TOKEN}"
HF_API  = f"https://huggingface.co/api/spaces/{SPACE_REPO}"
HF_AUTH = {"Authorization": f"Bearer {HF_TOKEN}"}
# Persistent keep-alive connections (huge win on mobile networks:
# avoids a fresh TCP+TLS handshake for every single API call)
_TG_S = requests.Session()
_HF_S = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
for _sess in (_TG_S, _HF_S):
    _sess.mount("https://", _adapter)

GPU_HARD_LIMIT = 900
MAX_CAPTION    = 1024

ADAPTER_MAP = {
    "dolly-in":   ("Camera-Control-Dolly-In",     "🎯 Dolly-In"),
    "dolly-out":  ("Camera-Control-Dolly-Out",    "🎥 Dolly-Out"),
    "dolly-left": ("Camera-Control-Dolly-Left",   "⬅️ Dolly-Left"),
    "dolly-right":("Camera-Control-Dolly-Right",  "➡️ Dolly-Right"),
    "jib-up":     ("Camera-Control-Jib-Up",       "⬆️ Jib-Up"),
    "jib-down":   ("Camera-Control-Jib-Down",     "⬇️ Jib-Down"),
    "static":     ("Camera-Control-Static",       "🔒 Static"),
}
ADAPTER_LABEL = {v[0]: v[1] for v in ADAPTER_MAP.values()}
DEFAULT_ADAPTER = "Camera-Control-Dolly-Left"

# --- Director camera-intent presets (ported from comfyui_ltx25_director, MIT) ---
CAMERA_MOVES = {
    "none": None,
    "static": "holds a static locked-off shot",
    "push_in": "pushes in toward the subject",
    "pull_out": "pulls out away from the subject",
    "dolly_in": "dollies in toward the subject",
    "dolly_out": "dollies out away from the subject",
    "pan_left": "pans left",
    "pan_right": "pans right",
    "truck_left": "trucks left",
    "truck_right": "trucks right",
    "tilt_up": "tilts up",
    "tilt_down": "tilts down",
    "orbit": "arcs around the subject",
    "tracking": "tracks the moving subject",
    "crane_up": "cranes / booms upward",
    "crane_down": "cranes / booms downward",
    "handheld": "shakes slightly with handheld micro-motion",
    "handheld_strong": "shakes strongly with handheld motion",
    "pov": "switches to a POV shot from the subject's viewpoint",
}
CAMERA_SPEEDS = {"slow": "slow speed", "medium": "medium speed", "fast": "fast speed"}
CAMERA_AMPS = {"small": "small amplitude", "medium": "medium amplitude", "large": "large amplitude"}
ANCHOR_LABELS = {1.0: "🔒 Tight", 0.9: "⚖️ Balanced", 0.8: "🌀 Loose"}
VO_LANGS = ["English (US)","Hindi","Tamil","Telugu","Bengali","Marathi","Gujarati","Kannada","Malayalam","Urdu","Punjabi"]

def compose_camera_sentence(cam: str | None, spd: str, amp: str) -> str:
    """Compile camera intent into an English sentence (Director style)."""
    move_txt = CAMERA_MOVES.get(cam or "none")
    if not move_txt:
        return ""
    parts = [f"The camera {move_txt}"]
    if spd in CAMERA_SPEEDS:
        parts.append(CAMERA_SPEEDS[spd])
    if amp in CAMERA_AMPS:
        parts.append(CAMERA_AMPS[amp])
    return ", ".join(parts) + "."
RESOLUTIONS = {(768, 512): "🖼 Landscape", (512, 512): "🔲 Square", (512, 768): "📱 Portrait"}
DURATIONS   = [3, 5, 10]
GPU_CHOICES = [90, 120, 180, 240, 300]

EXAMPLES = [
    ("dolly_left.jpg",  "Camera-Control-Dolly-Left",
     "Dolly-left camera move, sliding left in an open space, revealing a stationary car, off-frame left elements enter the shot, cinematic parallax"),
    ("dolly_in.jpg",    "Camera-Control-Dolly-In",
     "Slow dolly-in toward a face, gentle zoom-in, cinematic parallax."),
    ("dolly_right.jpg", "Camera-Control-Dolly-Right",
     "Dolly-right camera move in an open space, sliding right, cinematic parallax."),
    ("dolly_out.jpg",   "Camera-Control-Dolly-Out",
     "Slow dolly-out with gentle zoom-out, cinematic parallax."),
    ("Jib_down.jpg",    "Camera-Control-Jib-Down",
     "Jib-down camera move, smooth vertical descent, cinematic reveal."),
    ("jib-up.jpg",      "Camera-Control-Jib-Up",
     "Jib-up camera move, smooth vertical ascent, cinematic reveal."),
    ("cam_static.jpg",  "Camera-Control-Static",
     "Static camera, locked-off shot, no camera movement."),
]

WELCOME = (
    "📹 *LTX-2-LoRAs Camera Control Dolly*\n\n"
    "Send a *photo* (or 📄 uncompressed image) with your motion prompt as "
    "the caption — I animate it and publish photo + video to the channel.\n\n"
    "🏠 *Tap /menu for the full button menu*\n\n"
    "*🎛 Control Panel:* /settings\n"
    "_adapter • duration • resolution • GPU time • seed • enhance_\n"
    "*🎞 Examples gallery:* /examples\n"
    "*🎬 Timeline mode:* /director 3:shot1; 4:shot2\n"
    "*🖼 Vault:* /gallery · 💾 `/psave <name>` presets\n"
    "*📝 Text-only video:* `/t2v your prompt`\n\n"
    "*Quick caption tags:* #dolly-in #dolly-out #dolly-left #dolly-right "
    "#jib-up #jib-down #static #nolora #noenhance #3s/#5s/#10s\n\n"
    "*Diagnostics:* /status /live /ping /gpu /queue /cancel /refresh /reboot "
    "/logs /buildlogs\n\n"
    "_Generation takes ~2-5 minutes._ 💡 Send images as documents for best identity."
)

# -----------------------------------------------------------------------------
# Per-user settings (persistent, mirrors the Space UI controls)
# -----------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    "adapter": DEFAULT_ADAPTER,
    "duration": 3,
    "resolution": "auto",        # auto | WxH string
    "gpu": 300,
    "enhance": True,
    "randomize": True,
    "seed": 42,
    "cam": "none",
    "spd": "medium",
    "amp": "medium",
    "anchor": 1.0,
    "lstr": 0.8,
    "pack": "None",    # Skill Pack: None | Pose Control | Union Control | Detailer
}
_SETTINGS: dict = {}

def load_settings():
    global _SETTINGS
    try:
        with open(SETTINGS_FILE) as f:
            _SETTINGS = json.load(f)
    except Exception:
        _SETTINGS = {}

def save_settings():
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(_SETTINGS, f, indent=1)
    except Exception as e:
        print(f"[settings] save failed: {e}")

def _load_admins() -> dict:
    try:
        return json.load(open(ADMINS_FILE, encoding="utf-8"))
    except Exception:
        return {"admins": [], "allowed": []}

def _save_admins(d: dict):
    json.dump(d, open(ADMINS_FILE, "w", encoding="utf-8"))

_ADMINS = _load_admins()

def is_admin(chat_id: int) -> bool:
    return str(chat_id) in  str(chat_id) in _ADMINS["admins"]

def is_allowed(chat_id: int) -> bool:
    """Owner/admins always allowed. Open mode until owner claims; then allowlist only."""
    cid = str(chat_id)
    if is_admin(cid):
        return True
    if not _ADMINS["admins"]:
        return True
    return cid in _ADMINS["allowed"]
    # open-access mode until the owner claims the bot
    return not _ADMINS["admins"]  (cid in _ADMINS["allowed"] or not _ADMINS["admins"])

def claim_owner(chat_id: int) -> bool:
    """First /start in a private chat claims ownership (self-locking)."""
    if _ADMINS["admins"]:
        return False
    _ADMINS["admins"].append(str(chat_id))
    _ADMINS["allowed"].append(str(chat_id))
    _save_admins(_ADMINS)
    return True

def get_settings(chat_id: int) -> dict:
    s = _SETTINGS.setdefault(str(chat_id), dict(DEFAULT_SETTINGS))
    for k, v in DEFAULT_SETTINGS.items():
        s.setdefault(k, v)
    return s

def resolve_resolution(image_bytes: bytes | None, res_key: str) -> tuple[int, int]:
    """Return (width, height). 'auto' inspects the photo orientation."""
    if res_key != "auto":
        w, h = map(int, res_key.split("x"))
        return w, h
    if image_bytes and HAS_PIL:
        try:
            img = PILImage.open(io.BytesIO(image_bytes))
            iw, ih = img.size
            if ih > iw * 1.05:
                return 512, 768
            if iw > ih * 1.05:
                return 768, 512
            return 512, 512
        except Exception:
            pass
    return 768, 512

# -----------------------------------------------------------------------------
# Animation system
# -----------------------------------------------------------------------------
SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
GEN_STAGES = [
    "🖼 Conditioning source image…",
    "🧠 Enhancing prompt (identity-locked)…",
    "📦 Loading Gemma text encoder…",
    "⚡️ Building FP8 transformer…",
    "🌀 Diffusing audio-video latents…",
    "✨ Stage-2 upscaling x2…",
    "🎞 Encoding MP4…",
]

def fmt_secs(s: int) -> str:
    return f"{s//60}:{s%60:02d}"

def progress_bar(pct: float, width: int = 12) -> str:
    pct = max(0.0, min(1.0, pct))
    filled = round(pct * width)
    return "▰" * filled + "▱" * (width - filled)

class LiveAnim:
    EDIT_EVERY = 2.6

    def __init__(self, chat_id: int, title: str, adopt_msg_id: int | None = None):
        self.chat_id, self.title = chat_id, title
        self.stage, self.pct = "Starting…", 0.02
        self.t0, self.done = time.time(), False
        self.msg_id = adopt_msg_id
        self._lock = threading.Lock()
        if not self.msg_id:
            r = tg("sendMessage", chat_id=chat_id,
                   text=f"{SPINNER[0]} *{title}*\n{self.stage}", parse_mode="Markdown")
            self.msg_id = (r.get("result") or {}).get("message_id")
        if self.msg_id:
            LiveAnim._registry.add(self)
            threading.Thread(target=self._loop, daemon=True).start()

    def update(self, stage=None, pct=None):
        with self._lock:
            if stage is not None:
                self.stage = stage
            if pct is not None:
                self.pct = max(self.pct, min(0.99, pct))

    def close(self, final_text):
        with self._lock:
            self.done = True
        if self.msg_id:
            tg("editMessageText", chat_id=self.chat_id, message_id=self.msg_id,
               text=final_text, parse_mode="Markdown")

    def _render(self, i):
        return (f"{SPINNER[i % len(SPINNER)]} *{self.title}*\n{self.stage}\n"
                f"{progress_bar(self.pct)} {int(self.pct*100)}% • ⏱ {fmt_secs(int(time.time()-self.t0))}")

    def _loop(self):
        i = 0
        while True:
            with self._lock:
                if self.done:
                    break
            try:
                if self.msg_id:
                    tg("editMessageText", chat_id=self.chat_id, message_id=self.msg_id,
                       text=self._render(i), parse_mode="Markdown")
            except Exception:
                pass
            i += 1
            time.sleep(LiveAnim.EDIT_EVERY)

LiveAnim._registry = set()

# -----------------------------------------------------------------------------
# Telegram helpers
# -----------------------------------------------------------------------------
def tg(method: str, **payload):
    try:
        r = _TG_S.post(f"{TG_API}/{method}", json=payload, timeout=60)
        j = r.json()
        if not j.get("ok"):
            print(f"[tg] {method}: {r.text[:200]}")
        return j
    except Exception as e:
        print(f"[tg] {method} exc: {e}")
        return {}

def send_message(chat_id, text, reply_markup=None):
    p = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    if reply_markup:
        p["reply_markup"] = json.dumps(reply_markup)
    return tg("sendMessage", **p)

def download_tg_file(file_id: str) -> bytes:
    info = tg("getFile", file_id=file_id).get("result", {})
    path = info.get("file_path")
    if not path:
        raise RuntimeError("getFile returned no file_path")
    host = TG_API.replace(f"/bot{BOT_TOKEN}", "")
    r = _TG_S.get(f"{host}/file/bot{BOT_TOKEN}/{path}", timeout=120)
    r.raise_for_status()
    return r.content

def send_video_file(chat_id: int, data: bytes, caption: str):
    """sendVideo with multipart upload (tg() is JSON-only)."""
    try:
        r = _TG_S.post(f"{TG_API}/sendVideo",
                       data={"chat_id": chat_id, "caption": caption,
                             "supports_streaming": "true"},
                       files={"video": ("video.mp4", io.BytesIO(data), "video/mp4")},
                       timeout=300)
        if not r.json().get("ok"):
            print(f"[tg] sendVideo: {r.text[:200]}")
    except Exception as e:
        print(f"[tg] sendVideo exc: {e}")

def escape_html(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def parse_caption(caption: str):
    """Caption tag overrides (one-off); returns overrides dict."""
    ov = {}
    text = (caption or "").strip()
    for tag in re.findall(r"#([\wx\-]+)", text.lower()) if text else []:
        if tag == "nolora":
            ov["adapter"] = None
        elif tag == "noenhance":
            ov["enhance"] = False
        elif tag.startswith("cam_") and tag[4:] in CAMERA_MOVES:
            ov["cam"] = tag[4:]
        elif tag.startswith("spd_") and tag[4:] in CAMERA_SPEEDS:
            ov["spd"] = tag[4:]
        elif tag.startswith("amp_") and tag[4:] in CAMERA_AMPS:
            ov["amp"] = tag[4:]
        elif tag == "anchor_tight":
            ov["anchor"] = 1.0
        elif tag in ("anchor_balanced", "anchor09"):
            ov["anchor"] = 0.9
        elif tag in ("anchor_loose", "anchor08"):
            ov["anchor"] = 0.8
        elif tag in ("lora06",):
            ov["lstr"] = 0.6
        elif tag in ("lora10",):
            ov["lstr"] = 1.0
        elif tag == "rife":
            ov["rife"] = True
        elif tag == "norf":
            ov["rife"] = False
        elif tag in ("up", "upscale"):
            ov["upscale"] = True
        elif tag == "vomix":
            ov.update(vo_on=True, vo_mode="Mix over native audio")
        elif tag == "voreplace":
            ov.update(vo_on=True, vo_mode="Replace native audio")
        elif tag == "vooff":
            ov["vo_on"] = False
            ov["vo_text"] = ""
        elif tag.startswith("pack_"):
            pm = {"pose": "Pose Control", "union": "Union Control",
                  "detail": "Detailer", "detailer": "Detailer", "none": "None"}
            if tag[5:] in pm:
                ov["pack"] = pm[tag[5:]]
        elif tag in ADAPTER_MAP:
            ov["adapter"] = ADAPTER_MAP[tag][0]
        elif re.fullmatch(r"(\d+)s", tag) and int(tag[:-1]) in DURATIONS:
            ov["duration"] = int(tag[:-1])
        else:
            m = re.fullmatch(r"(\d+)x(\d+)", tag)
            if m and (int(m.group(1)), int(m.group(2))) in RESOLUTIONS:
                ov["resolution"] = f"{m.group(1)}x{m.group(2)}"
    clean = re.sub(r"#\S+", "", text).strip() if text else ""
    return clean, ov

# -----------------------------------------------------------------------------
# HF Space bridge
# -----------------------------------------------------------------------------
def space_upload_image(image_bytes: bytes, filename="photo.jpg") -> str:
    r = _HF_S.post(f"{SPACE_URL}/gradio_api/upload", headers=HF_AUTH,
                      files={"files": (filename, image_bytes, "image/jpeg")}, timeout=120)
    r.raise_for_status()
    paths = r.json()
    if not paths:
        raise RuntimeError("upload returned empty result")
    return paths[0]

def space_generate(job: dict, anim=None) -> tuple[str, int]:
    data = [
        {"path": job["remote_path"], "meta": {"_type": "gradio.FileData"}} if job.get("remote_path") else None,
        job["prompt"],
        job["adapter"],
        float(job["duration"]),
        bool(job["enhance"]),
        int(job["seed"]),
        bool(job["randomize"]),
        int(job["height"]),
        int(job["width"]),
        int(job["gpu"]),
        float(job.get("anchor", 1.0)),
        float(job.get("lstr", 0.8)),
        job.get("pack", "None"),
        bool(job.get("rife", False)),
        bool(job.get("upscale", False)),
        str(job.get("vo_text", "")),
        str(job.get("vo_lang", "English (US)")),
        str(job.get("vo_mix", "Replace native audio")),
    ]
    if anim:
        anim.update("📨 Submitting to ZeroGPU queue…", 0.05)
    r = _HF_S.post(f"{SPACE_URL}/gradio_api/call/generate_video",
                      headers={**HF_AUTH, "Content-Type": "application/json"},
                      json={"data": data}, timeout=60)
    r.raise_for_status()
    event_id = r.json()["event_id"]
    print(f"[space] submitted {event_id}: {job['summary']}")

    started, stage_idx = time.time(), 0
    with _HF_S.get(f"{SPACE_URL}/gradio_api/call/generate_video/{event_id}",
                      headers=HF_AUTH, stream=True, timeout=(30, 90)) as resp:
        resp.raise_for_status()
        event = None
        for raw in resp.iter_lines(decode_unicode=True):
            if time.time() - started > GPU_HARD_LIMIT:
                raise TimeoutError("Generation exceeded hard time limit")
            if anim:
                el = time.time() - started
                idx = min(int(el / 34.0), len(GEN_STAGES) - 1)
                if idx != stage_idx:
                    stage_idx = idx
                    anim.update(GEN_STAGES[idx], min(0.92, 0.10 + el / 240.0))
            if not raw or not isinstance(raw, str):
                continue
            if raw.startswith("event:"):
                event = raw.split(":", 1)[1].strip()
            elif raw.startswith("data:") and event:
                payload = raw.split(":", 1)[1].strip()
                if event == "error":
                    raise RuntimeError(f"Space error: {payload}")
                if event == "complete":
                    result = json.loads(payload)
                    if anim:
                        anim.update("🎬 Render complete!", 0.95)
                    return result[0]["url"], result[1]
                event = None
    raise RuntimeError("Stream ended without completion")

def space_download_video(url: str) -> bytes:
    r = _HF_S.get(url, headers=HF_AUTH, timeout=300)
    r.raise_for_status()
    return r.content

def fetch_example_image(filename: str) -> bytes:
    url = f"https://huggingface.co/spaces/{SPACE_REPO}/resolve/main/examples/{filename}"
    r = _HF_S.get(url, headers=HF_AUTH, timeout=120)
    r.raise_for_status()
    return r.content

# --- diagnostics -------------------------------------------------------------
def space_status() -> dict:
    r = _HF_S.get(HF_API, headers=HF_AUTH, timeout=20)
    r.raise_for_status()
    d = r.json()
    rt = d.get("runtime", {})
    return {"stage": rt.get("stage", "?"),
            "runtime_sha": (rt.get("sha") or "?")[:7],
            "repo_sha": d.get("sha", "?")[:7],
            "hardware": (rt.get("hardware") or {}).get("current", "?"),
            "replicas": rt.get("replicas") or {},
            "devMode": rt.get("devMode"),
            "sdk_version": (d.get("cardData") or {}).get("sdk_version"),
            "private": d.get("private"), "likes": d.get("likes")}

def render_status_card() -> str:
    s = space_status()
    icon = {"RUNNING": "🟢", "RUNNING_BUILDING": "🔨", "RUNNING_APP_STARTING": "⏳",
            "PAUSED": "🌙", "BUILD_ERROR": "🔴"}.get(s["stage"], "⚪️")
    rep = s["replicas"]
    return (f"{icon} *Space Live Status*\n"
            f"├ Stage: `{s['stage']}`\n"
            f"├ Deployed `{s['runtime_sha']}` (repo `{s['repo_sha']}`)\n"
            f"├ Hardware `{s['hardware']}` • replicas {rep.get('current','?')}/{rep.get('requested','?')}\n"
            f"└ SDK `{s['sdk_version']}` • devMode `{s['devMode']}`")

def gpu_snapshot() -> str:
    try:
        with _HF_S.get(f"{HF_API}/events", headers=HF_AUTH, stream=True, timeout=(15, 8)) as resp:
            for raw in resp.iter_lines(decode_unicode=True):
                if isinstance(raw, str) and raw.startswith("data:"):
                    v = raw.split(":", 1)[1].strip()
                    try:
                        int(v)
                        return v
                    except ValueError:
                        continue
    except Exception:
        pass
    return "?"

def tail_logs(which: str, n: int = 15) -> str:
    out = []
    try:
        with _HF_S.get(f"{HF_API}/logs/{which}", headers=HF_AUTH,
                          stream=True, timeout=(20, 10)) as resp:
            for line in resp.iter_lines(decode_unicode=True):
                if not (isinstance(line, str) and line.startswith("data:")):
                    continue
                try:
                    txt = (json.loads(line[5:]).get("data") or "").replace("\x1b[A", "").rstrip()
                    if txt.strip():
                        out.append(txt)
                    if len(out) >= n:
                        break
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return f"log fetch failed: {e}"
    return "\n".join(out[-n:]) if out else "(no log output yet)"

def ping_targets() -> str:
    res = []
    t = time.time(); _TG_S.get("https://api.telegram.org", timeout=10)
    res.append(f"Telegram API: {(time.time()-t)*1000:.0f} ms")
    t = time.time(); _HF_S.head(HF_API, headers=HF_AUTH, timeout=10)
    res.append(f"HF Hub API:   {(time.time()-t)*1000:.0f} ms")
    try:
        t = time.time(); rr = _HF_S.get(f"{SPACE_URL}/", headers=HF_AUTH, timeout=25)
        res.append(f"Space app:    {(time.time()-t)*1000:.0f} ms (HTTP {rr.status_code})")
    except Exception as e:
        res.append(f"Space app:    UNREACHABLE ({e.__class__.__name__})")
    return "\n".join(res)

# -----------------------------------------------------------------------------
# Channel album
# -----------------------------------------------------------------------------
def post_album_to_channel(image_bytes, video_bytes, prompt, meta):
    caption = f"🎬 {prompt}\n\n{meta}\n🤖 @ltx2loras_bot"[:MAX_CAPTION]
    media = [{"type": "photo", "media": "attach://file0", "caption": caption, "parse_mode": "HTML"},
             {"type": "video", "media": "attach://file1"}]
    files = {"file0": ("source.jpg", image_bytes, "image/jpeg"),
             "file1": ("generated.mp4", video_bytes, "video/mp4")}
    r = _TG_S.post(f"{TG_API}/sendMediaGroup",
                      data={"chat_id": CHANNEL_ID, "media": json.dumps(media)},
                      files=files, timeout=300)
    j = r.json()
    if not j.get("ok"):
        raise RuntimeError(f"sendMediaGroup failed: {r.text[:400]}")
    return j["result"]

# -----------------------------------------------------------------------------
# Settings panel (replicates the Space UI menu)
# -----------------------------------------------------------------------------
ADAPTER_BUTTONS = [('None', None), ('In', 'Camera-Control-Dolly-In'), ('Out', 'Camera-Control-Dolly-Out'), ('Left', 'Camera-Control-Dolly-Left'), ('Right', 'Camera-Control-Dolly-Right'), ('Jib-Up', 'Camera-Control-Jib-Up'), ('Jib-Down', 'Camera-Control-Jib-Down'), ('Static', 'Camera-Control-Static')]
PACK_BUTTONS = [('None', 'None'), ('Pose', 'Pose Control'), ('Union', 'Union Control'), ('Detailer', 'Detailer')]
def settings_keyboard(s: dict) -> dict:
    B = lambda label, action, val="x": {"text": label, "callback_data": "cfg|"+action+"|"+str(val)}
    cur_ad = s["adapter"]
    cur_pk = s.get("pack", "None") or "None"
    rows = []
    row = []
    for lbl, val in ADAPTER_BUTTONS:
        mark = "[OK] " if cur_ad == val else ""
        cb = "none" if val is None else val
        row.append({"text": mark+lbl, "callback_data": "cfg|adapter|"+cb})
        if len(row) == 4:
            rows.append(row); row = []
    if row: rows.append(row)
    pk = cur_pk if cur_pk != "None" else "None"
    rows.append([{"text": "Pack: "+pk, "callback_data": "noop"}])
    row = []
    for lbl, val in PACK_BUTTONS:
        mark = "[OK] " if cur_pk == val else ""
        row.append({"text": mark+lbl, "callback_data": "cfg|packsel|"+val})
    rows.append(row)
    rows.append([B(("[OK] " if s["duration"]==d else "")+str(d)+"s", "dur", d) for d in DURATIONS])
    res_row = [{"text": ("[OK] " if s["resolution"]=="auto" else "")+"Auto", "callback_data": "cfg|res|auto"}]
    for w_,h_ in RESOLUTIONS:
        key=str(w_)+"x"+str(h_)
        res_row.append(B(("[OK] " if s["resolution"]==key else "")+key, "res", key))
    rows.append(res_row)
    rows.append([B(("[OK] " if s["gpu"]==g else "")+str(g)+"s", "gpu", g) for g in GPU_CHOICES])
    anch=round(float(s.get("anchor",1.0)),2); lst=round(float(s.get("lstr",0.8)),2)
    ar=[]
    for val,lbl in [(1.0,"Tight"),(0.9,"Balanced"),(0.8,"Loose")]:
        ar.append(B(("[OK] " if anch==val else "")+lbl, "anchor", str(val)))
    for v in [0.6,0.8,1.0]:
        ar.append(B(("[OK] " if lst==v else "")+"L"+str(v), "lstr", str(v)))
    rows.append(ar)
    rows.append([{"text": "Enhance: "+("ON" if s["enhance"] else "OFF"), "callback_data": "cfg|toggle_enh|x"}, {"text": "Random: "+("ON" if s["randomize"] else "OFF"), "callback_data": "cfg|toggle_rand|x"}, {"text": "Seed: "+("Random" if s["randomize"] else str(s["seed"])), "callback_data": "noop"}])
    rows.append([{"text": "RIFE: "+("ON" if s.get("rife") else "OFF"), "callback_data": "cfg|toggle_rife|x"}, {"text": "Upscale: "+("ON" if s.get("upscale") else "OFF"), "callback_data": "cfg|toggle_up|x"}])
    vo="OFF" if not s.get("vo_on") else ("Mix" if "Mix" in s.get("vo_mode","") else "Replace")
    rows.append([{"text": "VO: "+vo, "callback_data": "cfg|cycle_vo|x"}, {"text": s.get("vo_lang","English (US)"), "callback_data": "cfg|cycle_volang|x"}, {"text": "/vonarr", "callback_data": "noop"}])
    rows.append([{"text": "Reset all", "callback_data": "cfg|reset|x"}, {"text": "Back to Menu", "callback_data": "menu|home"}])
    return {"inline_keyboard": rows}
def render_settings_panel(s: dict) -> str:
    res_txt = ("Auto-fit (matches photo)" if s["resolution"] == "auto"
               else RESOLUTIONS.get(tuple(map(int, s["resolution"].split("x"))), s["resolution"]))
    return (
        "🎛 *Control Panel* — mirrors every Space UI control\n"
        f"├ 🎛 Camera LoRA: `{s['adapter'] or 'None'}`\n"
        f"├ ⏱ Duration: `{s['duration']}s`\n"
        f"├ 📐 Resolution: `{res_txt}`\n"
        f"├ ⚡️ GPU budget: `{s['gpu']}s`\n"
        f"├ 🧠 Prompt enhancement: `{'ON' if s['enhance'] else 'OFF'}`\n"
        f"├ 🎬 Camera intent: `{s['cam']}` ({s['spd']}, {s['amp']})\n"
        f"├ 🎯 Stage-1 anchor: `{s.get('anchor',1.0)}` • 🔗 LoRA str: `{s.get('lstr',0.8)}`\n"
        f"└ 🌱 Seed: `{'Random' if s['randomize'] else s['seed']}`\n\n"
        "_Tap buttons to change • applies to your next generation_\n"
        "_Caption tags still override for single runs._\n"
        "Set exact seed: `/seed 12345` • Text-only video: `/t2v prompt`"
    )

def cycle_adapter(cur):
    keys = list(ADAPTER_MAP.keys()) + ["none"]
    vals = [ADAPTER_MAP[k][0] for k in ADAPTER_MAP] + [None]
    idx = (vals.index(cur) + 1) % len(vals) if cur in vals else 0
    return vals[idx]

def handle_callback(cb: dict):
    msg = cb.get("message") or {}
    chat_id = msg.get("chat", {}).get("id")
    data = cb.get("data", "")
    if not chat_id or not data.startswith("cfg|"):
        return
    _, action, val = data.split("|", 2)
    s = get_settings(chat_id)
    valid_adapters = {v for _, v in ADAPTER_BUTTONS if v} | {None}
    if action == "adapter" and (val == "none" or val in valid_adapters):
        s["adapter"] = None if val == "none" else val
    elif action == "packsel" and val in [pk for _, pk in PACK_BUTTONS]:
        s["pack"] = val
    elif action == "anchor" and val.replace(".", "").isdigit():
        s["anchor"] = round(float(val), 2)
    elif action == "lstr" and val.replace(".", "").isdigit():
        s["lstr"] = round(float(val), 2)
    elif action == "cycle_adapter":
        s["adapter"] = cycle_adapter(s["adapter"])
    elif action == "cycle_cam":
        keys = list(CAMERA_MOVES.keys())
        s["cam"] = keys[(keys.index(s.get("cam", "none")) + 1) % len(keys)]
    elif action == "cycle_spd":
        keys = list(CAMERA_SPEEDS.keys())
        s["spd"] = keys[(keys.index(s.get("spd", "medium")) + 1) % len(keys)]
    elif action == "cycle_amp":
        keys = list(CAMERA_AMPS.keys())
        s["amp"] = keys[(keys.index(s.get("amp", "medium")) + 1) % len(keys)]
    elif action == "cycle_anchor":
        vals = [1.0, 0.9, 0.8]
        cur = round(float(s.get("anchor", 1.0)), 2)
        s["anchor"] = vals[(vals.index(cur) + 1) % len(vals)]
    elif action == "cycle_lstr":
        vals = [0.6, 0.8, 1.0]
        cur = round(float(s.get("lstr", 0.8)), 2)
        s["lstr"] = vals[(vals.index(cur) + 1) % len(vals)]
    elif action == "cycle_pack":
        packs = ["None", "Pose Control", "Union Control", "Detailer"]
        cur = s.get("pack", "None")
        s["pack"] = packs[(packs.index(cur) + 1) % len(packs)] if cur in packs else "None"
    elif action == "toggle_rife":
        s["rife"] = not s.get("rife", False)
    elif action == "toggle_up":
        s["upscale"] = not s.get("upscale", False)
    elif action == "cycle_vo":
        cur = 0 if not s.get("vo_on") else (2 if "Mix" in s.get("vo_mode", "") else 1)
        states = [(False, "Replace native audio"),
                  (True, "Replace native audio"),
                  (True, "Mix over native audio")]
        on, mode = states[(cur + 1) % 3]
        s["vo_on"], s["vo_mode"] = on, mode
    elif action == "cycle_volang":
        cur = s.get("vo_lang", "English (US)")
        s["vo_lang"] = (VO_LANGS[(VO_LANGS.index(cur) + 1) % len(VO_LANGS)]
                        if cur in VO_LANGS else VO_LANGS[0])
    elif action == "dur" and int(val) in DURATIONS:
        s["duration"] = int(val)
    elif action == "res" and (val == "auto" or val in {f"{w}x{h}" for (w, h) in RESOLUTIONS}):
        s["resolution"] = val
    elif action == "gpu" and int(val) in GPU_CHOICES:
        s["gpu"] = int(val)
    elif action == "toggle_enh":
        s["enhance"] = not s["enhance"]
    elif action == "toggle_rand":
        s["randomize"] = not s["randomize"]
    elif action == "reset":
        s.update(dict(DEFAULT_SETTINGS))
    save_settings()
    tg("editMessageText", chat_id=chat_id, message_id=msg["message_id"],
       text=render_settings_panel(s), parse_mode="Markdown",
       reply_markup=json.dumps(settings_keyboard(s)))

# -----------------------------------------------------------------------------
def _handle_gallery_pick(cb: dict):
    msg = cb.get("message") or {}
    chat_id = msg.get("chat", {}).get("id")
    try:
        idx = int(cb.get("data", "gal|0").split("|")[1])
        runs = _GALLERY_CACHE.get(chat_id) or vault_recent(12)
        run = runs[idx]
        url = run.get("url")
        cap = f"🎬 {run.get('prompt','')[:300]}\n🌱 {run.get('seed')} · 📐 {run.get('width')}x{run.get('height')} · {run.get('ts','')}"
        tg("sendMessage", chat_id=chat_id, text="📤 Fetching from vault…")
        r = _HF_S.get(url, headers=HF_AUTH, timeout=300)
        r.raise_for_status()
        send_video_file(chat_id, r.content, cap)
    except Exception as e:
        traceback.print_exc()
        send_message(chat_id, f"❌ Gallery fetch failed: `{e}`")

def _handle_preset_apply(cb: dict):
    msg = cb.get("message") or {}
    chat_id = msg.get("chat", {}).get("id")
    name = cb.get("data", "").split("|", 1)[1]
    p = vault_get_preset(name) or {}
    if not p:
        send_message(chat_id, f"⚠️ Preset '{name}' vanished.")
        return
    s = get_settings(chat_id)
    known = set(DEFAULT_SETTINGS.keys())
    applied = {k: v for k, v in p.items() if k in known}
    s.update(applied)
    save_settings()
    txt = (" · ".join(f"{k}={v}" for k, v in sorted(applied.items())))[:600]
    send_message(chat_id, f"📥 Applied preset *{name}*:\n`{txt}`\n\nOpen /settings to verify.")

def cmd_allow(msg):
    if not _require_admin(msg): return
    parts = (msg.get("text") or "").split()
    if len(parts) < 2 or not parts[1].lstrip("-").isdigit():
        send_message(msg["chat"]["id"], "Usage: `/allow <user_id>`")
        return
    uid = parts[1]
    if uid not in _ADMINS["allowed"]:
        _ADMINS["allowed"].append(uid)
        _save_admins(_ADMINS)
    send_message(msg["chat"]["id"], f"✅ Allowed {uid}")

def cmd_deny(msg):
    if not _require_admin(msg): return
    parts = (msg.get("text") or "").split()
    if len(parts) < 2:
        send_message(msg["chat"]["id"], "Usage: `/deny <user_id>`")
        return
    uid = parts[1]
    if uid in _ADMINS["allowed"]:
        _ADMINS["allowed"].remove(uid)
        _save_admins(_ADMINS)
    send_message(msg["chat"]["id"], f"⛔ Denied {uid}")

def cmd_whoami(msg):
    cid = str(msg["chat"]["id"])
    role = "👑 owner" if is_admin(cid) else ("✅ allowed" if cid in _ADMINS["allowed"] else "🚫 not allowed")
    send_message(msg["chat"]["id"], f"ID: `{cid}`\nRole: {role}")


def cmd_vonarr(msg):
    parts = (msg.get("text") or "").split(maxsplit=1)
    s_ = get_settings(msg["chat"]["id"])
    if len(parts) < 2 or not parts[1].strip():
        s_["vo_on"] = False
        save_settings()
        send_message(msg["chat"]["id"], "🎤 Voiceover cleared.")
        return
    s_["vo_text"] = parts[1].strip()
    s_["vo_on"] = True
    save_settings()
    send_message(msg["chat"]["id"],
                 f"🎤 Narration saved ({len(parts[1])} chars) — "
                 f"{s_.get('vo_lang')} · {s_.get('vo_mode')}.\n"
                 f"Applies to your next generation.")

def cmd_gallery(msg):
    chat_id = msg["chat"]["id"]
    runs = vault_recent(12)
    if not runs:
        send_message(chat_id, "🖼 Vault empty — generate something first!")
        return
    rows = [[{"text": f"🎬 {_clean_snip(r.get('prompt'))}", "callback_data": f"gal|{i}"}]
            for i, r in enumerate(runs)]
    _GALLERY_CACHE[chat_id] = runs
    send_message(chat_id, f"🖼 *Gallery* — last {len(runs)} renders:",
                 reply_markup={"inline_keyboard": rows + [[{"text": "🔄 Refresh",
                             "callback_data": "gallery"}]]})

def _clean_snip(t):
    return re.sub(r"[`\*_\[\]]", "", str(t or ""))[:42]

def cmd_psave(msg):
    if not _require_admin(msg): return
    parts = (msg.get("text") or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        send_message(msg["chat"]["id"], "Usage: `/psave Cinematic Portrait v2`")
        return
    name = parts[1].strip()[:64]
    ok, msgtxt = vault_save_preset(name, dict(get_settings(msg["chat"]["id"])))
    send_message(msg["chat"]["id"], ("✅ " if ok else "❌ ") + msgtxt)

# -----------------------------------------------------------------------------
# Vault (private HF Dataset) — mirrors the Space's free storage layer
# -----------------------------------------------------------------------------
def _vault_api():
    return HfApi(token=HF_TOKEN)

def vault_recent(n: int = 12):
    if not HAS_HUB:
        return []
    try:
        f = hf_hub_download(VAULT_REPO, "runs/metadata.jsonl",
                            repo_type="dataset", token=HF_TOKEN)
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        return list(reversed(rows[-n:]))
    except Exception as e:
        print(f"[vault] recent failed: {e}")
        return []

def vault_list_presets():
    if not HAS_HUB:
        return []
    try:
        f = hf_hub_download(VAULT_REPO, "presets.json", repo_type="dataset", token=HF_TOKEN)
        return list(json.load(open(f, encoding="utf-8")).keys())
    except Exception:
        return []

def vault_get_preset(name: str):
    if not HAS_HUB:
        return None
    try:
        f = hf_hub_download(VAULT_REPO, "presets.json", repo_type="dataset", token=HF_TOKEN)
        return json.load(open(f, encoding="utf-8")).get(name)
    except Exception:
        return None

def vault_save_preset(name: str, values: dict):
    if not HAS_HUB:
        return False, "huggingface_hub not installed"
    try:
        api = _vault_api()
        api.create_repo(VAULT_REPO, repo_type="dataset", private=True, exist_ok=True)
        try:
            f = hf_hub_download(VAULT_REPO, "presets.json", repo_type="dataset", token=HF_TOKEN)
            data = json.load(open(f, encoding="utf-8"))
        except Exception:
            data = {}
        data[name] = values
        import tempfile
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(data, tmp); tmp.close()
        api.upload_file(path_or_fileobj=tmp.name, path_in_repo="presets.json",
                        repo_id=VAULT_REPO, repo_type="dataset")
        os.unlink(tmp.name)
        return True, f"saved '{name}'"
    except Exception as e:
        return False, str(e)

# Jobs
# -----------------------------------------------------------------------------
JOBS: "queue.Queue" = queue.Queue()
_GALLERY_CACHE = {}
_LAST_JOB_TS = {}
_COOLDOWN_SEC = 60
_GLOBAL_QUEUE_CAP = 8
RUN_LOCK = threading.Lock()
RUNNING = None
CANCELLED = set()

def launch_job(chat_id: int, image_bytes, prompt: str, overrides: dict, source_note: str):
    s = dict(get_settings(chat_id))
    s.update({k: v for k, v in overrides.items() if v is not None})
    cam_sentence = compose_camera_sentence(s.get("cam"), s.get("spd", "medium"), s.get("amp", "medium"))
    if cam_sentence and prompt:
        prompt = f"{prompt.rstrip('.')} {cam_sentence}"
    w, h = resolve_resolution(image_bytes, s["resolution"])
    seed = s["seed"] if not s["randomize"] else 42
    job = {
        "job_id": uuid.uuid4().hex[:8], "chat_id": chat_id,
        "image": image_bytes, "prompt": prompt,
        "adapter": s["adapter"], "duration": float(s["duration"]),
        "width": w, "height": h, "gpu": s["gpu"],
        "enhance": bool(s["enhance"]), "seed": seed,
        "randomize": bool(s["randomize"]),
        "anchor": float(s.get("anchor", 1.0)),
        "lstr": float(s.get("lstr", 0.8)),
        "pack": s.get("pack", "None"),
        "rife": bool(s.get("rife", False)),
        "upscale": bool(s.get("upscale", False)),
        "vo_text": (s.get("vo_text", "") if s.get("vo_on") else ""),
        "vo_lang": s.get("vo_lang", "English (US)"),
        "vo_mix": s.get("vo_mode", "Replace native audio"),
    }
    job["summary"] = (f"{source_note} | {job['adapter'] or 'noLoRA'} | {int(job['duration'])}s "
                      f"| {w}x{h} | gpu{job['gpu']} | enh={job['enhance']}")

    # ---- abuse guards: per-chat cap + global cap + cooldown ----
    now = time.time()
    last = _LAST_JOB_TS.get(chat_id, 0)
    if now - last < _COOLDOWN_SEC:
        raise RuntimeError(f"cooldown: wait {int(_COOLDOWN_SEC-(now-last))}s")
    mine = sum(1 for j in list(JOBS.queue) if j["chat_id"] == chat_id)
    with RUN_LOCK:
        running_mine = RUNNING and RUNNING["chat_id"] == chat_id
    if mine + (1 if running_mine else 0) >= 2:
        raise RuntimeError("limit: max 1 queued + 1 running per user")
    if JOBS.qsize() >= _GLOBAL_QUEUE_CAP:
        raise RuntimeError("queue full, try later")
    _LAST_JOB_TS[chat_id] = now
    JOBS.put(job)
    return job

def _run_director(job: dict, chat_id: int, anim):
    """Render shots sequentially, stitch with ffmpeg, publish one album."""
    import subprocess, shutil
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install: pkg install ffmpeg")

    shots = job["shots"]
    base_seed = int(job["seed"])
    tmpdir = tempfile.mkdtemp(prefix="ltxdir_")
    parts, listing_lines = [], []

    try:
        anim.update("📤 Uploading anchor photo…" if job.get("image") else "📝 T2V director mode…", 0.03)
        remote = space_upload_image(job["image"]) if job.get("image") else None

        for i, (dur, txt) in enumerate(shots):
            if job["job_id"] in CANCELLED:
                raise RuntimeError("cancelled by user")
            seed_k = base_seed + i if job["randomize"] else base_seed
            sub = {
                "remote_path": remote, "prompt": txt,
                "adapter": job["adapter"], "duration": float(dur),
                "enhance": bool(job["enhance"]), "seed": seed_k,
                "randomize": False, "height": job["height"],
                "width": job["width"], "gpu": job["gpu"],
                "anchor": job.get("anchor", 1.0), "lstr": job.get("lstr", 0.8),
                "pack": job.get("pack", "None"),
                "rife": job.get("rife", False), "upscale": job.get("upscale", False),
                "vo_text": "", "vo_lang": "English (US)", "vo_mix": "Replace native audio",
                "summary": f"shot {i+1}/{len(shots)}",
            }
            stage_txt = f"🎬 Shot {i+1}/{len(shots)}: {txt[:48]}…"
            anim.update(stage_txt, 0.05 + 0.85 * (i / len(shots)))
            url, used_seed = space_generate(sub, anim=anim)
            part = os.path.join(tmpdir, f"shot_{i+1:02d}.mp4")
            r = _HF_S.get(url, headers=HF_AUTH, timeout=300); r.raise_for_status()
            with open(part, "wb") as f:
                f.write(r.content)
            parts.append(part)
            listing_lines.append(f"{i+1}. [{int(dur)}s] {txt}")
            print(f"[director] shot {i+1}/{len(shots)} done (seed {used_seed})")

        anim.update("🎞 Stitching timeline with ffmpeg…", 0.92)
        lst = os.path.join(tmpdir, "list.txt")
        with open(lst, "w") as f:
            for part in parts:
                f.write(f"file '{part}'\n")
        final = os.path.join(tmpdir, "final.mp4")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
               "-i", lst, "-c", "copy", final]
        rc = subprocess.run(cmd, capture_output=True).returncode
        if rc != 0 or not os.path.exists(final) or os.path.getsize(final) == 0:
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                   "-i", lst, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                   "-pix_fmt", "yuv420p", "-c:a", "aac", final]
            subprocess.run(cmd, check=True, capture_output=True)

        anim.update("📰 Publishing director's cut…", 0.97)
        with open(final, "rb") as f:
            video_bytes = f.read()
        meta = (f"🎬 Director: {len(shots)} shots | 📐 {job['width']}x{job['height']} | "
                f"🎛 {escape_html(job['adapter'] or 'base')} | 🧠 {'on' if job['enhance'] else 'off'}")
        caption_body = escape_html("\n".join(listing_lines))
        caption = f"🎬 *Director's Cut*\n{caption_body}\n\n{meta}\n🤖 @ltx2loras_bot"[:MAX_CAPTION]
        media = []
        files = {}
        attach_idx = 0
        if job.get("image"):
            media.append({"type": "photo", "media": "attach://file0"})
            files["file0"] = ("anchor.jpg", job["image"], "image/jpeg")
            attach_idx = 1
        media.append({"type": "video", "media": f"attach://file{attach_idx}",
                      "caption": caption, "parse_mode": "HTML"})
        files[f"file{attach_idx}"] = ("directors_cut.mp4", video_bytes, "video/mp4")
        rr = requests.post(f"{TG_API}/sendMediaGroup",
                           data={"chat_id": CHANNEL_ID, "media": json.dumps(media)},
                           files=files, timeout=300)
        if not rr.json().get("ok"):
            raise RuntimeError(f"album failed: {rr.text[:300]}")

        anim.close(f"🎬 *Director's cut published!* ({len(shots)} shots stitched)")
        send_video_file(chat_id, video_bytes,
                        f"🎬 Director's cut — {len(shots)} shots, ~{job['total_dur']:.0f}s")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def worker():
    global RUNNING
    while True:
        job = JOBS.get()
        chat_id = job["chat_id"]
        if job["job_id"] in CANCELLED:
            CANCELLED.discard(job["job_id"])
            JOBS.task_done()
            continue
        anim = LiveAnim(chat_id, "LTX-2 Video Generation")
        with RUN_LOCK:
            RUNNING = job
        video_bytes = None
        try:
            if job.get("type") == "director":
                _run_director(job, chat_id, anim)
                JOBS.task_done()
                continue
            if job["image"]:
                anim.update("📤 Uploading photo to the Space…", 0.03)
                job["remote_path"] = space_upload_image(job["image"])
            else:
                job["remote_path"] = None
                anim.update("📝 Text-to-video mode…", 0.03)
            video_url, seed = space_generate(job, anim=anim)
            job["seed"] = seed
            anim.update("📥 Downloading rendered MP4…", 0.96)
            video_bytes = space_download_video(video_url)
            anim.update("📰 Publishing album to channel…", 0.98)
            meta = (f"🎛 {escape_html(job['adapter'] or 'Distilled base')} | ⏱ {int(job['duration'])}s | "
                    f"📐 {job['width']}x{job['height']} | ⚡ {job['gpu']}s | 🌱 {seed} | "
                    f"🧠 {'on' if job['enhance'] else 'off'}")
            post_album_to_channel(job["image"] or b"", video_bytes,
                                  escape_html(job["prompt"]), meta)
            anim.close("✅ *Done!* Album published to the channel.")
            tg("sendVideo", chat_id=chat_id, video=("result.mp4", io.BytesIO(video_bytes)),
               caption=f"✅ Published! (seed {seed})", supports_streaming=True)
        except Exception as e:
            traceback.print_exc()
            anim.close(f"❌ *Failed:* `{e}`")
            try:
                send_message(chat_id, "Try again, or /status to inspect the Space.")
            except Exception:
                pass
        finally:
            with RUN_LOCK:
                RUNNING = None
            JOBS.task_done()

# -----------------------------------------------------------------------------
# Command handlers
# -----------------------------------------------------------------------------
def cmd_settings(msg):
    chat_id = msg["chat"]["id"]
    s = get_settings(chat_id)
    tg("sendMessage", chat_id=chat_id, text=render_settings_panel(s),
       parse_mode="Markdown", reply_markup=json.dumps(settings_keyboard(s)))

def cmd_seed(msg):
    parts = (msg.get("text") or "").split()
    if len(parts) < 2 or not parts[1].lstrip("-").isdigit():
        send_message(msg["chat"]["id"], "Usage: `/seed 12345` (or `/seed random`)")
        return
    s = get_settings(msg["chat"]["id"])
    if parts[1].lower() == "random":
        s["randomize"] = True
    else:
        s["randomize"] = False
        s["seed"] = max(0, min(int(parts[1]), 2**31 - 1))
    save_settings()
    send_message(msg["chat"]["id"],
                 f"🌱 Seed set to `{'Random' if s['randomize'] else s['seed']}`")

def cmd_t2v(msg):
    if not is_allowed(msg["chat"]["id"]):
        send_message(msg["chat"]["id"], "🔒 Private bot — ask the owner to allow you.")
        return
    text = (msg.get("text") or "").split(maxsplit=1)
    if len(text) < 2 or not text[1].strip():
        send_message(msg["chat"]["id"], "Usage: `/t2v A red car driving through neon city at night`")
        return
    job = launch_job(msg["chat"]["id"], None, text[1].strip(), {}, "T2V")
    send_message(msg["chat"]["id"],
                 f"✅ *Queued* (position {JOBS.qsize()}) — text-to-video\n"
                 f"🎛 `{job['adapter'] or 'no LoRA'}` | ⏱ `{int(job['duration'])}s` | 📐 `{job['width']}x{job['height']}`")

def cmd_examples(msg):
    chat_id = msg["chat"]["id"]
    kb = {"inline_keyboard": [[{"text": f"🎞 {EXAMPLES[i][0].split('.')[0].replace('_','-').title()}",
                                "callback_data": f"ex|{i}"}] for i in range(len(EXAMPLES))]}
    send_message(chat_id,
                 "🎞 *Examples Gallery*\nPick one to preview its prompt & run it "
                 "(consumes ZeroGPU quota):", kb)

def handle_example_cb(cb):
    msg = cb.get("message") or {}
    chat_id = msg.get("chat", {}).get("id")
    idx = int(cb["data"].split("|")[1])
    name, adapter, prompt = EXAMPLES[idx]
    kb = {"inline_keyboard": [[
        {"text": "▶️ Run this example", "callback_data": f"exrun|{idx}"},
        {"text": "❌ Close", "callback_data": "exclose|x"},
    ]]}
    tg("editMessageText", chat_id=chat_id, message_id=msg["message_id"],
       text=(f"🎞 *Example: {name}*\n🎛 `{adapter}`\n\n_“{prompt}”_\n\n"
             f"⚠️ Runs a real generation using your GPU quota."),
       parse_mode="Markdown", reply_markup=json.dumps(kb))

def handle_example_run(cb):
    msg = cb.get("message") or {}
    chat_id = msg.get("chat", {}).get("id")
    if not is_allowed(chat_id):
        send_message(chat_id, "🔒 Private bot — ask the owner to allow you.")
        return
    idx = int(cb["data"].split("|")[1])
    name, adapter, prompt = EXAMPLES[idx]
    try:
        img = fetch_example_image(name)
    except Exception as e:
        send_message(chat_id, f"❌ Couldn't fetch example image: `{e}`")
        return
    job = launch_job(chat_id, img, prompt, {"adapter": adapter}, f"example:{name}")
    tg("editMessageText", chat_id=chat_id, message_id=msg["message_id"],
       text=(f"✅ Example *{name}* queued (position {JOBS.qsize()})\n"
             f"🎛 `{adapter}` | 📐 `{job['width']}x{job['height']}`"),
       parse_mode="Markdown")

def cmd_reset(msg):
    _SETTINGS[str(msg["chat"]["id"])] = dict(DEFAULT_SETTINGS)
    save_settings()
    send_message(msg["chat"]["id"], "↺ Settings restored to Space defaults.")

# --- diagnostics commands -----------------------------------------------------
def cmd_status(msg): send_message(msg["chat"]["id"], render_status_card())

def cmd_live(msg):
    chat_id = msg["chat"]["id"]
    anim = LiveAnim(chat_id, "Live Space Monitor")
    end = time.time() + 40
    try:
        while time.time() < end:
            try:
                anim.update(render_status_card().replace("*", ""),
                            min(0.99, 1 - (end - time.time()) / 40))
            except Exception as e:
                anim.update(f"fetch error: {e}")
            time.sleep(6)
    finally:
        anim.close(f"🩺 Monitor closed.\n\n{render_status_card()}")

def cmd_ping(msg): send_message(msg["chat"]["id"], "🏓 *Ping*\n" + ping_targets())

def cmd_gpu(msg):
    send_message(msg["chat"]["id"],
                 f"⚡️ *ZeroGPU*\n├ Stage: `{space_status()['stage']}`\n└ gpu-counter: `{gpu_snapshot()}`")

def cmd_queue(msg):
    with RUN_LOCK:
        cur = RUNNING
    lines = [f"🗂 *Bot Queue* — waiting: {JOBS.qsize()}"]
    lines.append(f"▶️ Running: `{cur['summary']}`" if cur else "💤 Idle")
    send_message(msg["chat"]["id"], "\n".join(lines))

def cmd_cancel(msg):
    chat_id = msg["chat"]["id"]
    removed = sum(1 for j in list(JOBS.queue) if j["chat_id"] == chat_id
                  and (CANCELLED.add(j["job_id"]) or True))
    with RUN_LOCK:
        running = RUNNING and RUNNING["chat_id"] == chat_id
    text = f"🗑 Cancelled {removed} queued job(s)." if removed else "No queued jobs found."
    if running:
        text += "\n⚠️ Already rendering — cannot interrupt ZeroGPU mid-run."
    send_message(chat_id, text)

def _require_admin(msg) -> bool:
    if is_admin(msg["chat"]["id"]):
        return True
    send_message(msg["chat"]["id"], "🔒 Admin-only command.")
    return False

def cmd_refresh(msg):
    if not _require_admin(msg): return
    anim = LiveAnim(msg["chat"]["id"], "Refreshing Space")
    anim.update("🔄 Sending restart signal…", 0.4)
    r = _HF_S.post(f"{HF_API}/restart", headers=HF_AUTH, timeout=30)
    anim.close("✅ Restart accepted." if r.ok else f"❌ Rejected ({r.status_code}).")

def cmd_reboot(msg):
    if not _require_admin(msg): return
    anim = LiveAnim(msg["chat"]["id"], "Factory Rebuild")
    anim.update("🏭 Requesting fresh image build…", 0.4)
    r = _HF_S.post(f"{HF_API}/restart?factory=true", headers=HF_AUTH, timeout=30)
    anim.close("🏭 *Factory rebuild started.* (~5-10 min)\nCheck /status until `RUNNING`."
               if r.ok else f"❌ Rejected ({r.status_code}).")

def _logs_cmd(which):
    def h(msg):
        n = 15
        parts = (msg.get("text") or "").split()
        if len(parts) > 1 and parts[1].isdigit():
            n = min(int(parts[1]), 40)
        anim = LiveAnim(msg["chat"]["id"], f"Fetching {which} logs")
        anim.update("📜 Tailing…", 0.6)
        body = tail_logs(which, n)
        anim.close(f"📜 *Last lines ({which}):*\n```\n{body[-3500:]}\n```")
    return h

# -----------------------------------------------------------------------------
# Main Menu (/menu) — full button navigation to every feature
# -----------------------------------------------------------------------------
def menu_keyboard() -> dict:
    b = lambda label, action: {"text": label, "callback_data": f"menu|{action}"}
    return {"inline_keyboard": [
        [b("🎛 Control Panel", "settings"), b("🎞 Examples", "examples")],
        [b("🎬 Director Timeline", "director"), b("🖼 Gallery", "gallery")],
        [b("💾 Presets", "presets")],
        [b("📝 Text-to-Video", "t2v"),      b("🌱 Set Seed", "seed")],
        [b("🟢 Space Status", "status"),    b("📊 Live Monitor", "live")],
        [b("🏓 Ping", "ping"),              b("⚡️ ZeroGPU", "gpu")],
        [b("🗂 Queue", "queue"),            b("🗑 Cancel Jobs", "cancel")],
        [b("🔄 Refresh Space", "refresh"),  b("🏭 Factory Reboot", "reboot")],
        [b("📜 Runtime Logs", "logs"),      b("🏗 Build Logs", "buildlogs")],
        [b("❓ Help / Guide", "help")],
    ]}

MENU_BACK = {"inline_keyboard": [[{"text": "⬅️ Back to Menu", "callback_data": "menu|home"}]]}

def menu_text() -> str:
    return ("🏠 *LTX-2 Bot — Main Menu*\n"
            "Every Space feature, one tap away.\n\n"
            "_Tap a button • results appear right here_")

def send_menu(chat_id: int):
    tg("sendMessage", chat_id=chat_id, text=menu_text(),
       parse_mode="Markdown", reply_markup=json.dumps(menu_keyboard()))

def edit_menu(msg_id: int, chat_id: int, text: str | None = None,
              keyboard: dict | None = None):
    tg("editMessageText", chat_id=chat_id, message_id=msg_id,
       text=text if text is not None else menu_text(), parse_mode="Markdown",
       reply_markup=json.dumps(keyboard if keyboard is not None else menu_keyboard()))

def handle_menu_cb(cb: dict):
    msg = cb.get("message") or {}
    chat_id = msg.get("chat", {}).get("id")
    mid = msg.get("message_id")
    action = cb.get("data", "").split("|", 1)[1]

    if action == "home":
        edit_menu(mid, chat_id)
        return

    def _clean(t):
        return re.sub(r"[`\*_\[\]]", "", str(t or ""))[:46]

    if action == "gallery":
        runs = vault_recent(12)
        if not runs:
            edit_menu(mid, chat_id,
                      "🖼 *Gallery* — vault is empty.\nGenerate something first; "
                      "renders auto-save to your private HF dataset.", MENU_BACK)
            return
        rows, mapping = [], []
        for i, r in enumerate(runs):
            label = f"🎬 {_clean(r.get('prompt','(t2v)'))} · {r.get('seed','?')}"
            rows.append([{"text": label, "callback_data": f"gal|{i}"}])
        rows.append([{"text": "🔄 Refresh", "callback_data": "menu|gallery"},
                     {"text": "⬅️ Back to Menu", "callback_data": "menu|home"}])
        # stash rows for the gal| handler on this chat
        _GALLERY_CACHE[chat_id] = runs
        edit_menu(mid, chat_id, f"🖼 *Gallery* — last {len(runs)} renders.\nTap one to receive it here:",
                  {"inline_keyboard": rows})

    elif action == "presets":
        names = vault_list_presets()
        rows = [[{"text": f"📥 {n}", "callback_data": f"preset|{n}"}] for n in names]
        rows.append([{"text": "💾 Save current settings", "callback_data": "psavehint|x"}])
        rows.append([{"text": "⬅️ Back to Menu", "callback_data": "menu|home"}])
        edit_menu(mid, chat_id,
                  ("💾 *Presets* — tap to load:\n" if names else "💾 *Presets* — none saved yet.\n")
                  + "_Save anytime with_ `/psave <name>` _(snapshots your whole panel)_",
                  {"inline_keyboard": rows})

    elif action == "director":
        edit_menu(mid, chat_id,
                  "🎬 *Director Timeline*\n\n"
                  "`/director 3:first shot; 4:second shot; 2:last beat`\n\n"
                  "• `N:` = seconds per shot (1-10)\n"
                  "• Shots render in order with your panel settings, then "
                  "auto-stitch into ONE video\n"
                  "• 📸 Photo mode: caption starting `director: 3:a; 3:b` "
                  "anchors EVERY shot to your uploaded image (identity lock)", MENU_BACK)
        return

    if action == "settings":
        s = get_settings(chat_id)
        kb = settings_keyboard(s)
        kb["inline_keyboard"].append([{"text": "⬅️ Back to Menu",
                                       "callback_data": "menu|home"}])
        edit_menu(mid, chat_id, render_settings_panel(s), kb)

    elif action == "examples":
        kb = {"inline_keyboard":
              [[{"text": f"🎞 {EXAMPLES[i][0].split('.')[0].replace('_','-').title()}",
                 "callback_data": f"ex|{i}"}] for i in range(len(EXAMPLES))] +
              [{"text": "⬅️ Back to Menu", "callback_data": "menu|home"}]}
        edit_menu(mid, chat_id,
                  "🎞 *Examples Gallery*\nPick one to preview its prompt & run it "
                  "(consumes ZeroGPU quota):", kb)

    elif action == "t2v":
        edit_menu(mid, chat_id,
                  "📝 *Text-to-Video*\nSend me:\n`/t2v your prompt here`\n\n"
                  "Uses your Control Panel settings (no image needed).",
                  MENU_BACK)

    elif action == "seed":
        s = get_settings(chat_id)
        edit_menu(mid, chat_id,
                  f"🌱 Current seed mode: *{'Random' if s['randomize'] else s['seed']}*\n\n"
                  "Set an exact seed:\n`/seed 12345`\n"
                  "Back to random:\n`/seed random`",
                  MENU_BACK)

    elif action in ("status", "ping", "gpu", "queue"):
        try:
            if action == "status":
                out = render_status_card()
            elif action == "ping":
                anim = LiveAnim(chat_id, "Pinging targets", adopt_msg_id=mid)
                anim.update("🏓 Measuring…", 0.5)
                out = ping_targets()
                anim.close("🏓 *Ping*\n" + out)
                return
            elif action == "gpu":
                out = (f"⚡️ *ZeroGPU*\n├ Stage: `{space_status()['stage']}`\n"
                       f"└ gpu-counter: `{gpu_snapshot()}`")
            else:
                with RUN_LOCK:
                    cur = RUNNING
                out = (f"🗂 *Bot Queue* — waiting: {JOBS.qsize()}\n"
                       + (f"▶️ Running: `{cur['summary']}`" if cur else "💤 Idle"))
        except Exception as e:
            out = f"❌ {e}"
        edit_menu(mid, chat_id, out, MENU_BACK)

    elif action == "live":
        end = time.time() + 40
        anim = LiveAnim(chat_id, "Live Space Monitor", adopt_msg_id=mid)
        try:
            while time.time() < end:
                try:
                    anim.update(render_status_card().replace("*", ""),
                                min(0.99, 1 - (end - time.time()) / 40))
                except Exception as e:
                    anim.update(f"fetch error: {e}")
                time.sleep(6)
        finally:
            anim.close(f"🩺 Monitor closed.\n\n{render_status_card()}")
        edit_menu(mid, chat_id)   # restore the menu afterwards

    elif action == "cancel":
        removed = sum(1 for j in list(JOBS.queue) if j["chat_id"] == chat_id
                      and (CANCELLED.add(j["job_id"]) or True))
        with RUN_LOCK:
            running = RUNNING and RUNNING["chat_id"] == chat_id
        out = f"🗑 Cancelled {removed} queued job(s)." if removed else "No queued jobs found."
        if running:
            out += "\n⚠️ Already rendering — cannot interrupt ZeroGPU mid-run."
        edit_menu(mid, chat_id, out, MENU_BACK)

    elif action in ("refresh", "reboot"):
        label = "🔄 Restarting…" if action == "refresh" else "🏭 Factory rebuild…"
        url = f"{HF_API}/restart" + ("?factory=true" if action == "reboot" else "")
        anim = LiveAnim(chat_id, "Factory Rebuild" if action == "reboot" else "Refreshing Space",
                        adopt_msg_id=mid)
        anim.update(label, 0.4)
        r = _HF_S.post(url, headers=HF_AUTH, timeout=30)
        if r.ok and action == "reboot":
            anim.close("🏭 *Factory rebuild started.* (~5-10 min)\nCheck 🟢 Space Status until `RUNNING`.")
        elif r.ok:
            anim.close("✅ *Restart accepted.* Container is rebooting.")
        else:
            anim.close(f"❌ Rejected ({r.status_code}).")
        edit_menu(mid, chat_id)   # bring the menu back underneath

    elif action in ("logs", "buildlogs"):
        which = "run" if action == "logs" else "build"
        anim = LiveAnim(chat_id, f"Fetching {which} logs", adopt_msg_id=mid)
        anim.update("📜 Tailing…", 0.6)
        body = tail_logs(which, 15)
        anim.close(f"📜 *Last lines ({which}):*\n```\n{body[-3300:]}\n```")
        edit_menu(mid, chat_id)

    elif action == "help":
        edit_menu(mid, chat_id, WELCOME, MENU_BACK)

    elif action == "psavehint":
        edit_menu(mid, chat_id,
                  "💾 Send me:\n`/psave My Preset Name`\n\n"
                  "Snapshots: adapter • duration • resolution • GPU budget • "
                  "enhance • seed mode • camera intent • anchor • LoRA strength • skill pack",
                  MENU_BACK)

# -----------------------------------------------------------------------------
# Director Timeline mode (inspired by comfyui_ltx25_director)
#   /director 3:first shot prompt; 4:second shot prompt; 2:final beat
# Shots render sequentially with your current panel settings, then get stitched
# into ONE video with ffmpeg. Attach a photo with caption starting "director:"
# to condition every shot with that image (identity anchor).
# -----------------------------------------------------------------------------
def parse_director_spec(spec: str):
    """'3:a; 4:b' -> [(3,'a'),(4,'b')] ; duration clamped 1-10s."""
    shots = []
    for part in re.split(r";|\n", spec or ""):
        part = part.strip()
        m = re.match(r"(\d+(?:\.\d+)?)\s*[:：]\s*(.+)", part)
        if not m:
            continue
        dur = min(10.0, max(1.0, float(m.group(1))))
        txt = m.group(2).strip()
        if txt:
            shots.append((dur, txt))
    return shots

def cmd_director(msg):
    if not is_allowed(msg["chat"]["id"]):
        send_message(msg["chat"]["id"], "🔒 Private bot — ask the owner to allow you.")
        return
    chat_id = msg["chat"]["id"]
    body = (msg.get("text") or "").split(maxsplit=1)
    spec = body[1] if len(body) > 1 else ""
    shots = parse_director_spec(spec)
    if not shots:
        send_message(chat_id,
            "🎬 *Director Timeline*\n\n"
            "`/director 3:first shot; 4:second shot; 2:last beat`\n\n"
            "• Each `N:` sets that shot's length (1-10s)\n"
            "• Uses your /settings for camera LoRA + 🎬 camera intent\n"
            "• Shots render in order, then auto-stitch into one video\n"
            "• Photo mode: send an image with caption starting "
            "`director: 3:shot one; 3:shot two` to anchor every shot "
            "with that exact face/scene")
        return
    total = sum(d for d, _ in shots)
    job = launch_job(chat_id, None, "", {}, f"director:{len(shots)} shots")
    # rebuild as composite
    JOBS.get()  # remove the single placeholder just queued
    job.update({"shots": shots, "type": "director", "total_dur": total})
    job["summary"] = f"🎬 director | {len(shots)} shots | {total:.0f}s total"
    JOBS.put(job)
    listing = "\n".join(f"  {i+1}. `{d}s` — {t[:60]}" for i, (d, t) in enumerate(shots))
    send_message(chat_id,
                 f"🎬 *Director timeline queued* (position {JOBS.qsize()})\n"
                 f"{len(shots)} shots • ~{total:.0f}s footage (+render time)\n{listing}")

COMMANDS = {
    "/start": lambda m: (claim_owner(m["chat"]["id"]),
                         tg("sendMessage", chat_id=m["chat"]["id"],
                            text=WELCOME + ("\n\n👑 You are the *owner* of this bot." 
                                            if str(m["chat"]["id"]) in _ADMINS["admins"] else ""),
                            parse_mode="Markdown",
                            reply_markup=json.dumps(menu_keyboard()))),
    "/help":  lambda m: tg("sendMessage", chat_id=m["chat"]["id"], text=WELCOME,
                           parse_mode="Markdown",
                           reply_markup=json.dumps(menu_keyboard())),
    "/menu":  lambda m: send_menu(m["chat"]["id"]),
    "/id":    lambda m: send_message(m["chat"]["id"], f"`{m['chat']['id']}`"),
    "/settings": cmd_settings,
    "/director": cmd_director,
    "/allow": cmd_allow,
    "/deny": cmd_deny,
    "/whoami": cmd_whoami,
    "/gallery":  cmd_gallery,
    "/vonarr":   cmd_vonarr,
    "/psave":    cmd_psave,
    "/psave":    cmd_psave,
    "/seed":     cmd_seed,
    "/t2v":      cmd_t2v,
    "/examples": cmd_examples,
    "/reset":    cmd_reset,
    "/status":   cmd_status,
    "/live":     cmd_live,
    "/ping":     cmd_ping,
    "/gpu":      cmd_gpu,
    "/queue":    cmd_queue,
    "/cancel":   cmd_cancel,
    "/refresh":  cmd_refresh,
    "/reboot":   cmd_reboot,
    "/logs":     _logs_cmd("run"),
    "/buildlogs": _logs_cmd("build"),
}

# -----------------------------------------------------------------------------
# Update routing
# -----------------------------------------------------------------------------
def _route_callback(cb: dict):
    """Runs in its own thread; callback already answered centrally."""
    try:
        data = cb.get("data", "")
        if data.startswith("cfg|"):
            handle_callback(cb)
        elif data.startswith("ex|"):
            handle_example_cb(cb)
        elif data.startswith("exrun|"):
            handle_example_run(cb)
        elif data.startswith("menu|"):
            handle_menu_cb(cb)
    except Exception:
        traceback.print_exc()

def process_update(update: dict):
    """Runs in its own thread so long jobs never block polling."""
    try:
        if "callback_query" in update:
            cb = update["callback_query"]
            # answer INSTANTLY (Telegram kills callbacks after ~15s)
            tg("answerCallbackQuery", callback_query_id=cb.get("id"))
            _route_callback(cb)
            return

        msg = update.get("message") or update.get("channel_post")
        if not msg:
            return
        text = msg.get("text") or ""

        if text.startswith("/"):
            handler = COMMANDS.get(text.split()[0].split("@")[0].lower())
            handler(msg) if handler else send_message(msg["chat"]["id"], "Unknown command. /help")
            return

        photos = msg.get("photo")
        doc = msg.get("document") or {}
        is_doc = doc.get("mime_type", "").startswith("image/")
        if photos or is_doc:
            anim = LiveAnim(msg["chat"]["id"], "Receiving Photo")
            anim.update("📸 Downloading photo from Telegram…", 0.5)
            if is_doc and (not photos or doc.get("file_size", 0) > 200_000):
                file_id, note = doc["file_id"], "📄 uncompressed"
            else:
                largest = max(photos, key=lambda p: p.get("width", 0) * p.get("height", 0))
                file_id, note = largest["file_id"], "🖼 compressed"
            raw_cap = msg.get("caption") or ""
            director_mode = bool(re.match(r"\s*director\s*[:：]", raw_cap, re.I))
            if director_mode:
                raw_cap = re.split(r"[:：]", raw_cap, maxsplit=1)[1]
            prompt_clean, overrides = parse_caption(raw_cap)
            image = download_tg_file(file_id)
            # access check before spending quota
            if not is_allowed(msg["chat"]["id"]):
                anim.close("🔒 This bot is private. Ask the owner to /allow your ID.")
                return
            if director_mode:
                shots = parse_director_spec(prompt_clean)
                if not shots:
                    anim.close("⚠️ No valid shots found.\nFormat: `director: 3:first; 3:second`")
                    return
                base = launch_job(msg["chat"]["id"], image, "", overrides, note)
                JOBS.get()
                total = sum(d for d, _ in shots)
                base.update({"shots": shots, "type": "director", "total_dur": total,
                             "summary": f"🎬 photo-director | {len(shots)} shots"})
                JOBS.put(base)
                anim.close(f"🎬 *Photo-director queued* ({len(shots)} shots, ~{total:.0f}s)\n"
                           f"Every shot anchored to your photo — identity preserved.")
                return
            try:
                job = launch_job(msg["chat"]["id"], image, prompt_clean or "Cinematic subtle motion, gentle camera drift",
                                 overrides, note)
            except RuntimeError as e:
                anim.close(f"⏳ {e}")
                return
            w, h = job["width"], job["height"]
            orient = "portrait" if h > w else ("landscape" if w > h else "square")
            anim.close(f"✅ {note} photo ({len(image)//1024} KB)\n"
                       f"✅ *Queued* (pos {JOBS.qsize()}) 🆔 `{job['job_id']}`\n"
                       f"📐 `{w}x{h}` ({orient}) | 🎛 `{job['adapter'] or 'none'}` | "
                       f"⏱ `{int(job['duration'])}s` | ⚡ `{job['gpu']}s` | "
                       f"🧠 `{'on' if job['enhance'] else 'off'}` | 🌱 `{'rnd' if job['randomize'] else job['seed']}`\n\n"
                       f"_“{job['prompt'][:260]}”_")
            return

        if text:
            send_message(msg["chat"]["id"],
                         "📸 Send a *photo* (caption = prompt), /t2v for text-only, or /settings for the control panel.")

    except Exception:
        traceback.print_exc()

def main():
    load_settings()
    threading.Thread(target=worker, daemon=True).start()
    me = tg("getMe").get("result") or {}
    print(f"🤖 @{me.get('username')} online — full Space UI replica. (Ctrl+C to stop)")
    offset = None
    while True:
        try:
            params = {"timeout": 25, "allowed_updates": ["message", "channel_post", "callback_query"]}
            if offset is not None:
                params["offset"] = offset
            r = _TG_S.get(f"{TG_API}/getUpdates", params=params, timeout=40)
            updates = (r.json() or {}).get("result", [])
            for update in updates:
                offset = update["update_id"] + 1
                # concurrent dispatch: never block the poll loop
                threading.Thread(target=process_update, args=(update,), daemon=True).start()
        except requests.RequestException as e:
            print(f"[poll] network hiccup: {e}")
            time.sleep(5)
        except Exception as e:
            # never let the polling loop die (bad JSON, etc.)
            print(f"[poll] unexpected: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
