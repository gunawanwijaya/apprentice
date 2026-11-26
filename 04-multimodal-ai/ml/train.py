import base64
import hashlib
import io
import json
import random
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import kagglehub
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFilter
from scipy.signal import resample_poly
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parent.parent
CONTRACT_PATH = ROOT_DIR / "feature_contract.json"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_DIR = Path(__file__).resolve().parent / "data"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_contract():
    contract = json.loads(CONTRACT_PATH.read_text())
    canon = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    schema_hash = hashlib.sha256(canon.encode()).hexdigest()[:12]
    return contract, schema_hash


# --- Deterministic Preprocessing (Mirrored 1:1 in Pure Go) ---
def image_to_chw_tensor(img: Image.Image, contract: dict) -> np.ndarray:
    v = contract["vision"]
    if img.size != (v["width"], v["height"]):
        img = img.resize((v["width"], v["height"]), Image.Resampling.BILINEAR)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0  # [H, W, 3]
    mean = np.array(v["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.array(v["std"], dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)  # [3, H, W]


def pcm_to_audio_features(pcm_i16: np.ndarray, contract: dict) -> np.ndarray:
    """Extracts [num_frames, 2] acoustic features (log_rms, zcr) matching Go bit-for-bit."""
    a = contract["audio"]
    sr = a["sample_rate"]
    frame_len = (sr * a["frame_ms"]) // 1000
    hop_len = (sr * a["hop_ms"]) // 1000
    num_frames = a["num_frames"]

    samples = pcm_i16.astype(np.float32) / 32768.0
    feats = np.zeros((num_frames, 2), dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_len
        end = start + frame_len
        if start >= len(samples):
            frame = np.zeros(frame_len, dtype=np.float32)
        elif end > len(samples):
            frame = np.pad(samples[start:], (0, end - len(samples)))
        else:
            frame = samples[start:end]

        rms = float(np.sqrt(np.mean(frame * frame) + 1e-8))
        log_rms = float(np.log(rms + 1e-6))
        signs = frame >= 0.0
        zcr = float(np.mean(signs[1:] != signs[:-1]))
        feats[i, 0] = log_rms
        feats[i, 1] = zcr

    return feats


def encode_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_wav_b64(pcm_i16: np.ndarray, sample_rate: int) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.astype("<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_wav_to_pcm_i16(wav_path: Path, target_sr: int, max_samples: int) -> np.ndarray | None:
    try:
        with wave.open(str(wav_path), "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != target_sr:
                return None
            frames = wf.readframes(max_samples)
            arr = np.frombuffer(frames, dtype="<i2").astype(np.int16)
            if len(arr) < 1600:
                return None
            if len(arr) < max_samples:
                arr = np.pad(arr, (0, max_samples - len(arr)))
            return arr[:max_samples]
    except Exception:
        return None


def load_pcm_16k_mono(audio_path: Path, target_sr: int, max_samples: int) -> np.ndarray | None:
    """Loads WAV/MP3/FLAC/OGG and returns 16kHz mono int16 samples of length max_samples."""
    pcm = decode_wav_to_pcm_i16(audio_path, target_sr, max_samples)
    if pcm is not None:
        return pcm

    try:
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if sr != target_sr and sr > 0:
            g = int(np.gcd(int(sr), int(target_sr)))
            mono = resample_poly(mono, target_sr // g, sr // g)
        if len(mono) < 1600:
            return None
        pcm_i16 = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
        if len(pcm_i16) < max_samples:
            pcm_i16 = np.pad(pcm_i16, (0, max_samples - len(pcm_i16)))
        return pcm_i16[:max_samples]
    except Exception:
        pass

    if shutil.which("ffmpeg"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            cmd = [
                "ffmpeg", "-y", "-i", str(audio_path),
                "-vn", "-ac", "1", "-ar", str(target_sr),
                "-f", "wav", tmp.name, "-loglevel", "error",
            ]
            if subprocess.run(cmd, check=False).returncode == 0:
                return decode_wav_to_pcm_i16(Path(tmp.name), target_sr, max_samples)

    return None


def try_download_kaggle(slug: str, subdir: str) -> Path | None:
    local_dir = DATA_DIR / subdir
    if local_dir.exists() and any(local_dir.rglob("*")):
        return local_dir
    try:
        return Path(kagglehub.dataset_download(slug))
    except Exception as exc:
        print(f"[info] Kaggle dataset '{slug}' unavailable ({exc}); using synthetic fallback")
        return None


def apply_quality_distortion(base_img: Image.Image, label: str, rng: random.Random) -> Image.Image:
    w, h = base_img.size
    img = base_img.resize((w, h), Image.Resampling.BILINEAR).convert("RGB")
    if label == "blurry":
        return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(2.5, 4.2)))
    if label == "heavy_glare":
        out = img.copy()
        draw = ImageDraw.Draw(out)
        cx, cy = rng.randint(w // 4, (3 * w) // 4), rng.randint(h // 4, (3 * h) // 4)
        r = rng.randint(max(12, w // 6), max(20, w // 3))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255))
        return out.filter(ImageFilter.GaussianBlur(radius=1.0))
    if label == "low_light":
        arr = (np.asarray(img, dtype=np.float32) * rng.uniform(0.15, 0.28)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    return img


# --- Synthetic Fallbacks ---
def make_synthetic_ekyc_dataset(contract: dict, n_per_class: int = 160):
    rng = random.Random(42)
    w, h = contract["vision"]["width"], contract["vision"]["height"]
    images, labels = [], []

    for cls_idx, label in enumerate(contract["vision"]["ekyc_labels"]):
        for _ in range(n_per_class):
            bg = (rng.randint(170, 215), rng.randint(195, 230), rng.randint(215, 245))
            img = Image.new("RGB", (w, h), bg)
            draw = ImageDraw.Draw(img)
            draw.rectangle([8, 12, w - 8, h - 12], outline=(30, 60, 110), width=2)
            draw.rectangle([14, 18, w - 14, 28], fill=(25, 85, 160))
            draw.rectangle([w - 40, 34, w - 14, 72], fill=(180, 50, 50))
            for row in range(4):
                y = 38 + row * 14
                line_w = rng.randint(40, 62)
                draw.rectangle([16, y, 16 + line_w, y + 4], fill=(20, 25, 35))
            images.append(apply_quality_distortion(img, label, rng))
            labels.append(cls_idx)

    return images, np.array(labels, dtype=np.int64)


def make_synthetic_unboxing_dataset(contract: dict, n_per_class: int = 160):
    rng = random.Random(101)
    w, h = contract["vision"]["width"], contract["vision"]["height"]
    images, labels = [], []

    for cls_idx, label in enumerate(contract["vision"]["unboxing_labels"]):
        for _ in range(n_per_class):
            img = Image.new("RGB", (w, h), (rng.randint(215, 240), rng.randint(215, 240), rng.randint(210, 235)))
            draw = ImageDraw.Draw(img)
            box_color = (rng.randint(175, 205), rng.randint(130, 160), rng.randint(80, 105))
            draw.rectangle([18, 22, w - 18, h - 18], fill=box_color, outline=(110, 75, 40), width=2)
            draw.rectangle([w // 2 - 6, 22, w // 2 + 6, h - 18], fill=(225, 200, 145))

            if label == "damaged_box":
                draw.polygon([(18, 22), (62, 22), (38, 66), (18, 52)], fill=(55, 35, 20))
                draw.line([(28, 30), (92, 96)], fill=(35, 20, 10), width=5)
                draw.line([(45, 28), (105, 82)], fill=(45, 25, 15), width=4)
            elif label == "empty_box":
                draw.rectangle([28, 32, w - 28, h - 28], fill=(28, 24, 22), outline=(90, 65, 40), width=2)
            elif label == "blurry_frame":
                img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(2.8, 4.2)))

            images.append(img)
            labels.append(cls_idx)

    return images, np.array(labels, dtype=np.int64)


def make_synthetic_audio_dataset(contract: dict, n_per_class: int = 240):
    rng = np.random.default_rng(77)
    sr = contract["audio"]["sample_rate"]
    n_samples = sr + 400
    t = np.arange(n_samples, dtype=np.float32) / sr
    waves_out, labels = [], []

    for cls_idx, label in enumerate(contract["audio"]["labels"]):
        for _ in range(n_per_class):
            if label == "silence_or_noise":
                amp = rng.uniform(0.001, 0.02)
                sig = rng.normal(0.0, amp, size=n_samples).astype(np.float32)
            else:
                f0 = rng.uniform(110.0, 260.0)
                env = (0.5 + 0.5 * np.sin(2.0 * np.pi * rng.uniform(2.5, 5.5) * t)).astype(np.float32)
                sig = (
                    0.35 * np.sin(2.0 * np.pi * f0 * t)
                    + 0.20 * np.sin(2.0 * np.pi * (2.0 * f0) * t)
                    + 0.10 * np.sin(2.0 * np.pi * (3.0 * f0) * t)
                ) * env + rng.normal(0.0, 0.01, size=n_samples)
                sig = np.clip(sig, -0.95, 0.95).astype(np.float32)

            pcm_i16 = np.clip(sig * 32767.0, -32768, 32767).astype(np.int16)
            waves_out.append(pcm_i16)
            labels.append(cls_idx)

    return waves_out, np.array(labels, dtype=np.int64)


# --- Real Dataset Loaders (With Automatic Synthetic Fallback) ---
def load_ekyc_dataset(contract: dict, n_per_class: int = 160):
    rng = random.Random(42)
    w, h = contract["vision"]["width"], contract["vision"]["height"]
    src_root = try_download_kaggle("urbikn/sroie-datasetv2", "ekyc")
    img_paths = []
    if src_root is not None:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            img_paths.extend(sorted(src_root.rglob(ext)))

    if len(img_paths) >= 40:
        print(f"Using real document images ({len(img_paths)} files) for eKYC quality dataset")
        images, labels = [], []
        for cls_idx, label in enumerate(contract["vision"]["ekyc_labels"]):
            for i in range(n_per_class):
                p = img_paths[(cls_idx * n_per_class + i) % len(img_paths)]
                try:
                    with Image.open(p) as im:
                        base = im.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
                    images.append(apply_quality_distortion(base, label, rng))
                    labels.append(cls_idx)
                except Exception:
                    continue
        if len(images) >= 80:
            return images, np.array(labels, dtype=np.int64)

    return make_synthetic_ekyc_dataset(contract, n_per_class)


def load_unboxing_dataset(contract: dict, n_per_class: int = 160):
    rng = random.Random(101)
    w, h = contract["vision"]["width"], contract["vision"]["height"]
    src_root = try_download_kaggle("ayesha19765/damaged-package-detection", "unboxing")
    img_paths = []
    if src_root is not None:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            img_paths.extend(sorted(src_root.rglob(ext)))

    if len(img_paths) >= 20:
        print(f"Using real package images ({len(img_paths)} files) for Unboxing dataset")
        damaged_pool = [p for p in img_paths if "damage" in str(p).lower() or "defect" in str(p).lower()]
        intact_pool = [p for p in img_paths if "intact" in str(p).lower() or "normal" in str(p).lower() or "good" in str(p).lower()]
        if not damaged_pool:
            damaged_pool = img_paths
        if not intact_pool:
            intact_pool = img_paths

        images, labels = [], []
        for cls_idx, label in enumerate(contract["vision"]["unboxing_labels"]):
            pool = damaged_pool if label == "damaged_box" else (intact_pool if label == "intact_parcel" else img_paths)
            for i in range(n_per_class):
                p = pool[(cls_idx * n_per_class + i) % len(pool)]
                try:
                    with Image.open(p) as im:
                        base = im.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
                    draw = ImageDraw.Draw(base)
                    if label == "damaged_box" and "damage" not in str(p).lower() and "defect" not in str(p).lower():
                        draw.polygon([(10, 10), (46, 10), (26, 48), (10, 36)], fill=(55, 35, 20))
                        draw.line([(18, 18), (86, 90)], fill=(35, 20, 10), width=4)
                    elif label == "empty_box":
                        draw.rectangle([22, 26, w - 22, h - 26], fill=(24, 22, 20), outline=(90, 65, 40), width=2)
                    elif label == "blurry_frame":
                        base = base.filter(ImageFilter.GaussianBlur(radius=rng.uniform(2.8, 4.2)))
                    images.append(base)
                    labels.append(cls_idx)
                except Exception:
                    continue
        if len(images) >= 80:
            return images, np.array(labels, dtype=np.int64)

    return make_synthetic_unboxing_dataset(contract, n_per_class)


def load_audio_dataset(contract: dict, n_per_class: int = 240):
    sr = contract["audio"]["sample_rate"]
    max_samples = sr + 400
    rng = np.random.default_rng(77)
    src_root = try_download_kaggle("jayjoshi37/deepfake-audio-dataset-fake-vs-real-speech", "audio")
    audio_paths = []
    if src_root is not None:
        for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg"):
            audio_paths.extend(sorted(src_root.rglob(ext)))

    if len(audio_paths) >= 40:
        real_paths = [p for p in audio_paths if "real" in str(p).lower()]
        speech_pool = real_paths if len(real_paths) >= 20 else audio_paths
        print(f"Using real speech audio clips ({len(speech_pool)} files) for Audio VAD")

        waves_out, labels = [], []
        for cls_idx, label in enumerate(contract["audio"]["labels"]):
            for i in range(n_per_class):
                if label == "speech_present":
                    pcm = load_pcm_16k_mono(speech_pool[i % len(speech_pool)], sr, max_samples)
                    if pcm is not None:
                        waves_out.append(pcm)
                        labels.append(cls_idx)
                else:
                    amp = float(rng.uniform(0.001, 0.015))
                    sig = rng.normal(0.0, amp, size=max_samples).astype(np.float32)
                    waves_out.append(np.clip(sig * 32767.0, -32768, 32767).astype(np.int16))
                    labels.append(cls_idx)

        if len(waves_out) >= 120:
            return waves_out, np.array(labels, dtype=np.int64)

    return make_synthetic_audio_dataset(contract, n_per_class)


# --- Compact 2D-CNN (Vision) & 1D-CNN (Audio VAD) ---
class CompactVisionCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CompactAudioVAD(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.head = nn.Linear(32, num_classes)

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).mean(dim=2)
        return self.head(h)


def train_and_export(name: str, model: nn.Module, X_tr, y_tr, X_te, y_te, onnx_path: Path, input_name: str, dummy_shape, epochs: int = 15):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=32, shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            loss_fn(model(bx), by).backward()
            opt.step()

    model.eval()
    dummy = torch.zeros(dummy_shape, dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=[input_name],
        output_names=["logits"],
        dynamic_axes={input_name: {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14,
        dynamo=False,
    )

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    preds, times = [], []
    for i in range(len(y_te)):
        t0 = time.perf_counter()
        logits = sess.run(["logits"], {input_name: X_te[i : i + 1]})[0]
        times.append((time.perf_counter() - t0) * 1000)
        preds.append(int(np.argmax(logits[0])))

    f1 = float(f1_score(y_te, preds, average="macro"))
    p50_ms = float(np.median(times))
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"{name:<22} | Macro-F1: {f1:.4f} | CPU p50: {p50_ms:6.3f} ms | Size: {size_mb:5.2f} MB")
    return sess, {"model": name, "macro_f1": round(f1, 4), "cpu_p50_ms": round(p50_ms, 3), "size_mb": round(size_mb, 3)}


def main():
    set_seed(42)
    contract, schema_hash = load_contract()
    v, a = contract["vision"], contract["audio"]
    results = []

    print(f"Multimodal Training | Vision: {v['width']}x{v['height']} RGB CHW | Audio: {a['sample_rate']}Hz {a['num_frames']} frames\n")

    # 1. eKYC Vision Model
    ekyc_imgs, ekyc_y = load_ekyc_dataset(contract)
    ekyc_X = np.stack([image_to_chw_tensor(im, contract) for im in ekyc_imgs])
    idx_tr, idx_te = train_test_split(np.arange(len(ekyc_y)), test_size=0.2, stratify=ekyc_y, random_state=42)
    ekyc_sess, res = train_and_export(
        "eKYC Vision (2D-CNN)",
        CompactVisionCNN(len(v["ekyc_labels"])),
        ekyc_X[idx_tr], ekyc_y[idx_tr], ekyc_X[idx_te], ekyc_y[idx_te],
        ARTIFACTS_DIR / "ekyc_vision.onnx",
        "image", (1, 3, v["height"], v["width"]),
    )
    results.append(res)

    ekyc_cases = []
    for idx in idx_te[:20]:
        logits = ekyc_sess.run(["logits"], {"image": ekyc_X[idx : idx + 1]})[0][0].tolist()
        ekyc_cases.append({
            "png_b64": encode_png_b64(ekyc_imgs[idx]),
            "expected_logits": [round(x, 6) for x in logits],
            "expected_label": v["ekyc_labels"][int(np.argmax(logits))],
        })

    # 2. Unboxing Vision Model
    unb_imgs, unb_y = load_unboxing_dataset(contract)
    unb_X = np.stack([image_to_chw_tensor(im, contract) for im in unb_imgs])
    u_tr, u_te = train_test_split(np.arange(len(unb_y)), test_size=0.2, stratify=unb_y, random_state=42)
    unb_sess, res = train_and_export(
        "Unboxing Vision (CNN)",
        CompactVisionCNN(len(v["unboxing_labels"])),
        unb_X[u_tr], unb_y[u_tr], unb_X[u_te], unb_y[u_te],
        ARTIFACTS_DIR / "unboxing_vision.onnx",
        "image", (1, 3, v["height"], v["width"]),
    )
    results.append(res)

    unb_cases = []
    for idx in u_te[:20]:
        logits = unb_sess.run(["logits"], {"image": unb_X[idx : idx + 1]})[0][0].tolist()
        unb_cases.append({
            "png_b64": encode_png_b64(unb_imgs[idx]),
            "expected_logits": [round(x, 6) for x in logits],
            "expected_label": v["unboxing_labels"][int(np.argmax(logits))],
        })

    # 3. Audio VAD Model
    wavs, aud_y = load_audio_dataset(contract)
    aud_X = np.stack([pcm_to_audio_features(w, contract) for w in wavs])
    a_tr, a_te = train_test_split(np.arange(len(aud_y)), test_size=0.2, stratify=aud_y, random_state=42)
    aud_sess, res = train_and_export(
        "Audio VAD (1D-CNN)",
        CompactAudioVAD(len(a["labels"])),
        aud_X[a_tr], aud_y[a_tr], aud_X[a_te], aud_y[a_te],
        ARTIFACTS_DIR / "audio_vad.onnx",
        "audio_features", (1, a["num_frames"], 2),
    )
    results.append(res)

    audio_cases = []
    for idx in a_te[:20]:
        logits = aud_sess.run(["logits"], {"audio_features": aud_X[idx : idx + 1]})[0][0].tolist()
        audio_cases.append({
            "wav_b64": encode_wav_b64(wavs[idx], a["sample_rate"]),
            "expected_logits": [round(x, 6) for x in logits],
            "expected_label": a["labels"][int(np.argmax(logits))],
        })

    (ARTIFACTS_DIR / "benchmark_summary.json").write_text(json.dumps(results, indent=2))
    golden = {
        "schema_version": contract["schema_version"],
        "schema_hash": schema_hash,
        "ekyc_cases": ekyc_cases,
        "unboxing_cases": unb_cases,
        "audio_cases": audio_cases,
    }
    (ARTIFACTS_DIR / "golden_fixtures.json").write_text(json.dumps(golden, indent=2))
    print(f"\nExported 3 ONNX models + 60 multimodal golden fixtures (schema_hash={schema_hash})")


if __name__ == "__main__":
    main()
