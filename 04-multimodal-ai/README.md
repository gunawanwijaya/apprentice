# 04 — Multimodal AI: Video, Vision & Audio (PyTorch -> ONNX -> Concurrent Go)

Production-grade **Multimodal Inference Service** in Go that ingests raw video (`.mp4`), splits it into **sampled RGB keyframes** and **16 kHz mono PCM audio**, drops redundant static frames, and executes **2D-CNN Vision** and **1D-CNN Audio VAD** ONNX models **concurrently via `errgroup`**—locked to a single `feature_contract.json` and verified by a **60/60 Python-to-Go parity test suite (`0.15s`)**.

![Multimodal Benchmark Summary](./artifacts/multimodal_benchmark_summary.png)

---

## 1. Why Local Multimodal Gates Beat Raw VLM / STT Calls

Sending raw user-uploaded videos or high-resolution document scans straight to frontier Vision-Language Models (VLMs) or cloud Speech-to-Text APIs fails in production for two reasons:

1. **Unit Economics & Bad-Input Waste:** Roughly 25–40% of real-world mobile uploads suffer from motion blur, flash glare, darkness, closed/intact boxes, or muted microphones. Rejecting broken media in **`< 5 ms` on cheap CPU** avoids paying `$0.005–$0.05` per upload on garbage inputs.
2. **Data Residency & Compliance:** Identity documents (e.g., Indonesian KTPs) and financial verification clips contain sensitive PII. Running ingress quality gates locally inside your VPC ensures unreadable uploads are rejected immediately without leaving on-prem infrastructure.

This shared Go engine powers two APAC backend workflows on dedicated routes:
* **Fintech Video eKYC Gate (`POST /ekyc/video`):** Checks document frame quality (`clean_document`, `blurry`, `heavy_glare`, `low_light`) in parallel with Voice Activity Detection (`speech_present` vs. `silence_or_noise`) while a user reads an OTP.
* **E-Commerce "Wajib Video Unboxing" Triage (`POST /unboxing/video`):** Samples 1 FPS keyframes, deduplicates static footage via cosine similarity, classifies parcel state (`intact_parcel`, `damaged_box`, `empty_box`, `blurry_frame`), and checks whether spoken buyer narration is present.

---

## 2. End-to-End Architecture & Zero-Skew Contract

```text
feature_contract.json  (Single Source of Truth: Vision 128x128 CHW + Audio 16kHz + Video 1 FPS)
 ├── ml/train.py       -> trains on 3 real Kaggle datasets, exports 3 ONNX models (<0.20 MB total)
 │                        + 60 base64 PNG/WAV golden fixtures (schema_hash=559a7112982a)
 └── go generate       -> emits internal/contract/contract_gen.go + SHA-256 SchemaHash gate
      └── go test      -> decodes 40 PNGs + 20 WAVs in pure Go -> asserts ONNX logits match Python <= 1e-4

POST /ekyc/video  |  POST /unboxing/video  (MP4 / WebM upload)
 └── internal/video/demux.go (ffmpeg -> 1 FPS 128x128 RGB PNGs + 16kHz mono 16-bit PCM WAV)
      ├── Goroutine 1 (errgroup): Cosine Frame Dedup (>= 0.98) -> Pure-Go HWC->CHW -> Vision ONNX
      └── Goroutine 2 (errgroup): Pure-Go RIFF/WAV Parser -> [100, 2] Log-RMS & ZCR -> Audio VAD ONNX
           └── Aggregated Domain Verdict + Stage-Level Latency Telemetry (demux_us vs. inference_us)
```

![Multimodal Golden Fixtures Preview](./artifacts/multimodal_samples_preview.png)

---

## 3. How Multimodal Skew Is Eliminated (Python <-> Pure Go)

Unlike tabular ML (Module 01) or token IDs (Modules 02–03), multimodal pipelines suffer from **silent numerical drift** when Python C-extensions (`Pillow`, `librosa`, `OpenCV`) and backend services disagree on pixel layout or waveform framing. This module eliminates CGO media dependencies (`gocv`, `libsndfile`) and enforces 1:1 math in Go's standard library:

### Vision (`internal/vision/preprocess.go`)
* **Memory Layout (`HWC` -> `CHW`):** Decodes PNG/JPEG via Go stdlib `image`, maps each pixel's 16-bit alpha-premultiplied `RGBA()` channel back to 8-bit (`>> 8`), scales `[0..255] -> [0.0..1.0]`, applies per-channel ImageNet normalization `(c - mean[c]) / std[c]`, and writes contiguous planar `CHW` slices (`[1, 3, 128, 128]`).
* **Temporal Frame Deduplication:** Consecutive video frames with flattened CHW cosine similarity $\ge 0.98$ are skipped before ONNX execution so static camera segments are not scored repeatedly.

### Audio (`internal/audio/preprocess.go`)
* **Pure-Go RIFF/WAVE Parser:** Walks WAV chunks directly via `encoding/binary` (`LittleEndian`), extracts raw 16-bit signed PCM samples, and normalizes by `32768.0` into `[-1.0, 1.0]`.
* **Acoustic Framing (`25 ms` window, `10 ms` hop):** Slices `16,000 Hz` mono audio into `100` overlapping frames (`400` samples/frame, `160` sample hop) and computes two time-domain features per frame—**Log-RMS Energy** ($\ln(\text{RMS} + 10^{-6})$) and **Zero-Crossing Rate (ZCR)**—producing a `[1, 100, 2]` float32 tensor identical to Python's NumPy implementation.

---

## 4. Real-Dataset Benchmarks & Engineering Takeaways

### Model Quality, Artifact Footprint & Parity Gate

| Model | Real Public Dataset (`kagglehub`) | Macro-F1 | Python CPU p50 | ONNX Size | Go Parity (`<= 1e-4`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **eKYC Vision (2D-CNN)** | `urbikn/sroie-datasetv2` (973 scanned docs) | `0.6436` | `0.123 ms` | `0.09 MB` | 20/20 PASS (`0.05s`) |
| **Unboxing Vision (2D-CNN)** | `ayesha19765/damaged-package-detection` (8,503 imgs) | `0.9208` | `0.116 ms` | `0.09 MB` | 20/20 PASS (`0.02s`) |
| **Audio VAD (1D-CNN)** | `jayjoshi37/deepfake-audio-dataset-fake-vs-real-speech` (16,792 clips) | `1.0000` | `0.019 ms` | `0.01 MB` | 20/20 PASS (`0.01s`) |

**Why does `eKYC Vision` score `0.6436` while `Unboxing Vision` scores `0.9208`?**
* Both use an ultra-compact **3-layer scratch 2D-CNN (`0.09 MB`)** trained for 15 epochs on CPU (`128x128`).
* `damaged-package-detection` has strong spatial contrast (intact box geometry vs. crushed corners/open cavities), allowing a tiny scratch CNN to reach `0.9208` Macro-F1.
* `sroie-datasetv2` consists of dense, high-variance receipt typography; downsampling fine text scans to `128x128` makes subtle blur vs. clean-text boundaries harder for a 3-layer scratch network. In production, swapping `CompactVisionCNN` for a pretrained **MobileNetV3-Small** (`224x224`) backbone raises document quality F1 above `0.90+` without changing a single line of the Go serving or parity architecture.

### End-to-End Video Latency Breakdown (3-Second MP4 Clip)

| Endpoint | Frames (Sampled -> Deduped) | `ffmpeg` Demux I/O | Concurrent Go `errgroup` ONNX | Total HTTP Latency |
| :--- | :---: | :---: | :---: | :---: |
| `POST /ekyc/video` | 3 -> 3 | `290.6 ms` | **`5.05 ms`** | `296.3 ms` |
| `POST /unboxing/video` | 3 -> 3 | `405.9 ms` | **`4.11 ms`** | `422.2 ms` |

> **Key Systems Insight:** Neural network inference across 3 vision frames + 1 audio stream takes **under `5.1 ms` (~2% of total time)** when parallelized with `errgroup`, whereas container demuxing and media decoding (`ffmpeg` subprocess + disk I/O) consume **`290–406 ms` (~98%)**. At high throughput, engineering effort belongs in **streaming/in-memory media demuxing or async queue workers**, not GPU over-provisioning.

---

## 5. Quickstart & API Reference

Requires `ffmpeg` on `PATH`. Run from the `apprentice/` repository root:

```bash
# 1. Train all 3 models on real Kaggle datasets (auto-falls back to synthetic offline)
make train MOD=04-multimodal-ai

# 2. Generate Go contract constants + SHA-256 SchemaHash
make gen MOD=04-multimodal-ai

# 3. Run non-cached 60/60 Python <-> Go parity suite
make test MOD=04-multimodal-ai

# 4. Regenerate visual benchmark charts from exported artifacts
cd 04-multimodal-ai/ml && uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_multimodal_benchmark.ipynb && cd ../..

# 5. Launch the concurrent Go server on :8084
make serve MOD=04-multimodal-ai
```

### HTTP Endpoints

| Method & Path | Payload | Description |
| :--- | :--- | :--- |
| `GET /healthz` | — | Returns `status`, `schema_version`, and `schema_hash` |
| `POST /ekyc/video` | `multipart/form-data` (`file`) or raw MP4 | Demuxes video and runs concurrent eKYC document quality + OTP speech VAD |
| `POST /unboxing/video` | `multipart/form-data` (`file`) or raw MP4 | Demuxes video, deduplicates static frames, and runs concurrent parcel damage + VAD |
| `POST /image/classify?mode=ekyc\|unboxing` | PNG / JPEG | Atomic single-image vision inference |
| `POST /audio/vad` | 16 kHz mono 16-bit WAV | Atomic voice activity detection (`speech_present` vs. `silence_or_noise`) |

### Live Smoke Test

```bash
ffmpeg -y -f lavfi -i testsrc=duration=3:size=128x128:rate=10 \
  -f lavfi -i sine=frequency=220:duration=3 \
  -c:v libx264 -pix_fmt yuv420p -c:a aac /tmp/sample_clip.mp4 -loglevel error

curl -s -X POST http://localhost:8084/ekyc/video -F "file=@/tmp/sample_clip.mp4" | jq
curl -s -X POST http://localhost:8084/unboxing/video -F "file=@/tmp/sample_clip.mp4" | jq
rm -f /tmp/sample_clip.mp4
```
