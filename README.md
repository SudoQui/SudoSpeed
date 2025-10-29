# README.md
# SudoSpeed — Real-time AU Speed-Sign Detection & Zone Tracking (for a helmet HUD)

> **TL;DR:** SudoSpeed detects Australian speed-limit signs and maintains the current *zone* in real time. It’s designed to feed a motorcycle **helmet HUD** (e.g., MotorHUD), but **this repo focuses on the vision stack only** (detector + recogniser + zone logic)—no helmet hardware code here.

<p align="center">
  <img src="demo/imgs/teaser_01.jpg" alt="SudoSpeed teaser" width="720">
</p>

---

## ✨ Features
- **Real-time speed-sign detection** (YOLO-based), tuned for AU signs  
- **Digit recognition** for robust 40/50/60/80/100/110 decisioning  
- **Zone tracking logic** with hysteresis, confirmation window, and cooldown  
- **Multi-source I/O:** Webcam / Video / Image folder  
- **ONNXRuntime path** for CPU/GPU + low-power devices  
- **Simple outputs** (console, overlay) + hooks to stream to a HUD/ESP32

---

## 🧱 Architecture
- **Optimized for Raspberry Pi 3 & Pi Camera**  
  - Tuned image sizes and operators for Pi-class CPUs/ISPs  
  - ONNXRuntime path for lightweight inference

- **Backbone: Nano CSPDarknet (swap)**  
  - Compact receptive field for small round sign targets  
  - **Augmented for motion blur & glare** (aggressive shutter/contrast sims)

- **Learned post-processing module**  
  - Replaces traditional NMS with a small learned head  
  - Smarter, context-aware deduping and stability under clutter

- **Quantization-Aware Training (QAT)**  
  - Calibrated INT8 path for efficient edge deployment on Raspberry Pi  
  - Preserves accuracy while reducing latency and power

**Pipeline (high level):**  
Camera → Detector (Nano CSPDarknet) → **Learned Post-Proc** → Crop → Digit Recogniser → **Zone Logic** (confirm + hysteresis + cooldown) → HUD output (`Z:<kph>`)

---

## 📦 Pretrained Weights
Put your weights in `weights/`:

```
weights/
├─ detector.pt              # YOLO detector (PyTorch)
├─ recogniser.pt            # Digit/patch recogniser (PyTorch)
├─ detector.onnx            # optional ONNX export
└─ recogniser.onnx          # optional ONNX export
```

Typical source paths you might already have:
- `runs/detect/train2/weights/best.pt` → copy/rename to `weights/detector.pt`  
- `runs/classify/train7/weights/best.pt` → copy/rename to `weights/recogniser.pt`

> Prefer **ONNX** on low-power devices. Use `.pt` when training, prototyping, or on CUDA-enabled machines.

---

## 🎬 Demos
Put media under `demo/` and the README will render them on GitHub.

```
demo/
├─ imgs/
│  ├─ teaser_01.jpg
│  └─ frames_*.jpg
└─ videos/
   ├─ city_short.mp4
   └─ highway_short.mp4
```

**Image example**

<p align="center">
  <img src="demo/imgs/teaser_01.jpg" alt="Teaser frame" width="720"><br/>
  <em>Detector + zone overlay (example)</em>
</p>

**Video examples**
- City test (short):  
  https://github.com/<YOUR_USER>/<YOUR_REPO>/raw/main/demo/videos/city_short.mp4
- Highway test (short):  
  https://github.com/<YOUR_USER>/<YOUR_REPO>/raw/main/demo/videos/highway_short.mp4

> For inline previews in the README, consider adding an animated GIF derived from a short clip.

---

## 🚀 Quickstart

### 1) Environment
- Python **3.10–3.12** (3.11 recommended)
- (Option A) **PyTorch + CUDA** (training / GPU inference), or  
- (Option B) **ONNXRuntime** (`onnxruntime` or `onnxruntime-gpu`) for light/portable inference

```bash
pip install -r requirements.txt
# Typical contents:
# ultralytics
# opencv-python
# numpy
# onnxruntime          # or onnxruntime-gpu
# pillow
# rich
```

### 2) Run on a video

```bash
python run_speed_reader.py ^
  --source "demo/videos/city_short.mp4" ^
  --det weights/detector.pt ^
  --rec weights/recogniser.pt ^
  --view-overlay
```

**Windows path tip:** Use forward slashes or double backslashes:

```
--source "C:/Users/you/Videos/cityTest.mp4"
# or
--source "C:\\Users\\you\\Videos\\cityTest.mp4"
```

### 3) Webcam

```bash
python run_speed_reader.py --source 0 --det weights/detector.onnx --rec weights/recogniser.onnx --view-overlay
```

### 4) Folder of images

```bash
python run_speed_reader.py --source demo/imgs --det weights/detector.pt --rec weights/recogniser.pt --save
```

---

## ⚙️ CLI Options (common)

```
--source            Path/int; 0 for webcam, file, or folder
--det               Detector weights (.pt or .onnx)
--rec               Recogniser weights (.pt or .onnx)
--imgsz             Inference image size (e.g., 640)
--conf              Confidence threshold (e.g., 0.25)
--iou               NMS/Head IoU threshold (e.g., 0.45)
--device            cuda | cpu
--view-overlay      Draw boxes/labels and current zone on frames
--save              Save annotated outputs (to runs/ or out/)
--fps-max           Cap processing FPS (useful for reproducibility)
```

---

## 🧠 Zone Logic (how it avoids flicker & false positives)

* **Collect candidates per frame:** Detect sign → crop → recognise digits → `(value, confidence)`
* **Stability filter:** Keep only plausible values (e.g., {40,50,60,80,100,110}) above `conf_min`
* **Confirmation window:** A value must appear consistently in a sliding window (e.g., last `N=8` frames with ≥ `K=5` hits)
* **Hysteresis & cooldown:**

  * Stronger signal needed to *change* zone than to *keep* it
  * After a zone update, ignore new changes for `cooldown_ms` (prevents rapid flips)
* **Expiry:** If no valid signs seen for `T` seconds, retain last known zone (configurable)

Example config (`config/zone.yaml`):

```yaml
confirmation:
  window: 8
  min_hits: 5
hysteresis:
  keep_bias: 0.1
cooldown_ms: 1500
expiry_s: 90
values: [40, 50, 60, 80, 100, 110]
```

---

## 📁 Repo Layout

```
.
├─ run_speed_reader.py        # main entry: video/webcam/folder
├─ sudospeed/
│  ├─ detector.py             # loads YOLO/ONNX detector (Nano CSPDarknet backbone)
│  ├─ recogniser.py           # digit/patch recogniser
│  ├─ pipeline.py             # per-frame pipeline + overlays
│  ├─ zone_logic.py           # confirmation, hysteresis, cooldown
│  └─ utils.py                # drawing, IO helpers
├─ config/
│  ├─ detector.yaml
│  ├─ recogniser.yaml
│  └─ zone.yaml
├─ weights/                   # (add yours) detector/recogniser .pt/.onnx
├─ demo/
│  ├─ imgs/                   # images for README
│  └─ videos/                 # short clips for README
├─ requirements.txt
└─ README.md
```

---

## 📊 Performance Notes

* **Pi 3 / Pi Zero 2 W:** prefer **ONNX** + low `imgsz` (e.g., 416–512) and cap FPS
* **PC/Laptop (CUDA):** PyTorch `.pt` or ONNXRuntime-GPU both work well
* **Noise handling:** Add negatives and hard examples; use moderate `--conf` and good IoU thresholds
* **QAT + INT8:** Use quantization-aware training to unlock INT8 deployment on Raspberry Pi

---

## 🧪 Training (optional)

1. **Detector** (Ultralytics YOLO):

   ```bash
   yolo detect train data=./data/sudospeed.yaml model=yolov8n.pt imgsz=640 epochs=100
   ```
2. **Recogniser** (digit classifier / patch classifier):
   Prepare digit crops; standard train/val split; export best `.pt`
3. **Export to ONNX** (recommended for embedded):

   ```bash
   yolo export model=runs/detect/train2/weights/best.pt format=onnx opset=12
   ```

---

## 🪫 Integration with a Helmet HUD

SudoSpeed emits a **current zone** (e.g., `Z:60`) whenever it updates. You can:

* Print to stdout, write to a socket/serial, or
* Publish over BLE/SPP/XBee to an **ESP32** hub which drives a HUD.

> This repo does **not** include helmet hardware code—SudoSpeed just supplies the vision signal your helmet can display.

---

## 🔒 Safety & Disclaimer

* For research and prototyping only. Do **not** rely on this as a sole source of truth while riding/driving.
* Always follow road signs and local laws.
* You assume all risks of use.

---

## 🙏 Acknowledgements

* Ultralytics YOLOv8
* OpenCV, ONNXRuntime, PyTorch
* Community datasets and contributors

---

## 📝 License

MIT (see `LICENSE`)

---

## 🤝 Contributing

Issues and PRs welcome! Please attach:

* A short clip or frames showing the issue
* Your command line and config
* System details (OS, Python, CPU/GPU)

---

## 📫 Contact

* Maintainer: **Mustafa**
* Project: **SudoSpeed** (vision for a helmet HUD)
