# README.md
# SudoSpeed — Real-time AU Speed-Sign Detection and Zone Tracking

> SudoSpeed detects Australian speed-limit signs and maintains the current zone in real time. It is designed to feed a motorcycle helmet HUD such as MotorHUD. This repository focuses on the vision stack only: detector, recogniser, and zone logic.

<p align="center">
  <img src="demo/imgs/teaser_01.png" alt="SudoSpeed teaser" width="720">
</p>

---

## Features
* Real time sign detection using a YOLO based model tuned for AU signs  
* Digit recognition for robust 40 50 60 80 100 110 decisions  
* Zone tracking with confirmation window, hysteresis, and cooldown  
* Sources: webcam, single video file, or image folder  
* ONNXRuntime path for portable inference on Raspberry Pi  
* Simple outputs to console or overlay with hooks to stream to a HUD

---

## Quickstart

### Environment
* Python 3.10 to 3.12  
* One of the following for inference  
  * PyTorch  
  * ONNXRuntime or ONNXRuntime GPU

```bash
pip install -r requirements.txt
```

### Run on the test video

```bash
python run_speed_reader.py ^
  --source "demo/videos/test_short.mp4" ^
  --det weights/detector.pt ^
  --rec weights/recogniser.pt ^
  --view-overlay
```

Windows path tip. Use forward slashes or double backslashes.
```
--source "C:/Users/you/Videos/test_short.mp4"
--source "C:\\Users\\you\\Videos\\test_short.mp4"
```

---

## Weights
Place your weights under `weights`.

```
weights/
├─ detector.pt
├─ recogniser.pt
├─ detector.onnx        # optional
└─ recogniser.onnx      # optional
```

Common origins.  
`runs/detect/train2/weights/best.pt` becomes `weights/detector.pt`  
`runs/classify/train7/weights/best.pt` becomes `weights/recogniser.pt`  

Prefer ONNX for Raspberry Pi. Use `.pt` for training or CUDA.

---

## Zone logic at a glance
* Collect candidates per frame then record value and confidence.  
* Keep plausible values only and apply a minimum confidence.  
* Confirm within a sliding window such as N equals 8 with K equals 5 hits.  
* Apply hysteresis to keep the current zone unless a stronger signal appears.  
* Apply a short cooldown after each update.

Example `config/zone.yaml`
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

## Demo media
Exactly one image and one video.

```
demo/
├─ imgs/
│  └─ teaser_01.png
└─ videos/
   └─ test_short.mp4
```

---

## Repository layout

```
.
SudoSpeed/
├─ run_speed_reader.py
├─ Sudospeed.py
├─ detector.pt
├─ detector.onnx
├─ recogniser.pt
├─ recogniser.onnx
├─ demo_imgs/
│  └─ ... (images)
├─ demo_videos/
│  └─ ... (videos)
├─ requirements.txt
└─ README.md

```

---

## License
MIT

## Contact
Maintainer: Mustafa  
Project: SudoSpeed
