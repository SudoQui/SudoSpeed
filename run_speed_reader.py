# run_speed_reader.py
# Detector + Recogniser + Zone confirmation (temporal consistency, hysteresis, cooldown)
# GPU accelerated (CUDA), FP16, batched classifier, optional frame skipping, non blocking UI.

import os
import time
import math
import cv2
import torch
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO

# ==========================
# EDIT THESE 3 PATHS
# ==========================
DETECT_WEIGHTS = r"C:\Users\musta\Documents\Programming\SudoSpeed\runs\detect\train2\weights\best.pt"
REC_WEIGHTS    = r"C:\Users\musta\Documents\Programming\SudoSpeed\runs\classify\train7\weights\best.pt"
DATA_ROOT      = r"C:\Users\musta\Documents\Programming\SudoSpeed"  # for optional outputs

# ==========================
# SOURCE (camera or video or folder or single image)
# ==========================
# 0 for default webcam; or a raw string path to a video; or a folder of images; or a single image file
# Example video file:
# SOURCE = r"C:\Users\musta\Videos\cityTest.mp4"
SOURCE = r"C:\Users\musta\Pictures\speed.png"

# ==========================
# RUNTIME / SPEED CONFIG
# ==========================
IMG_SIZE_CLS = 224              # recogniser input size used in training
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda:0" if USE_GPU else "cpu"
FP16 = bool(USE_GPU)            # use half precision on GPU
DETECT_IMGSZ = 640              # 512–640 is a good speed/accuracy tradeoff
FAST_PREVIEW = False            # True = no drawings, fastest compute throughput
PROCESS_EVERY_N = 1             # process every Nth frame (try 2 or 3 for speed)
DRAW_EVERY = 2                  # draw overlays every Nth frame to reduce UI cost

DRAW_BOX_COLOR = (0, 255, 0)
DRAW_TEXT_COLOR = (0, 255, 0)
FONT_SCALE = 0.6
THICK = 2
SAVE_VIDEO = False
CONF_DET_THRESH = 0.25          # detector confidence threshold (for boxes)

# Optional small speed up on GPU
if USE_GPU:
    torch.backends.cudnn.benchmark = True

# ==========================
# GATING / LOGIC CONFIG
# ==========================
CONF_THR = 0.75                 # recogniser min confidence (0.70–0.80 suggested)
MIN_BOX_H_FRAC = 0.03           # bbox height ≥ 3% of frame height
RIGHT_SIDE_FRAC = 0.30          # accept x center in right 70% (x_center >= 0.30*W)
ASPECT_MIN, ASPECT_MAX = 0.6, 1.4  # near square gate (w/h)

# Temporal window: K hits within ~W frames/seconds (adaptive to FPS)
WIN_FRAMES = 12
CONFIRM_HITS = 3
TARGET_WINDOW_SEC_AT_30FPS = 0.4

# Hysteresis + cooldown
COOLDOWN_S = 3.0                # ignore conflicting values briefly after change
EXTRA_HITS_TO_CHANGE = 1        # require stronger evidence to change zone than first set
# STRONGER_CONF_MARGIN = 0.15   # hook left for future

# Expiry (time based unless you feed odometry)
MAX_GAP_ANY_SIGN_S = 60.0       # if no sign for >60s, maybe drop to default
DEFAULT_ZONE = None             # e.g., "50" if you want fallback; None = keep last

# ==========================
# STATE (minimal state machine)
# ==========================
current_zone = None
zone_set_ts = -1e18
pending_votes = defaultdict(lambda: deque(maxlen=64))  # value -> deque[timestamps]
last_seen = {}                                         # value -> last time seen (float seconds)

# ==========================
# UTILITIES
# ==========================
def prep_for_recogniser(bgr_img, use_edges=True, grayscale=True):
    """Apply grayscale + CLAHE (+ optional edges) and return 3 channel gray (BGR) to feed classifier."""
    if not grayscale:
        return bgr_img
    g = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    if use_edges:
        e = cv2.Canny(g, 80, 160)
        g = cv2.addWeighted(g, 0.85, e, 0.15, 0)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def draw_label(img, x1, y1, text):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICK)
    x2 = x1 + tw + 8
    y2 = max(0, y1 - th - 8)
    cv2.rectangle(img, (x1, y2), (x2, y1), (0, 0, 0), -1)
    cv2.putText(img, text, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, DRAW_TEXT_COLOR, THICK, cv2.LINE_AA)

def plausible_geometry(box_xyxy, frame_w, frame_h):
    """Basic geometric gates: size, side of road ROI, aspect ratio near square."""
    x1, y1, x2, y2 = box_xyxy
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    h_frac = h / float(frame_h)
    if h_frac < MIN_BOX_H_FRAC:
        return False
    xc = (x1 + x2) / 2.0
    if xc < RIGHT_SIDE_FRAC * frame_w:
        return False
    aspect = w / float(h)
    if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
        return False
    return True

def window_secs(estimated_fps):
    """Adaptive window about 0.4 s at 30 FPS, scales with measured fps."""
    if estimated_fps <= 1e-3:
        return TARGET_WINDOW_SEC_AT_30FPS
    scale = 30.0 / max(1e-6, estimated_fps)
    return TARGET_WINDOW_SEC_AT_30FPS * scale

def cull_old_votes(now_s, win_s):
    """Remove votes older than the active time window."""
    for v in list(pending_votes.keys()):
        dq = pending_votes[v]
        while dq and (now_s - dq[0] > win_s):
            dq.popleft()
        if not dq:
            pending_votes.pop(v, None)

def too_long_since_any_sign(now_s):
    if not last_seen:
        return False
    most_recent = max(last_seen.values())
    return (now_s - most_recent) > MAX_GAP_ANY_SIGN_S

def set_zone(v, now_s):
    global current_zone, zone_set_ts
    current_zone = v
    zone_set_ts = now_s
    print(f"[ZONE] {v} @ {time.strftime('%H:%M:%S')}")

# ==========================
# CORE: process detections -> update state
# ==========================
def process_frame_logic(detections, now_s, frame_w, frame_h, est_fps):
    """
    detections: list of (value:str, conf:float, (x1,y1,x2,y2))
    Updates global state: pending_votes, last_seen, current_zone, zone_set_ts.
    Returns: best_value, best_hits, in_cooldown, promoted
    """
    global current_zone

    win_s = window_secs(est_fps)

    # 1) validate and register votes
    for value, conf, (x1, y1, x2, y2) in detections:
        if conf < CONF_THR:
            continue
        if not plausible_geometry((x1, y1, x2, y2), frame_w, frame_h):
            continue
        pending_votes[value].append(now_s)
        last_seen[value] = now_s

    # cull old votes
    cull_old_votes(now_s, win_s)

    # 2) score candidates
    best_value, best_score = None, -1.0
    for v, times in pending_votes.items():
        hits = len(times)
        age_s = max(0.0, now_s - times[-1]) if times else 1e9
        recency_bonus = max(0.0, (win_s - age_s)) * 0.25
        score = hits + recency_bonus
        if score > best_score:
            best_value, best_score = v, score

    # 3) confirm with hysteresis and cooldown
    promoted = None
    in_cooldown = (now_s - zone_set_ts) < COOLDOWN_S

    if best_value is not None:
        hits = len(pending_votes[best_value])
        if current_zone is None:
            if hits >= CONFIRM_HITS:
                set_zone(best_value, now_s)
                promoted = best_value
        elif best_value != current_zone:
            need_hits = CONFIRM_HITS + EXTRA_HITS_TO_CHANGE
            if (hits >= need_hits) and (not in_cooldown):
                set_zone(best_value, now_s)
                promoted = best_value

    # 4) expiry
    if current_zone is not None:
        if too_long_since_any_sign(now_s):
            if DEFAULT_ZONE is not None and DEFAULT_ZONE != current_zone:
                set_zone(DEFAULT_ZONE, now_s)

    best_hits = len(pending_votes.get(best_value, [])) if best_value else 0
    return best_value, best_hits, in_cooldown, promoted

# ==========================
# MAIN
# ==========================
def main():
    print("Device:", DEVICE, "| FP16:", FP16)
    if USE_GPU:
        print("GPU:", torch.cuda.get_device_name(0))
    os.makedirs(os.path.join(DATA_ROOT, "outputs"), exist_ok=True)

    # Load models
    detector = YOLO(DETECT_WEIGHTS)
    recogniser = YOLO(REC_WEIGHTS)

    # Move to device and optionally fuse for speed
    try:
        detector.to(DEVICE)
        recogniser.to(DEVICE)
    except Exception:
        pass
    try:
        detector.fuse()
        recogniser.fuse()
    except Exception:
        pass

    # Set up source
    is_cam = isinstance(SOURCE, int)
    out_writer = None
    paths = None

    if is_cam:
        cap = cv2.VideoCapture(SOURCE, cv2.CAP_DSHOW)  # CAP_DSHOW tends to be reliable on Windows
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam (SOURCE=0). Try SOURCE=1 or a video file path.")
        print("Source type: webcam")
    else:
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        if os.path.isdir(SOURCE):
            paths = [os.path.join(SOURCE, f) for f in os.listdir(SOURCE) if f.lower().endswith(exts)]
            paths.sort()
            cap = None
            print(f"Source type: image folder with {len(paths)} files")
        elif os.path.isfile(SOURCE) and SOURCE.lower().endswith(exts):
            paths = [SOURCE]  # single image
            cap = None
            print("Source type: single image")
        else:
            cap = cv2.VideoCapture(SOURCE)  # assume video
            if not cap.isOpened():
                raise RuntimeError(f"Could not open source: {SOURCE}")
            print("Source type: video file")

    window = "SudoSpeed: Speed Reader (GPU/FP16)"
    last_t = time.time()
    ema_fps = None  # smooth FPS estimate for window sizing
    frame_idx = 0

    while True:
        # Get a frame
        if is_cam or (cap and cap.isOpened()):
            ok, frame = cap.read()
            if not ok:
                break
        else:
            if not paths:
                break
            fp = paths.pop(0)
            frame = cv2.imread(fp)
            if frame is None:
                continue

        frame_idx += 1
        H, W = frame.shape[:2]

        # Optionally skip heavy processing for speed
        do_process = (frame_idx % PROCESS_EVERY_N) == 0

        detections = []

        if do_process:
            # Detector
            with torch.inference_mode():
                det_res = detector.predict(
                    frame,
                    device=DEVICE,
                    imgsz=DETECT_IMGSZ,
                    conf=CONF_DET_THRESH,
                    half=FP16,
                    verbose=False
                )[0]

            # Collect crops for batched recogniser
            crops, boxes_xyxy = [], []
            for b in det_res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                # clip
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                # size only quick gate
                if (y2 - y1) / float(H) < MIN_BOX_H_FRAC:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_in = prep_for_recogniser(crop, use_edges=True, grayscale=True)
                crops.append(crop_in)
                boxes_xyxy.append((x1, y1, x2, y2))

            # Batched recogniser
            with torch.inference_mode():
                if crops:
                    cls_batch = recogniser.predict(
                        crops,
                        imgsz=IMG_SIZE_CLS,
                        device=DEVICE,
                        half=FP16,
                        verbose=False,
                        batch=32  # adjust based on VRAM
                    )
                    for (x1, y1, x2, y2), res in zip(boxes_xyxy, cls_batch):
                        idx = int(res.probs.top1)
                        name = res.names[idx]  # e.g. "40"
                        conf = float(res.probs.top1conf)
                        detections.append((name, conf, (x1, y1, x2, y2)))

            # Draw raw predictions
            if not FAST_PREVIEW and detections:
                for name, conf, (x1, y1, x2, y2) in detections:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), DRAW_BOX_COLOR, 2)
                    draw_label(frame, x1, y1, f"{name} ({conf:.2f})")

        # Timing and FPS
        now = time.time()
        dt = max(1e-6, (now - last_t))
        inst_fps = 1.0 / dt
        last_t = now
        if ema_fps is None:
            ema_fps = inst_fps
        else:
            ema_fps = 0.9 * ema_fps + 0.1 * inst_fps

        # Update confirmation logic
        best_value, best_hits, in_cooldown, promoted = process_frame_logic(
            detections, now, W, H, ema_fps
        )

        # Overlay status
        if not FAST_PREVIEW and ((frame_idx % DRAW_EVERY) == 0):
            cv2.putText(frame, f"FPS: {ema_fps:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cz_txt = f"ZONE: {current_zone}" if current_zone is not None else "ZONE: —"
            cz_color = (0, 200, 255) if in_cooldown else (50, 220, 50)
            cv2.putText(frame, cz_txt, (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, cz_color, 2, cv2.LINE_AA)

            cand_txt = f"CANDIDATE: {best_value or '—'}  hits:{best_hits}  cooldown:{'Y' if in_cooldown else 'N'}"
            cv2.putText(frame, cand_txt, (10, 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

        # Optional save video
        if SAVE_VIDEO and isinstance(SOURCE, int) and not FAST_PREVIEW:
            if out_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_path = os.path.join(DATA_ROOT, "outputs", "annotated.mp4")
                out_writer = cv2.VideoWriter(out_path, fourcc, 30.0, (W, H))
                print("Saving video to:", out_path)
            out_writer.write(frame)

        # Show
        if not FAST_PREVIEW:
            cv2.imshow(window, frame)

            # If we are on images and this was the last one, wait for a key
            if not is_cam and cap is None and (not paths):
                key = cv2.waitKey(0) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

    # Cleanup
    if 'cap' in locals() and cap:
        cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
