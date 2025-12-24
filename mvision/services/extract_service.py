# file: mvision/services/extract_service.py
from __future__ import annotations

import csv
import sys
import os
import time
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple, Callable
from queue import Queue, Empty
import logging
import shutil
import random

import cv2
import numpy as np
import torch

from mvision.constants import (
    RTSP_STREAMS,
    YOLO_WEIGHTS,
    DEVICE,
    CONF_THRES,
    IOU_THRES,
    EMB_CSV,
    CROPS_ROOT,
)
from mvision.db.session import get_session, engine
from mvision.db.models import Base, User

# ------------------------- runtime perf knobs -------------------------
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass
try:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
except Exception:
    pass
try:
    cv2.setNumThreads(1)
except Exception:
    pass

# ------------------------- optional libs -------------------------
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from torchreid.utils import FeatureExtractor as TorchreidExtractor
except Exception as e:
    print("TorchReID import error:", e)
    TorchreidExtractor = None

try:
    from insightface.app import FaceAnalysis
    INSIGHT_OK = True
except Exception:
    FaceAnalysis = None
    INSIGHT_OK = False

try:
    import onnxruntime as ort
except Exception:
    ort = None

# ------------------------- config -------------------------
EXPECTED_EMBED_DIM = 512

# Face capture/link gates (capture-time only)
USE_FACE = True
FACE_MODEL = "buffalo_l"
FACE_DET_SIZE = (640, 640)         # (w, h)
FACE_PROVIDER = "auto"             # "auto" | "cuda" | "cpu"
FACE_MIN_SCORE = 0.5
FACE_MIN_SIZE = 32                 # px min(side) of detected face
FACE_IOU_LINK = 0.05
FACE_OVER_FACE_LINK = 0.60
FACE_EVERY = 2

# YOLO inference size
YOLO_IMGSZ = 640

# Queues
CAP_QUEUE_MAX = 2
IO_QUEUE_MAX  = 1024

# Viewer FPS throttle
VIEWER_SLEEP_MS = 5

# ======= stronger embeddings knobs =======
REID_MODELS = ["osnet_x1_0", "osnet_x0_25"]  # strong → light
BODY_TTA_FLIP = True
FACE_TTA_FLIP = True

# ------------------------- logging -------------------------
logger = logging.getLogger("mvision.extract_service")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# ------------------------- globals -------------------------
yolo_model: YOLO | None = None
reid_extractor: TorchreidExtractor | None = None   # legacy single ref
reid_extractors: List[TorchreidExtractor] = []     # ensemble
face_app: FaceAnalysis | None = None

yolo_lock = threading.Lock()
reid_lock = threading.Lock()
csv_lock = threading.Lock()
frames_lock = threading.Lock()

latest_frames: Dict[int, np.ndarray] = {}

capture_threads: Dict[int, threading.Thread] = {}
extract_threads: Dict[int, threading.Thread] = {}
cap_queues: Dict[int, Queue] = {}

io_thread: Optional[threading.Thread] = None
io_queue: Queue | None = None

viewer_thread: Optional[threading.Thread] = None
stop_event = threading.Event()
is_running = False
current_person_id: Optional[int] = None
current_person_name: Optional[str] = None

# ---- extraction progress state (for /extract) ----
progress_lock = threading.Lock()
progress_state: Dict[int, Dict[str, object]] = {}

# ------------------ progress helpers ------------------
def _progress_reset(user_id: int, user_name: str) -> None:
    with progress_lock:
        progress_state[user_id] = {
            "user_name": user_name,
            "stage": "idle",
            "total_body": 0,
            "total_face": 0,
            "done_body": 0,
            "done_face": 0,
            "percent": 0,
            "message": "waiting",
        }

def _progress_set(user_id: int, **kv) -> None:
    with progress_lock:
        st = progress_state.get(user_id)
        if not st:
            st = {}
            progress_state[user_id] = st
        st.update(kv)

        tb, tf = int(st.get("total_body", 0)), int(st.get("total_face", 0))
        db, df = int(st.get("done_body", 0)), int(st.get("done_face", 0))
        db = max(0, min(db, tb))
        df = max(0, min(df, tf))
        st["done_body"], st["done_face"] = db, df

        stage = str(st.get("stage", "")).lower()
        if stage == "embedding_body" and tb > 0:
            pct = int(round((db / max(1, tb)) * 100))
        elif stage == "embedding_face" and tf > 0:
            pct = int(round((df / max(1, tf)) * 100))
        else:
            tot = max(1, tb + tf)
            pct = int(round(((db + df) / tot) * 100))
        st["percent"] = max(0, min(100, pct))

def get_progress_for_user(user_id: int) -> Dict[str, object]:
    with progress_lock:
        st = progress_state.get(user_id)
        return dict(st) if st else {
            "stage": "unknown",
            "percent": 0,
            "message": "no job",
            "total_body": 0,
            "total_face": 0,
            "done_body": 0,
            "done_face": 0,
        }

# ------------------ init_db (startup hook) ------------------
def init_db() -> None:
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("DB initialized.")
    except Exception:
        logger.exception("DB init failed")

# ------------------ utils ------------------
def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n == 0.0 or not np.isfinite(n):
        return v
    return v / n

def _as_512f(vec: np.ndarray | list | None) -> Optional[np.ndarray]:
    if vec is None:
        return None
    try:
        a = np.asarray(vec, dtype=np.float32).reshape(-1)
        if a.size != EXPECTED_EMBED_DIM:
            return None
        if not np.isfinite(a).all():
            return None
        return l2_normalize(a)
    except Exception:
        return None

def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = a_area + b_area - inter
    return inter / denom if denom > 0 else 0.0

def inter_over_face(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    face_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    return inter / face_area if face_area > 0 else 0.0

def face_center_in(a: Tuple[int, int, int, int], b: Tuple[float, float, float, float]) -> bool:
    fx1, fy1, fx2, fy2 = a
    px1, py1, px2, py2 = b
    cx = (fx1 + fx2) * 0.5
    cy = (fy1 + fy2) * 0.5
    return (px1 <= cx <= px2) and (py1 <= cy <= py2)

def safe_iter_faces(obj):
    if obj is None:
        return []
    try:
        return list(obj)
    except TypeError:
        return [obj]

def extract_face_embedding(face):
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
    return emb

def _cuda_ep_loadable() -> bool:
    if ort is None:
        return False
    try:
        if sys.platform.startswith("mac"):
            return False
        from pathlib import Path as _P
        import ctypes as _ct
        capi_dir = _P(ort.__file__).parent / "capi"
        name = "onnxruntime_providers_cuda.dll" if os.name == "nt" else "libonnxruntime_providers_cuda.so"
        lib_path = capi_dir / name
        if not lib_path.exists():
            return False
        _ct.CDLL(str(lib_path))
        return True
    except Exception:
        return False

# ------------------ models init ------------------
def init_face_engine(use_face: bool, device: str, face_model: str, det_w: int, det_h: int, face_provider: str):
    if not use_face:
        return None
    if not INSIGHT_OK:
        logger.warning("insightface not installed; face recognition disabled.")
        return None
    try:
        is_cuda = ("cuda" in device.lower()) and torch.cuda.is_available()
        cuda_ok = _cuda_ep_loadable()

        providers = ["CPUExecutionProvider"]
        if face_provider == "cuda":
            if cuda_ok:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif face_provider == "auto":
            if is_cuda and cuda_ok:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        app = FaceAnalysis(name=face_model, providers=providers)
        ctx_id = 0 if providers[0].startswith("CUDA") else -1
        try:
            app.prepare(ctx_id=ctx_id, det_size=(det_w, det_h))
        except TypeError:
            app.prepare(ctx_id=ctx_id)
        logger.info("InsightFace ready (model=%s, providers=%s).", face_model, providers)
        try:
            dummy = np.zeros((max(8, det_h), max(8, det_w), 3), dtype=np.uint8)
            app.get(dummy)
        except Exception:
            pass
        return app
    except Exception:
        logger.exception("InsightFace init failed")
        return None

def init_models():
    """Load YOLO + TorchReID(backbones) + InsightFace once on startup."""
    global yolo_model, reid_extractor, reid_extractors, face_app

    gpu = torch.cuda.is_available() and ("cuda" in str(DEVICE).lower())
    logger.info("init_models device=%s gpu_available=%s", DEVICE, torch.cuda.is_available())

    # YOLO
    if YOLO is not None:
        try:
            weights = YOLO_WEIGHTS
            if not Path(weights).exists():
                logger.warning("%s not found, falling back to yolov8n.pt", weights)
            yolo = YOLO(weights if Path(YOLO_WEIGHTS).exists() else "yolov8n.pt")
            if gpu:
                try:
                    yolo.to(DEVICE)
                except Exception:
                    pass
            try:
                with torch.inference_mode():
                    _dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
                    yolo.predict(_dummy, device=DEVICE if gpu else "cpu",
                                 conf=0.25, iou=0.5, imgsz=YOLO_IMGSZ, verbose=False)
            except Exception:
                pass
            yolo_model = yolo
            logger.info("YOLO ready (gpu=%s)", gpu)
        except Exception:
            logger.exception("YOLO load failed; detection disabled.")
            yolo_model = None
    else:
        logger.warning("ultralytics not installed; detection disabled.")

    # TorchReID ensemble
    reid_extractors.clear()
    if TorchreidExtractor is not None:
        try:
            dev = DEVICE if gpu else "cpu"
            for m in REID_MODELS:
                try:
                    ext = TorchreidExtractor(model_name=m, device=dev)
                    try:
                        dummy = np.zeros((256, 128, 3), dtype=np.uint8)
                        _ = ext([dummy])
                    except Exception:
                        pass
                    reid_extractors.append(ext)
                    logger.info("TorchReID ready: %s (%s)", m, dev)
                except Exception:
                    logger.exception("TorchReID init failed for %s", m)
            globals()["reid_extractor"] = reid_extractors[0] if reid_extractors else None
        except Exception:
            logger.exception("TorchReID init failed; body embeddings disabled.")
            reid_extractors.clear()
            globals()["reid_extractor"] = None
    else:
        logger.warning("torchreid not installed; body embeddings disabled.")

    # InsightFace
    globals()["face_app"] = init_face_engine(
        USE_FACE, DEVICE, FACE_MODEL, FACE_DET_SIZE[0], FACE_DET_SIZE[1], FACE_PROVIDER
    )

# ------------------ CSV + gallery helpers ------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def ensure_csv_header():
    _ensure_dir(Path(EMB_CSV).parent if Path(EMB_CSV).parent.as_posix() not in (".", "") else Path("."))
    if not Path(EMB_CSV).exists():
        with csv_lock:
            with open(EMB_CSV, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "user_id","user_name","ts","cam_idx","frame_idx","det_idx",
                        "x1","y1","x2","y2","conf_or_score",
                        "body_embedding","face_embedding","crop_path","kind",
                    ]
                )
        logger.info("Created CSV header: %s", EMB_CSV)

def _gallery_body_dir(user_name: str) -> Path:
    d = Path(CROPS_ROOT) / "gallery" / user_name
    _ensure_dir(d)
    return d

def _gallery_face_dir(user_name: str) -> Path:
    d = Path(CROPS_ROOT) / "gallery_face" / user_name
    _ensure_dir(d)
    return d

def clear_user_galleries(user_name: str):
    for base in ("gallery", "gallery_face"):
        root = Path(CROPS_ROOT) / base / user_name
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        _ensure_dir(root)
    logger.info("Cleared galleries for '%s'", user_name)

def _epoch_ms() -> int:
    return int(time.time() * 1000) * 1000 + random.randint(0, 999)

# ------------------ face helpers (capture-time) ------------------
def detect_faces_raw(frame: np.ndarray):
    outs: List[Dict] = []
    if face_app is None:
        return outs
    try:
        faces = face_app.get(np.ascontiguousarray(frame))
        for f in safe_iter_faces(faces):
            bbox = getattr(f, "bbox", None)
            if bbox is None:
                continue
            b = np.asarray(bbox).reshape(-1)
            if b.size < 4:
                continue
            x1, y1, x2, y2 = map(float, b[:4])
            score = float(getattr(f, "det_score", 0.0))
            outs.append({"bbox": (x1, y1, x2, y2), "score": score})
    except Exception:
        logger.exception("FaceAnalysis error")
    return outs

def link_face_to_person(face_bbox: Tuple[float, float, float, float],
                        person_bbox: Tuple[int, int, int, int]) -> bool:
    if face_center_in(tuple(map(int, face_bbox)), person_bbox):
        return True
    if iou_xyxy(person_bbox, face_bbox) >= FACE_IOU_LINK:
        return True
    if inter_over_face(person_bbox, face_bbox) >= FACE_OVER_FACE_LINK:
        return True
    return False

# ------------------ RTSP helpers ------------------
def _configure_rtsp_capture(cap: cv2.VideoCapture) -> None:
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

def _read_latest_frame(cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
    grabbed = cap.grab()
    if not grabbed:
        ok, fr = cap.read()
        return ok, (fr if ok else None)
    for _ in range(3):
        if not cap.grab():
            break
    ok, frame = cap.retrieve()
    return ok, (frame if ok else None)

# ------------------ async workers (IO) ------------------
def start_workers_if_needed():
    global io_thread, io_queue
    if io_queue is None:
        io_queue = Queue(maxsize=IO_QUEUE_MAX)
    if io_thread is None or not io_thread.is_alive():
        io_thread = threading.Thread(target=io_worker, name="io_worker", daemon=True)
        io_thread.start()

def io_worker():
    while not stop_event.is_set():
        try:
            job = io_queue.get(timeout=0.1)  # type: ignore
        except Empty:
            continue
        except Exception:
            break
        try:
            crop_path: Path = job["crop_path"]
            _ensure_dir(crop_path.parent)
            cv2.imwrite(str(crop_path), job["image"])
            with csv_lock:
                with open(EMB_CSV, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(job["csv_row"])
        except Exception:
            logger.exception("IO worker error")

# ------------------ capture & processing loops ------------------
def capture_thread_fn(cam_idx: int, rtsp_url: str):
    q = cap_queues[cam_idx]
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error("[cap%d] cannot open RTSP: %s", cam_idx, rtsp_url)
        return
    _configure_rtsp_capture(cap)
    logger.info("[cap%d] started on %s", cam_idx, rtsp_url)
    try:
        while not stop_event.is_set():
            ok, frame = _read_latest_frame(cap)
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    break
            try:
                q.put_nowait(frame)
            except Exception:
                pass
    finally:
        cap.release()
        logger.info("[cap%d] stopped", cam_idx)

def embedding_loop_for_cam(cam_idx: int, rtsp_url: str):
    """
    Save crops only (unchanged semantics)
    """
    global current_person_id, current_person_name

    ensure_csv_header()
    start_workers_if_needed()

    user_folder = current_person_name or "unknown"
    body_dir = _gallery_body_dir(user_folder)
    face_dir = _gallery_face_dir(user_folder)

    frame_idx = 0
    logger.info("[Loop cam%d] started on %s", cam_idx, rtsp_url)
    q = cap_queues[cam_idx]

    try:
        while not stop_event.is_set():
            try:
                frame = q.get(timeout=0.2)
            except Empty:
                continue
            except Exception:
                break
            if frame is None:
                continue

            frame_idx += 1
            H, W = frame.shape[:2]
            detections: List[tuple[float, float, float, float, float]] = []

            vis = frame.copy()

            if yolo_model is not None:
                try:
                    with yolo_lock, torch.inference_mode():
                        res = yolo_model.predict(
                            frame,
                            conf=float(CONF_THRES),
                            iou=float(IOU_THRES),
                            imgsz=YOLO_IMGSZ,
                            verbose=False,
                            device=DEVICE if torch.cuda.is_available() and ("cuda" in str(DEVICE).lower()) else "cpu",
                        )
                    boxes = res[0].boxes if (res and len(res)) else None
                    if boxes is not None:
                        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                        confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
                        cls = boxes.cls.detach().cpu().numpy().astype(np.int32)
                        keep = (cls == 0)
                        xyxy, confs = xyxy[keep], confs[keep]
                        for (x1, y1, x2, y2), c in zip(xyxy, confs):
                            x1f = float(max(0, min(W - 1, x1)))
                            y1f = float(max(0, min(H - 1, y1)))
                            x2f = float(max(0, min(W - 1, x2)))
                            y2f = float(max(0, min(H - 1, y2)))
                            if (x2f - x1f) < 4 or (y2f - y1f) < 4:
                                continue
                            detections.append((x1f, y1f, x2f, y2f, float(c)))
                except Exception:
                    logger.exception("[Loop cam%d] YOLO error", cam_idx)

            faces = []
            if USE_FACE and (frame_idx % max(1, FACE_EVERY) == 0):
                faces = detect_faces_raw(frame)

            det_boxes_i: List[tuple[int, int, int, int, float]] = []
            for (x1, y1, x2, y2, conf) in detections:
                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                x1i, y1i = max(0, x1i), max(0, y1i)
                x2i, y2i = min(W, x2i), min(H, y2i)
                if x2i <= x1i or y2i <= y1i:
                    continue
                det_boxes_i.append((x1i, y1i, x2i, y2i, conf))

                try:
                    cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                    cv2.putText(
                        vis, f"{current_person_id or ''} cam{cam_idx}", (x1i, max(0, y1i - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )
                except Exception:
                    pass

                crop_bgr = frame[y1i:y2i, x1i:x2i]
                if crop_bgr.size > 0:
                    ts = time.time()
                    fname = f"person_{_epoch_ms()}.jpg"
                    path = body_dir / fname
                    row = [
                        current_person_id, current_person_name, ts,
                        cam_idx, frame_idx, len(det_boxes_i)-1,
                        x1i, y1i, x2i, y2i,
                        conf,
                        "", "", str(path), "body",
                    ]
                    try:
                        if io_queue is not None:
                            io_queue.put_nowait({"crop_path": path, "image": crop_bgr, "csv_row": row})
                    except Exception:
                        pass

            if det_boxes_i and faces:
                ts = time.time()
                for det_idx, (x1i, y1i, x2i, y2i, _conf) in enumerate(det_boxes_i):
                    t_xyxy = (x1i, y1i, x2i, y2i)
                    best_idx, best_score = -1, -1.0
                    best_box = None
                    for idx, fm in enumerate(faces):
                        if link_face_to_person(fm["bbox"], t_xyxy):
                            s = float(fm.get("score", 0.0))
                            if s > best_score:
                                best_score, best_idx = s, idx
                                best_box = fm["bbox"]
                    if best_idx < 0:
                        continue
                    fx1, fy1, fx2, fy2 = map(int, best_box)
                    if best_score < FACE_MIN_SCORE:
                        continue
                    if min(fx2 - fx1, fy2 - fy1) < FACE_MIN_SIZE:
                        continue

                    body_crop = frame[y1i:y2i, x1i:x2i]
                    if body_crop.size <= 0:
                        continue
                    fname = f"person_{_epoch_ms()}.jpg"
                    path = _gallery_face_dir(current_person_name or "unknown") / fname
                    row = [
                        current_person_id, current_person_name, ts,
                        cam_idx, frame_idx, det_idx,
                        x1i, y1i, x2i, y2i,
                        best_score,
                        "", "", str(path), "face",
                    ]
                    try:
                        if io_queue is not None:
                            io_queue.put_nowait({"crop_path": path, "image": body_crop, "csv_row": row})
                    except Exception:
                        pass

            with frames_lock:
                latest_frames[cam_idx] = vis

            if frame_idx % 50 == 0:
                logger.info("[Loop cam%d] frame %d, persons=%d, faces=%d",
                            cam_idx, frame_idx, len(det_boxes_i), len(faces) if faces else 0)

    finally:
        logger.info("[Loop cam%d] stopped", cam_idx)

# ------------------ viewer ------------------
def viewer_loop():
    logger.info("[Viewer] started")
    window_name = "Multi-RTSP Viewer"
    window_created = False
    try:
        while not stop_event.is_set():
            with frames_lock:
                frames = [
                    latest_frames[idx]
                    for idx in sorted(latest_frames.keys())
                    if latest_frames.get(idx) is not None
                ]

            if not frames:
                time.sleep(0.01)
                continue

            if not window_created:
                try:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    window_created = True
                except Exception:
                    pass

            min_h = min(f.shape[0] for f in frames)
            resized = []
            for f in frames:
                h, w = f.shape[:2]
                if h != min_h:
                    new_w = int(w * (min_h / h))
                    f = cv2.resize(f, (new_w, min_h))
                resized.append(f)

            try:
                combined = np.concatenate(resized, axis=1)
            except Exception:
                time.sleep(0.01)
                continue

            try:
                cv2.imshow(window_name, combined)
            except Exception:
                window_created = False
                time.sleep(0.01)
                continue

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                logger.info("[Viewer] key pressed, stopping...")
                stop_event.set()
                break

            time.sleep(VIEWER_SLEEP_MS / 1000.0)
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        logger.info("[Viewer] stopped")

# ------------------ finalize / extract from folders ------------------
def _list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    pats = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files: List[Path] = []
    for p in pats:
        files.extend(root.glob(p))
    return sorted(files)

def _load_bgr(path: Path) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return img if img is not None else None
    except Exception:
        return None

def _mean_embed(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
    if not vectors:
        return None
    arr = np.stack(vectors, axis=0).astype(np.float32)
    m = np.mean(arr, axis=0)
    return l2_normalize(m)

def _count_images(user_name: str) -> Tuple[int, int]:
    body = len(_list_images(_gallery_body_dir(user_name)))
    face = len(_list_images(_gallery_face_dir(user_name)))
    return body, face

# ======= quality helpers =======
def _ahash8(img_bgr: np.ndarray) -> int:
    try:
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA)
        avg = float(g.mean())
        bits = (g > avg).astype(np.uint8)
        out = 0
        for i, b in enumerate(bits.flatten()):
            out |= (int(b) & 1) << i
        return out
    except Exception:
        return random.getrandbits(64)

def _is_blurry(img_bgr: np.ndarray, var_thr: float = 30.0) -> bool:
    try:
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var()) < var_thr
    except Exception:
        return False

def _prep_reid(img_bgr: np.ndarray) -> np.ndarray:
    try:
        out = cv2.resize(img_bgr, (128, 256), interpolation=cv2.INTER_LINEAR)
    except Exception:
        out = img_bgr
    return _to_rgb(out)

# ---------- BODY embeddings from gallery (ENSEMBLE + TTA) ----------
def _compute_body_embedding_from_gallery(user_name: str,
                                         on_step: Optional[Callable[[int], None]] = None
                                         ) -> Optional[np.ndarray]:
    if not reid_extractors:
        logger.warning("TorchReID not available; skipping body embedding.")
        return None
    root = _gallery_body_dir(user_name)
    imgs = _list_images(root)
    if not imgs:
        logger.warning("No BODY crops found for '%s'", user_name)
        return None

    B = 64 if (torch.cuda.is_available() and ("cuda" in str(DEVICE).lower())) else 32
    vectors_per_image: List[np.ndarray] = []
    seen_hashes: set[int] = set()

    prepared: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
    for p in imgs:
        img = _load_bgr(p)
        if img is None or img.size == 0:
            if on_step: on_step(1)
            continue
        h, w = img.shape[:2]
        if min(h, w) < 32 or _is_blurry(img):
            if on_step: on_step(1)
            continue
        hval = _ahash8(img)
        if hval in seen_hashes:
            if on_step: on_step(1)
            continue
        seen_hashes.add(hval)

        rgb = _prep_reid(img)
        rgb_flip = cv2.flip(rgb, 1) if BODY_TTA_FLIP else None
        prepared.append((rgb, rgb_flip))
        if on_step: on_step(1)

    if not prepared:
        logger.warning("No valid BODY crops for '%s'", user_name)
        return None

    per_model_embs: List[List[np.ndarray]] = [ [] for _ in reid_extractors ]

    for midx, ext in enumerate(reid_extractors):
        batch_imgs: List[np.ndarray] = []
        idx_map: List[Tuple[int, int]] = []
        for i, (orig, flip) in enumerate(prepared):
            batch_imgs.append(orig); idx_map.append((i, 0))
            if flip is not None:
                batch_imgs.append(flip); idx_map.append((i, 1))

        feats_norm: List[np.ndarray] = []
        for start in range(0, len(batch_imgs), B):
            chunk = batch_imgs[start:start+B]
            try:
                with reid_lock, torch.inference_mode():
                    feats = ext(chunk)
                for f in feats:
                    f = f.detach().cpu().numpy() if hasattr(f, "detach") else np.asarray(f)
                    f512 = _as_512f(f)
                    feats_norm.append(f512 if f512 is not None else np.zeros((EXPECTED_EMBED_DIM,), np.float32))
            except Exception:
                logger.exception("TorchReID batch failed (model=%s)", REID_MODELS[midx])
                feats_norm.extend([np.zeros((EXPECTED_EMBED_DIM,), np.float32) for _ in chunk])

        img_accum = [ [] for _ in prepared ]
        for (img_i, _tta_i), feat in zip(idx_map, feats_norm):
            img_accum[img_i].append(feat)
        fused = [ l2_normalize(np.mean(np.stack(v, axis=0), axis=0)) if v else np.zeros((EXPECTED_EMBED_DIM,), np.float32)
                  for v in img_accum ]
        per_model_embs[midx] = fused

    for i in range(len(prepared)):
        parts = [ per_model_embs[m][i] for m in range(len(reid_extractors)) ]
        fused_img = l2_normalize(np.mean(np.stack(parts, axis=0), axis=0))
        vectors_per_image.append(fused_img)

    return _mean_embed(vectors_per_image)

# ---------- FACE embeddings from gallery (TTA) ----------
def _compute_face_embedding_from_gallery(user_name: str,
                                         on_step: Optional[Callable[[int], None]] = None
                                         ) -> Optional[np.ndarray]:
    if face_app is None:
        logger.warning("InsightFace not available; skipping face embedding.")
        return None
    root = _gallery_face_dir(user_name)
    imgs = _list_images(root)
    if not imgs:
        logger.warning("No FACE crops found for '%s'", user_name)
        return None

    vectors: List[np.ndarray] = []
    seen_hashes: set[int] = set()
    MAX_SIDE = 640
    MIN_BODY_SIDE = 40

    for p in imgs:
        img = _load_bgr(p)
        if img is None or img.size == 0:
            if on_step: on_step(1); continue
        h, w = img.shape[:2]
        if min(h, w) < MIN_BODY_SIDE or _is_blurry(img):
            if on_step: on_step(1); continue

        hval = _ahash8(img)
        if hval in seen_hashes:
            if on_step: on_step(1); continue
        seen_hashes.add(hval)

        if max(h, w) > MAX_SIDE:
            scale = MAX_SIDE / float(max(h, w))
            img_proc = cv2.resize(img, (max(1, int(w*scale)), max(1, int(h*scale))), interpolation=cv2.INTER_LINEAR)
        else:
            img_proc = img

        def _best_face(bgr_img) -> Optional[np.ndarray]:
            try:
                faces = face_app.get(np.ascontiguousarray(bgr_img))
            except Exception:
                faces = []
            best, best_score = None, -1.0
            for f in safe_iter_faces(faces):
                bbox = getattr(f, "bbox", None)
                if bbox is None: continue
                b = np.asarray(bbox).reshape(-1)
                if b.size < 4: continue
                x1, y1, x2, y2 = b[:4].astype(float)
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                score = float(getattr(f, "det_score", 0.0))
                ranking = score * (area ** 0.5)
                if ranking > best_score:
                    best, best_score = f, ranking
            if best is None:
                return None
            det_score = float(getattr(best, "det_score", 0.0))
            if det_score < FACE_MIN_SCORE:
                return None
            emb = extract_face_embedding(best)
            return _as_512f(emb)

        e0 = _best_face(img_proc)
        e1 = None
        if FACE_TTA_FLIP:
            try:
                e1 = _best_face(cv2.flip(img_proc, 1))
            except Exception:
                e1 = None

        feats = [e for e in (e0, e1) if e is not None]
        if feats:
            fused = l2_normalize(np.mean(np.stack(feats, axis=0), axis=0))
            vectors.append(fused)

        if on_step: on_step(1)

    if not vectors:
        logger.warning("No valid FACE embeddings for '%s'", user_name)
        return None
    return _mean_embed(vectors)

def _update_db_embeddings(user_id: int,
                          body_emb: Optional[np.ndarray],
                          face_emb: Optional[np.ndarray]) -> Dict[str, object]:
    session = get_session()
    try:
        user = session.query(User).filter(User.id == user_id).first()
        if not user:
            logger.warning("User ID %s not found. Skipping DB update.", user_id)
            return {"status": "error", "message": "user not found"}

        changed = False
        if body_emb is not None:
            user.body_embedding = body_emb.tolist()  # pgvector expects list
            changed = True
        if face_emb is not None:
            user.face_embedding = face_emb.tolist()
            changed = True
        if changed:
            user.last_embedding_update_ts = datetime.now(timezone.utc)
            session.commit()
            logger.info("DB updated: user=%s body=%s face=%s",
                        user_id, body_emb is not None, face_emb is not None)
            return {"status": "ok", "body": body_emb is not None, "face": face_emb is not None}
        else:
            logger.info("No embeddings to update for user=%s", user_id)
            return {"status": "no_embeddings"}
    except Exception:
        session.rollback()
        logger.exception("DB error on update")
        return {"status": "error", "message": "db error"}
    finally:
        session.close()

# ---- Background extraction worker (runs on POST /extract) ----
def _extract_worker(user_id: int, user_name: str):
    try:
        _progress_reset(user_id, user_name)
        _progress_set(user_id, stage="scanning", message="Counting images...")
        total_body, total_face = _count_images(user_name)
        if total_body + total_face == 0:
            _progress_set(user_id, total_body=0, total_face=0, percent=100,
                          stage="done", message="No images to process")
            return

        _progress_set(user_id, total_body=total_body, total_face=total_face,
                      done_body=0, done_face=0, percent=0)

        body_emb = None
        if total_body > 0 and reid_extractors:
            _progress_set(user_id, stage="embedding_body", message=f"Processing body ({total_body})")
            def on_body(n: int):
                st = get_progress_for_user(user_id)
                _progress_set(user_id, done_body=int(st["done_body"]) + int(n))
            body_emb = _compute_body_embedding_from_gallery(user_name, on_step=on_body)
        else:
            _progress_set(user_id, message="Skipping body (no images or model unavailable)")

        face_emb = None
        if total_face > 0 and face_app is not None:
            _progress_set(user_id, stage="embedding_face", message=f"Processing face ({total_face})")
            def on_face(n: int):
                st = get_progress_for_user(user_id)
                _progress_set(user_id, done_face=int(st["done_face"]) + int(n))
            face_emb = _compute_face_embedding_from_gallery(user_name, on_step=on_face)
        else:
            _progress_set(user_id, message="Skipping face (no images or model unavailable)")

        res = _update_db_embeddings(user_id, body_emb, face_emb)
        status = str(res.get("status", "unknown"))
        if status == "ok":
            _progress_set(user_id, stage="done", message="Embeddings saved", percent=100)
        else:
            _progress_set(user_id, stage="done", message=status, percent=100)
    except Exception as e:
        logger.exception("extract worker failed")
        _progress_set(user_id, stage="error", message=f"error: {e}")

def extract_embeddings_async(user_id: int) -> Dict[str, object]:
    """Kick off background extraction from saved galleries with progress reporting."""
    if not reid_extractors or (USE_FACE and face_app is None):
        init_models()

    session = get_session()
    try:
        user = session.query(User).filter(User.id == user_id).first()
        if not user:
            return {"status": "error", "message": f"user id {user_id} not found"}
        user_name = user.name
    except Exception:
        session.rollback()
        logger.exception("DB error during user lookup.")
        return {"status": "error", "message": "DB error"}
    finally:
        session.close()

    t = threading.Thread(target=_extract_worker, args=(user_id, user_name), daemon=True, name=f"extract-{user_id}")
    t.start()
    return {"status": "started", "id": user_id, "name": user_name}

# ---- NEW: synchronous extraction (immediate DB write) ----
def extract_embeddings_sync(user_id: int) -> Dict[str, object]:
    """Run extraction now and write embeddings to DB before returning."""
    if not reid_extractors or (USE_FACE and face_app is None):
        init_models()

    session = get_session()
    try:
        user = session.query(User).filter(User.id == user_id).first()
        if not user:
            return {"status": "error", "message": f"user id {user_id} not found"}
        user_name = user.name
    except Exception:
        session.rollback()
        logger.exception("DB error during user lookup.")
        return {"status": "error", "message": "DB error"}
    finally:
        session.close()

    _progress_reset(user_id, user_name)
    _progress_set(user_id, stage="scanning", message="Counting images...")
    total_body, total_face = _count_images(user_name)
    _progress_set(user_id, total_body=total_body, total_face=total_face)

    body_emb = _compute_body_embedding_from_gallery(user_name, on_step=lambda n: None) if total_body > 0 else None
    face_emb = _compute_face_embedding_from_gallery(user_name, on_step=lambda n: None) if total_face > 0 else None
    res = _update_db_embeddings(user_id, body_emb, face_emb)
    _progress_set(user_id, stage="done", percent=100, message=res.get("status", "done"))
    return res

# ------------------ control API ------------------
def start_extraction(user_id: int, show_viewer: bool = True) -> dict:
    """Start capture+crop across all RTSP streams for the given user_id."""
    global is_running, current_person_id, current_person_name
    global viewer_thread

    if is_running:
        return {"status": "ok", "message": "already running", "num_cams": len(RTSP_STREAMS),
                "id": current_person_id, "name": current_person_name}

    session = get_session()
    try:
        user = session.query(User).filter(User.id == int(user_id)).first()
        if not user:
            return {"status": "error", "message": f"user id {user_id} not found"}
        user_name = str(user.name)
    except Exception:
        session.rollback()
        logger.exception("DB error during user lookup (start_extraction).")
        return {"status": "error", "message": "DB error"}
    finally:
        session.close()

    if yolo_model is None or (USE_FACE and face_app is None):
        init_models()

    stop_event.clear()
    with frames_lock:
        latest_frames.clear()
    for d in (capture_threads, extract_threads, cap_queues):
        d.clear()

    _progress_reset(int(user_id), user_name)
    _progress_set(int(user_id), stage="starting", message="initializing")

    current_person_id = int(user_id)
    current_person_name = user_name

    start_workers_if_needed()

    for cam_idx, rtsp_url in enumerate(RTSP_STREAMS):
        cap_queues[cam_idx] = Queue(maxsize=CAP_QUEUE_MAX)
        t_cap = threading.Thread(target=capture_thread_fn, name=f"cap_{cam_idx}",
                                 args=(cam_idx, rtsp_url), daemon=True)
        t_ext = threading.Thread(target=embedding_loop_for_cam, name=f"ext_{cam_idx}",
                                 args=(cam_idx, rtsp_url), daemon=True)
        capture_threads[cam_idx] = t_cap
        extract_threads[cam_idx] = t_ext
        t_cap.start()
        t_ext.start()

    if show_viewer and (viewer_thread is None or not viewer_thread.is_alive()):
        viewer_thread = threading.Thread(target=viewer_loop, name="viewer", daemon=True)
        viewer_thread.start()

    is_running = True
    _progress_set(int(user_id), stage="running", message="processing")
    logger.info("Extraction started for user_id=%s name=%s on %d stream(s)",
                user_id, user_name, len(RTSP_STREAMS))

    return {"status": "ok", "message": "started", "num_cams": len(RTSP_STREAMS),
            "id": int(user_id), "name": user_name}

def stop_extraction(reason: str = "user") -> dict:
    """Gracefully stop all threads and reset state. Then auto-extract & store embeddings."""
    global is_running, current_person_id, current_person_name
    try:
        # snapshot the user before wiping globals
        uid_snapshot = current_person_id
        uname_snapshot = current_person_name

        stop_event.set()

        for d in (capture_threads, extract_threads):
            for idx, th in list(d.items()):
                try:
                    th.join(timeout=1.5)
                except Exception:
                    pass

        with frames_lock:
            latest_frames.clear()
        capture_threads.clear()
        extract_threads.clear()
        cap_queues.clear()

        if viewer_thread is not None and viewer_thread.is_alive():
            try:
                pass
            except Exception:
                pass

        is_running = False
        logger.info("Extraction stopped (%s).", reason)

        # --------- NEW: auto-trigger background embedding write to DB ----------
        if uid_snapshot is not None:
            if not reid_extractors or (USE_FACE and face_app is None):
                init_models()
            # spawn a detached worker to compute & store embeddings for the just-stopped user
            threading.Thread(
                target=_extract_worker,
                args=(int(uid_snapshot), uname_snapshot or "unknown"),
                name=f"auto-extract-{uid_snapshot}",
                daemon=True,
            ).start()
            logger.info("Auto-extract triggered for user_id=%s after stop.", uid_snapshot)
        # ----------------------------------------------------------------------

        return {"status": "ok", "message": f"stopped ({reason})"}
    finally:
        current_person_id = None
        current_person_name = None

def remove_embeddings(id: int):
    """Remove stored embeddings (body + face)."""
    session = get_session()
    try:
        user = session.query(User).filter(User.id == id).first()
        if not user:
            logger.error("User with id %d not found. Skipping embedding removal.", id)
            return {"status": "error", "message": f"user id {id} not found" }
        user.body_embedding = None
        user.face_embedding = None
        user.last_embedding_update_ts = datetime.now(timezone.utc)
        session.commit()
        logger.warning("Embeddings removed for user id=%d", id)
        return {"status": "ok", "id": id}
    except Exception:
        session.rollback()
        logger.exception("remove embeddings: DB error during user lookup.")
        return {"status":"error", "message":"DB error"}
    finally:
        session.close()

def get_status():
    """Return current extract_service status."""
    return {
        "running": is_running,
        "num_cams": len(RTSP_STREAMS),
        "rtsp_streams": RTSP_STREAMS,
        "id": current_person_id,
    }



