"""
mvision/services/extract_service.py

Multi-RTSP extraction & embedding service (Postgres + pgvector).

- Saves per-crop records to CSV and crop files.
- Updates the User.embedding (pgvector) and User.last_embedding_update_ts (UTC).
- Does NOT assume a separate Embedding model/table (your project uses only users table).
- Minimal throttling applied to user updates to reduce DB write pressure (configurable).
"""
from __future__ import annotations

import csv
import time
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict

import logging

import cv2
import numpy as np
import torch

from sqlalchemy.sql import func

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

# optional imports
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from torchreid.utils.feature_extractor import FeatureExtractor as TorchreidExtractor
except Exception as e:
    print("TorchReID import error:", e)
    TorchreidExtractor = None

# -------------------------
# Config / constants
# -------------------------
# Must match your pgvector size on User.embedding
EXPECTED_EMBED_DIM = 512

# Minimum seconds between updates to a single user's embedding (to reduce DB writes)
EMB_UPDATE_MIN_INTERVAL = 2.0  # seconds, adjust as needed

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("mvision.extract_service")
if not logger.handlers:
    # simple default handler if app hasn't configured logging
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# -------------------------
# Runtime globals & locks
# -------------------------
yolo_model: YOLO | None = None
reid_extractor: TorchreidExtractor | None = None

yolo_lock = threading.Lock()
reid_lock = threading.Lock()
csv_lock = threading.Lock()
db_lock = threading.Lock()
frames_lock = threading.Lock()

latest_frames: Dict[int, np.ndarray] = {}
viewer_thread: Optional[threading.Thread] = None
extract_threads: Dict[int, threading.Thread] = {}
stop_event = threading.Event()
is_running = False
current_person_id: Optional[int] = None
current_person_name: Optional[str] = None

# track last update time per user to throttle DB writes
_last_user_update_ts: Dict[int, float] = {}


# ------------------ DB helpers ------------------
def init_db():
    """Create tables if not exists (uses SQLAlchemy models)."""
    # Base.metadata.create_all(bind=engine)
    logger.info("DB initialized...")


def upsert_user_embedding_by_id(user_id: int, emb_list: List[float]):
    """
    Update only embedding and last_embedding_update_ts for a given user ID.
    Does NOT create a new user. Does NOT modify any other fields.

    Returns:
        User ORM instance on success,
        None if user not found or dimension mismatch.
    """

    # ensure float list
    if not isinstance(emb_list, list):
        try:
            emb_list = list(map(float, emb_list))
        except Exception:
            logger.exception("Failed to convert embedding to list of floats.")
            return None

    # dimension check
    if len(emb_list) != EXPECTED_EMBED_DIM:
        logger.warning(
            "Embedding dimension mismatch for id=%s: expected %d but got %d",
            user_id,
            EXPECTED_EMBED_DIM,
            len(emb_list),
        )
        return None

    with db_lock:
        session = get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()

            if not user:
                logger.warning("User ID %s not found. Skipping embedding update.", user_id)
                return None

            # update only 2 fields
            user.embedding = emb_list
            user.last_embedding_update_ts = datetime.now(timezone.utc)

            session.commit()
            session.refresh(user)
            logger.info("Updated embedding for user id=%s", user_id)
            return user

        except Exception:
            session.rollback()
            logger.exception("DB error while updating embedding for id=%s", user_id)
            return None
        finally:
            session.close()



# ------------------ utility ------------------
def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n == 0 or not np.isfinite(n):
        return v
    return v / n


# ------------------ models init ------------------
def init_models():
    """
    Load YOLO + TorchReID once on startup.
    """
    global yolo_model, reid_extractor

    gpu = torch.cuda.is_available() and ("cuda" in DEVICE)
    logger.info("init_models device=%s gpu_available=%s", DEVICE, torch.cuda.is_available())

    # YOLO
    if YOLO is not None:
        try:
            weights = YOLO_WEIGHTS
            if not Path(weights).exists():
                logger.warning("%s not found, falling back to yolov8n.pt", weights)
                weights = "yolov8n.pt"
            yolo_model = YOLO(weights)
            if gpu:
                try:
                    yolo_model.to(DEVICE)
                except Exception:
                    # some YOLO wrapper ignore .to(); ignore failures
                    pass
            logger.info("YOLO ready")
        except Exception:
            logger.exception("YOLO load failed; detection disabled.")
            yolo_model = None
    else:
        logger.warning("ultralytics not installed; detection disabled.")

    # TorchReID
    if TorchreidExtractor is not None:
        try:
            reid_extractor = TorchreidExtractor(
                model_name="osnet_x0_25",
                device=(DEVICE if gpu else "cpu"),
            )
            logger.info("TorchReID extract_service ready")
        except Exception:
            logger.exception("TorchReID init failed; embeddings disabled.")
            reid_extractor = None
    else:
        logger.warning("torchreid not installed; embeddings disabled.")


# ------------------ CSV + crops helpers ------------------
def ensure_csv_header():
    if not Path(EMB_CSV).exists():
        with csv_lock:
            with open(EMB_CSV, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "user_id",
                        "ts",
                        "cam_idx",
                        "frame_idx",
                        "det_idx",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                        "conf",
                        "embedding",
                        "crop_path",
                    ]
                )
        logger.info("Created CSV header: %s", EMB_CSV)


def ensure_crops_dir(cam_idx: int) -> Path:
    cam_dir = Path(CROPS_ROOT) / f"{current_person_name}" / f"cam{cam_idx}"
    cam_dir.mkdir(parents=True, exist_ok=True)
    return cam_dir


# ------------------ worker loops ------------------
def _should_update_user(id: int) -> bool:
    """Return True if sufficient time passed since last update for this user."""
    now = time.time()
    last = _last_user_update_ts.get(id)
    if last is None or (now - last) >= EMB_UPDATE_MIN_INTERVAL:
        _last_user_update_ts[id] = now
        return True
    return False


def embedding_loop_for_cam(cam_idx: int, rtsp_url: str):
    """
    Open RTSP for one camera, run YOLO on persons,
    draw bounding boxes, extract embeddings, save crops + CSV,
    update User.embedding and last_embedding_update_ts for current_person_id (throttled).
    """
    global current_person_id
    global current_person_name

    ensure_csv_header()
    cam_dir = ensure_crops_dir(cam_idx)

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error("[cam%d] cannot open RTSP: %s", cam_idx, rtsp_url)
        return

    frame_idx = 0
    logger.info("[Loop cam%d] started on %s", cam_idx, rtsp_url)

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning("[Loop cam%d] frame read failed, breaking", cam_idx)
                break

            frame_idx += 1
            H, W = frame.shape[:2]
            detections = []  # list of (x1, y1, x2, y2, conf)
            vis = frame.copy()

            # --- YOLO detection ---
            if yolo_model is not None:
                try:
                    with yolo_lock:
                        res = yolo_model(frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
                    boxes = res[0].boxes if (res and len(res)) else None
                    if boxes is not None:
                        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                        confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
                        cls = boxes.cls.detach().cpu().numpy().astype(np.int32)

                        keep = (cls == 0)  # class 0 = person
                        xyxy, confs = xyxy[keep], confs[keep]

                        for (x1, y1, x2, y2), c in zip(xyxy, confs):
                            x1f = float(max(0, min(W - 1, x1)))
                            y1f = float(max(0, min(H - 1, y1)))
                            x2f = float(max(0, min(W - 1, x2)))
                            y2f = float(max(0, min(H - 1, y2)))
                            detections.append((x1f, y1f, x2f, y2f, float(c)))
                except Exception:
                    logger.exception("[Loop cam%d] YOLO error", cam_idx)

            # --- Embedding extraction + CSV + user update ---
            if reid_extractor is not None and detections and current_person_id:
                det_idx = 0
                for (x1, y1, x2, y2, conf) in detections:
                    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                    x1i, y1i = max(0, x1i), max(0, y1i)
                    x2i, y2i = min(W, x2i), min(H, y2i)

                    if x2i <= x1i or y2i <= y1i:
                        continue

                    # draw box
                    cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                    label = f"{current_person_id} cam{cam_idx}"
                    cv2.putText(
                        vis,
                        label,
                        (x1i, max(0, y1i - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                    crop = frame[y1i:y2i, x1i:x2i]

                    # save crop
                    crop_filename = f"frame_{current_person_id}_{current_person_name}_{frame_idx:06d}_det{det_idx}.jpg"
                    crop_path = cam_dir / crop_filename
                    try:
                        cv2.imwrite(str(crop_path), crop)
                    except Exception:
                        logger.exception("[Loop cam%d] crop save error", cam_idx)
                        continue

                    # extract embedding
                    try:
                        with reid_lock:
                            feats = reid_extractor([crop])  # list-of-image API

                        feat = feats[0]
                        if hasattr(feat, "detach"):
                            feat = feat.detach().cpu().numpy()
                        elif not isinstance(feat, np.ndarray):
                            feat = np.asarray(feat)

                        feat = l2_normalize(feat)
                        emb_list = feat.tolist()
                        emb_str = " ".join(f"{v:.6f}" for v in feat.tolist())
                        ts = time.time()

                        # append to CSV
                        with csv_lock:
                            with open(EMB_CSV, "a", newline="", encoding="utf-8") as f:
                                w = csv.writer(f)
                                w.writerow(
                                    [
                                        current_person_id,
                                        current_person_name,
                                        ts,
                                        cam_idx,
                                        frame_idx,
                                        det_idx,
                                        x1i,
                                        y1i,
                                        x2i,
                                        y2i,
                                        conf,
                                        emb_str,
                                        str(crop_path),
                                    ]
                                )

                        # Update User.embedding (throttled)
                        try:
                            if _should_update_user(current_person_id):
                                upsert_user_embedding_by_id(current_person_id, emb_list)
                        except Exception:
                            logger.exception(
                                "[Loop cam%d] failed updating user embedding for '%s'",
                                cam_idx,
                                current_person_id,
                            )

                        det_idx += 1
                    except Exception:
                        logger.exception("[Loop cam%d] embedding/processing error", cam_idx)
            else:
                # draw boxes for visualization only
                for (x1, y1, x2, y2, conf) in detections:
                    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                    x1i, y1i = max(0, x1i), max(0, y1i)
                    x2i, y2i = min(W, x2i), min(H, y2i)
                    if x2i <= x1i or y2i <= y1i:
                        continue
                    cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                    label = f"cam{cam_idx}"
                    cv2.putText(
                        vis,
                        label,
                        (x1i, max(0, y1i - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            # store latest frame
            with frames_lock:
                latest_frames[cam_idx] = vis

            # periodic log
            if frame_idx % 50 == 0:
                logger.info("[Loop cam%d] frame %d, detections=%d", cam_idx, frame_idx, len(detections))

    finally:
        cap.release()
        logger.info("[Loop cam%d] stopped", cam_idx)


# ------------------ viewer ------------------
def viewer_loop():
    """Combine all latest_frames horizontally and show in one OpenCV window."""
    logger.info("[Viewer] started")
    window_name = "Multi-RTSP Viewer"
    try:
        while not stop_event.is_set():
            with frames_lock:
                if not latest_frames:
                    frames = []
                else:
                    frames = [
                        latest_frames[idx]
                        for idx in sorted(latest_frames.keys())
                        if latest_frames[idx] is not None
                    ]

            if not frames:
                time.sleep(0.01)
                continue

            # make same height
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

            cv2.imshow(window_name, combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                logger.info("[Viewer] key pressed, stopping...")
                stop_event.set()
                break

            time.sleep(0.01)
    finally:
        cv2.destroyAllWindows()
        logger.info("[Viewer] stopped")


# ------------------ control API used by routes ------------------
def start_extraction(id: int):
    """
    Start extraction threads for all RTSP streams and viewer thread.
    id: the user_id whose embeddings will be updated.
    """
    global extract_threads, viewer_thread, is_running
    global current_person_id, current_person_name

    if is_running:
        logger.warning("Extraction already running for '%s'", current_person_name)
        return {"status": "already_running", "id": current_person_id, "name": current_person_name}

    if not RTSP_STREAMS:
        logger.error("RTSP_STREAMS is empty.")
        return {"status": "error", "message": "RTSP_STREAMS empty"}

    # -----------------------------------------
    # Fetch user name based on ID
    # -----------------------------------------
    session = get_session()
    try:
        user = session.query(User).filter(User.id == id).first()

        if not user:
            logger.error("User with id %d not found. Extraction aborted.", id)
            return {"status": "error", "message": f"user id {id} not found"}

        current_person_id = id
        current_person_name = user.name  # SET NAME HERE

        logger.info("Extraction will run for user: id=%d name='%s'",
                    current_person_id, current_person_name)

    except Exception as e:
        session.rollback()
        logger.exception("DB error during user lookup.")
        return {"status": "error", "message": "DB error"}
    finally:
        session.close()

    # -----------------------------------------
    # start extraction threads
    # -----------------------------------------
    stop_event.clear()
    extract_threads = {}

    with frames_lock:
        latest_frames.clear()

    for idx, url in enumerate(RTSP_STREAMS):
        t = threading.Thread(target=embedding_loop_for_cam, args=(idx, url), daemon=True)
        extract_threads[idx] = t
        t.start()

    viewer_thread = threading.Thread(target=viewer_loop, daemon=True)
    viewer_thread.start()

    is_running = True

    return {
        "status": "started",
        "num_cams": len(RTSP_STREAMS),
        "id": current_person_id,
        "name": current_person_name,
    }



def stop_extraction():
    """
    Stop extraction and viewer threads gracefully.
    """
    global extract_threads, is_running, viewer_thread, current_person_id

    if not is_running:
        logger.info("stop_extraction called but extract_service is not running.")
        return {"status": "not_running"}

    stop_event.set()
    for idx, t in extract_threads.items():
        try:
            t.join(timeout=10.0)
        except Exception:
            logger.exception("Error joining thread for cam %d", idx)

    if viewer_thread is not None:
        try:
            viewer_thread.join(timeout=10.0)
        except Exception:
            logger.exception("Error joining viewer thread")

    extract_threads = {}
    is_running = False
    # clear current person id (optional)
    current_person_id = None
    logger.info("Extraction stopped.")
    return {"status": "stopped"}


def get_status():
    """Return current extract_service status."""
    return {
        "running": is_running,
        "num_cams": len(RTSP_STREAMS),
        "rtsp_streams": RTSP_STREAMS,
        "id": current_person_id,
    }
