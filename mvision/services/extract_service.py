# """
# mvision/services/extract_service.py

# Multi-RTSP extraction & body_embedding service (Postgres + pgvector).

# - Saves per-crop records to CSV and crop files.
# - Updates the User.body_embedding (pgvector) and User.last_embedding_update_ts (UTC). 
# - Minimal throttling applied to user updates to reduce DB write pressure (configurable).
# """
# from __future__ import annotations

# import csv
# import time
# import threading
# from pathlib import Path
# from datetime import datetime, timezone
# from typing import Optional, List, Dict

# import logging

# import cv2
# import numpy as np
# import torch

# from sqlalchemy.sql import func

# from mvision.constants import (
#     RTSP_STREAMS,
#     YOLO_WEIGHTS,
#     DEVICE,
#     CONF_THRES,
#     IOU_THRES,
#     EMB_CSV,
#     CROPS_ROOT,
# )
# from mvision.db.session import get_session, engine
# from mvision.db.models import Base, User

# # optional imports
# try:
#     from ultralytics import YOLO
# except Exception:
#     YOLO = None

# try:
#     from torchreid.utils.feature_extractor import FeatureExtractor as TorchreidExtractor
# except Exception as e:
#     print("TorchReID import error:", e)
#     TorchreidExtractor = None

# # -------------------------
# # Config / constants
# # -------------------------
# # Must match your pgvector size on User.body_embedding
# EXPECTED_EMBED_DIM = 512

# # Minimum seconds between updates to a single user's body_embedding (to reduce DB writes)
# EMB_UPDATE_MIN_INTERVAL = 2.0  # seconds, adjust as needed

# # -------------------------
# # Logging
# # -------------------------
# logger = logging.getLogger("mvision.extract_service")
# if not logger.handlers:
#     # simple default handler if app hasn't configured logging
#     ch = logging.StreamHandler()
#     ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
#     logger.addHandler(ch)
# logger.setLevel(logging.INFO)

# # -------------------------
# # Runtime globals & locks
# # -------------------------
# yolo_model: YOLO | None = None
# reid_extractor: TorchreidExtractor | None = None

# yolo_lock = threading.Lock()
# reid_lock = threading.Lock()
# csv_lock = threading.Lock()
# db_lock = threading.Lock()
# frames_lock = threading.Lock()

# latest_frames: Dict[int, np.ndarray] = {}
# viewer_thread: Optional[threading.Thread] = None
# extract_threads: Dict[int, threading.Thread] = {}
# stop_event = threading.Event()
# is_running = False
# current_person_id: Optional[int] = None
# current_person_name: Optional[str] = None

# # track last update time per user to throttle DB writes
# _last_user_update_ts: Dict[int, float] = {}


# # ------------------ DB helpers ------------------
# def init_db():
#     """Create tables if not exists (uses SQLAlchemy models)."""
#     # Base.metadata.create_all(bind=engine)
#     logger.info("DB initialized...")


# def upsert_user_embedding_by_id(user_id: int, emb_list: List[float]):
#     """
#     Update only body_embedding and last_embedding_update_ts for a given user ID.
#     Does NOT create a new user. Does NOT modify any other fields.

#     Returns:
#         User ORM instance on success,
#         None if user not found or dimension mismatch.
#     """

#     # ensure float list
#     if not isinstance(emb_list, list):
#         try:
#             emb_list = list(map(float, emb_list))
#         except Exception:
#             logger.exception("Failed to convert body_embedding to list of floats.")
#             return None

#     # dimension check
#     if len(emb_list) != EXPECTED_EMBED_DIM:
#         logger.warning(
#             "Embedding dimension mismatch for id=%s: expected %d but got %d",
#             user_id,
#             EXPECTED_EMBED_DIM,
#             len(emb_list),
#         )
#         return None

#     with db_lock:
#         session = get_session()
#         try:
#             user = session.query(User).filter(User.id == user_id).first()

#             if not user:
#                 logger.warning("User ID %s not found. Skipping body_embedding update.", user_id)
#                 return None

#             # update only 2 fields
#             user.body_embedding = emb_list
#             user.last_embedding_update_ts = datetime.now(timezone.utc)

#             session.commit()
#             session.refresh(user)
#             logger.info("Updated body_embedding for user id=%s", user_id)
#             return user

#         except Exception:
#             session.rollback()
#             logger.exception("DB error while updating body_embedding for id=%s", user_id)
#             return None
#         finally:
#             session.close()



# # ------------------ utility ------------------
# def l2_normalize(v: np.ndarray) -> np.ndarray:
#     v = np.asarray(v, dtype=np.float32)
#     n = np.linalg.norm(v)
#     if n == 0 or not np.isfinite(n):
#         return v
#     return v / n


# # ------------------ models init ------------------
# def init_models():
#     """
#     Load YOLO + TorchReID once on startup.
#     """
#     global yolo_model, reid_extractor

#     gpu = torch.cuda.is_available() and ("cuda" in DEVICE)
#     logger.info("init_models device=%s gpu_available=%s", DEVICE, torch.cuda.is_available())

#     # YOLO
#     if YOLO is not None:
#         try:
#             weights = YOLO_WEIGHTS
#             if not Path(weights).exists():
#                 logger.warning("%s not found, falling back to yolov8n.pt", weights)
#                 weights = "yolov8n.pt"
#             yolo_model = YOLO(weights)
#             if gpu:
#                 try:
#                     yolo_model.to(DEVICE)
#                 except Exception:
#                     # some YOLO wrapper ignore .to(); ignore failures
#                     pass
#             logger.info("YOLO ready")
#         except Exception:
#             logger.exception("YOLO load failed; detection disabled.")
#             yolo_model = None
#     else:
#         logger.warning("ultralytics not installed; detection disabled.")

#     # TorchReID
#     if TorchreidExtractor is not None:
#         try:
#             reid_extractor = TorchreidExtractor(
#                 model_name="osnet_x0_25",
#                 device=(DEVICE if gpu else "cpu"),
#             )
#             logger.info("TorchReID extract_service ready")
#         except Exception:
#             logger.exception("TorchReID init failed; embeddings disabled.")
#             reid_extractor = None
#     else:
#         logger.warning("torchreid not installed; embeddings disabled.")


# # ------------------ CSV + crops helpers ------------------
# def ensure_csv_header():
#     if not Path(EMB_CSV).exists():
#         with csv_lock:
#             with open(EMB_CSV, "w", newline="", encoding="utf-8") as f:
#                 w = csv.writer(f)
#                 w.writerow(
#                     [
#                         "user_id",
#                         "ts",
#                         "cam_idx",
#                         "frame_idx",
#                         "det_idx",
#                         "x1",
#                         "y1",
#                         "x2",
#                         "y2",
#                         "conf",
#                         "body_embedding",
#                         "crop_path",
#                     ]
#                 )
#         logger.info("Created CSV header: %s", EMB_CSV)


# def ensure_crops_dir(cam_idx: int) -> Path:
#     cam_dir = Path(CROPS_ROOT) / f"{current_person_name}" / f"cam{cam_idx}"
#     cam_dir.mkdir(parents=True, exist_ok=True)
#     return cam_dir


# # ------------------ worker loops ------------------
# def _should_update_user(id: int) -> bool:
#     """Return True if sufficient time passed since last update for this user."""
#     now = time.time()
#     last = _last_user_update_ts.get(id)
#     if last is None or (now - last) >= EMB_UPDATE_MIN_INTERVAL:
#         _last_user_update_ts[id] = now
#         return True
#     return False


# def embedding_loop_for_cam(cam_idx: int, rtsp_url: str):
#     """
#     Open RTSP for one camera, run YOLO on persons,
#     draw bounding boxes, extract embeddings, save crops + CSV,
#     update User.body_embedding and last_embedding_update_ts for current_person_id (throttled).
#     """
#     global current_person_id
#     global current_person_name

#     ensure_csv_header()
#     cam_dir = ensure_crops_dir(cam_idx)

#     cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
#     if not cap.isOpened():
#         logger.error("[cam%d] cannot open RTSP: %s", cam_idx, rtsp_url)
#         return

#     frame_idx = 0
#     logger.info("[Loop cam%d] started on %s", cam_idx, rtsp_url)

#     try:
#         while not stop_event.is_set():
#             ok, frame = cap.read()
#             if not ok or frame is None:
#                 logger.warning("[Loop cam%d] frame read failed, breaking", cam_idx)
#                 break

#             frame_idx += 1
#             H, W = frame.shape[:2]
#             detections = []  # list of (x1, y1, x2, y2, conf)
#             vis = frame.copy()

#             # --- YOLO detection ---
#             if yolo_model is not None:
#                 try:
#                     with yolo_lock:
#                         res = yolo_model(frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
#                     boxes = res[0].boxes if (res and len(res)) else None
#                     if boxes is not None:
#                         xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
#                         confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
#                         cls = boxes.cls.detach().cpu().numpy().astype(np.int32)

#                         keep = (cls == 0)  # class 0 = person
#                         xyxy, confs = xyxy[keep], confs[keep]

#                         for (x1, y1, x2, y2), c in zip(xyxy, confs):
#                             x1f = float(max(0, min(W - 1, x1)))
#                             y1f = float(max(0, min(H - 1, y1)))
#                             x2f = float(max(0, min(W - 1, x2)))
#                             y2f = float(max(0, min(H - 1, y2)))
#                             detections.append((x1f, y1f, x2f, y2f, float(c)))
#                 except Exception:
#                     logger.exception("[Loop cam%d] YOLO error", cam_idx)

#             # --- Embedding extraction + CSV + user update ---
#             if reid_extractor is not None and detections and current_person_id:
#                 det_idx = 0
#                 for (x1, y1, x2, y2, conf) in detections:
#                     x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
#                     x1i, y1i = max(0, x1i), max(0, y1i)
#                     x2i, y2i = min(W, x2i), min(H, y2i)

#                     if x2i <= x1i or y2i <= y1i:
#                         continue

#                     # draw box
#                     cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
#                     label = f"{current_person_id} cam{cam_idx}"
#                     cv2.putText(
#                         vis,
#                         label,
#                         (x1i, max(0, y1i - 5)),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6,
#                         (0, 255, 0),
#                         2,
#                     )

#                     crop = frame[y1i:y2i, x1i:x2i]

#                     # save crop
#                     crop_filename = f"frame_{current_person_id}_{current_person_name}_{frame_idx:06d}_det{det_idx}.jpg"
#                     crop_path = cam_dir / crop_filename
#                     try:
#                         cv2.imwrite(str(crop_path), crop)
#                     except Exception:
#                         logger.exception("[Loop cam%d] crop save error", cam_idx)
#                         continue

#                     # extract body_embedding
#                     try:
#                         with reid_lock:
#                             feats = reid_extractor([crop])  # list-of-image API

#                         feat = feats[0]
#                         if hasattr(feat, "detach"):
#                             feat = feat.detach().cpu().numpy()
#                         elif not isinstance(feat, np.ndarray):
#                             feat = np.asarray(feat)

#                         feat = l2_normalize(feat)
#                         emb_list = feat.tolist()
#                         body_embedding_str = " ".join(f"{v:.6f}" for v in feat.tolist())
#                         ts = time.time()

#                         # append to CSV
#                         with csv_lock:
#                             with open(EMB_CSV, "a", newline="", encoding="utf-8") as f:
#                                 w = csv.writer(f)
#                                 w.writerow(
#                                     [
#                                         current_person_id,
#                                         current_person_name,
#                                         ts,
#                                         cam_idx,
#                                         frame_idx,
#                                         det_idx,
#                                         x1i,
#                                         y1i,
#                                         x2i,
#                                         y2i,
#                                         conf,
#                                         body_embedding_str,
#                                         str(crop_path),
#                                     ]
#                                 )

#                         # Update User.body_embedding (throttled)
#                         try:
#                             if _should_update_user(current_person_id):
#                                 upsert_user_embedding_by_id(current_person_id, emb_list)
#                         except Exception:
#                             logger.exception(
#                                 "[Loop cam%d] failed updating user body_embedding for '%s'",
#                                 cam_idx,
#                                 current_person_id,
#                             )

#                         det_idx += 1
#                     except Exception:
#                         logger.exception("[Loop cam%d] body_embedding/processing error", cam_idx)
#             else:
#                 # draw boxes for visualization only
#                 for (x1, y1, x2, y2, conf) in detections:
#                     x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
#                     x1i, y1i = max(0, x1i), max(0, y1i)
#                     x2i, y2i = min(W, x2i), min(H, y2i)
#                     if x2i <= x1i or y2i <= y1i:
#                         continue
#                     cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
#                     label = f"cam{cam_idx}"
#                     cv2.putText(
#                         vis,
#                         label,
#                         (x1i, max(0, y1i - 5)),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6,
#                         (0, 255, 0),
#                         2,
#                     )

#             # store latest frame
#             with frames_lock:
#                 latest_frames[cam_idx] = vis

#             # periodic log
#             if frame_idx % 50 == 0:
#                 logger.info("[Loop cam%d] frame %d, detections=%d", cam_idx, frame_idx, len(detections))

#     finally:
#         cap.release()
#         logger.info("[Loop cam%d] stopped", cam_idx)


# # ------------------ viewer ------------------
# def viewer_loop():
#     """Combine all latest_frames horizontally and show in one OpenCV window."""
#     logger.info("[Viewer] started")
#     window_name = "Multi-RTSP Viewer"
#     try:
#         while not stop_event.is_set():
#             with frames_lock:
#                 if not latest_frames:
#                     frames = []
#                 else:
#                     frames = [
#                         latest_frames[idx]
#                         for idx in sorted(latest_frames.keys())
#                         if latest_frames[idx] is not None
#                     ]

#             if not frames:
#                 time.sleep(0.01)
#                 continue

#             # make same height
#             min_h = min(f.shape[0] for f in frames)
#             resized = []
#             for f in frames:
#                 h, w = f.shape[:2]
#                 if h != min_h:
#                     new_w = int(w * (min_h / h))
#                     f = cv2.resize(f, (new_w, min_h))
#                 resized.append(f)

#             try:
#                 combined = np.concatenate(resized, axis=1)
#             except Exception:
#                 time.sleep(0.01)
#                 continue

#             cv2.imshow(window_name, combined)
#             key = cv2.waitKey(1) & 0xFF
#             if key in (ord("q"), 27):  # q or ESC
#                 logger.info("[Viewer] key pressed, stopping...")
#                 stop_event.set()
#                 break

#             time.sleep(0.01)
#     finally:
#         cv2.destroyAllWindows()
#         logger.info("[Viewer] stopped")


# # ------------------ control API used by routes ------------------
# def start_extraction(id: int):
#     """
#     Start extraction threads for all RTSP streams and viewer thread.
#     id: the user_id whose embeddings will be updated.
#     """
#     global extract_threads, viewer_thread, is_running
#     global current_person_id, current_person_name

#     if is_running:
#         logger.warning("Extraction already running for '%s'", current_person_name)
#         return {"status": "already_running", "id": current_person_id, "name": current_person_name}

#     if not RTSP_STREAMS:
#         logger.error("RTSP_STREAMS is empty.")
#         return {"status": "error", "message": "RTSP_STREAMS empty"}

#     # -----------------------------------------
#     # Fetch user name based on ID
#     # -----------------------------------------
#     session = get_session()
#     try:
#         user = session.query(User).filter(User.id == id).first()

#         if not user:
#             logger.error("User with id %d not found. Extraction aborted.", id)
#             return {"status": "error", "message": f"user id {id} not found"}

#         current_person_id = id
#         current_person_name = user.name  # SET NAME HERE

#         logger.info("Extraction will run for user: id=%d name='%s'",
#                     current_person_id, current_person_name)

#     except Exception as e:
#         session.rollback()
#         logger.exception("DB error during user lookup.")
#         return {"status": "error", "message": "DB error"}
#     finally:
#         session.close()

#     # -----------------------------------------
#     # start extraction threads
#     # -----------------------------------------
#     stop_event.clear()
#     extract_threads = {}

#     with frames_lock:
#         latest_frames.clear()

#     for idx, url in enumerate(RTSP_STREAMS):
#         t = threading.Thread(target=embedding_loop_for_cam, args=(idx, url), daemon=True)
#         extract_threads[idx] = t
#         t.start()

#     viewer_thread = threading.Thread(target=viewer_loop, daemon=True)
#     viewer_thread.start()

#     is_running = True

#     return {
#         "status": "started",
#         "num_cams": len(RTSP_STREAMS),
#         "id": current_person_id,
#         "name": current_person_name,
#     }



# def stop_extraction():
#     """
#     Stop extraction and viewer threads gracefully.
#     """
#     global extract_threads, is_running, viewer_thread, current_person_id

#     if not is_running:
#         logger.info("stop_extraction called but extract_service is not running.")
#         return {"status": "not_running"}

#     stop_event.set()
#     for idx, t in extract_threads.items():
#         try:
#             t.join(timeout=10.0)
#         except Exception:
#             logger.exception("Error joining thread for cam %d", idx)

#     if viewer_thread is not None:
#         try:
#             viewer_thread.join(timeout=10.0)
#         except Exception:
#             logger.exception("Error joining viewer thread")

#     extract_threads = {}
#     is_running = False
#     # clear current person id (optional)
#     current_person_id = None
#     logger.info("Extraction stopped.")
#     return {"status": "stopped"}


# def remove_embeddings(id: int):
#     """    
#     Remove stored embeddings
#     """  
#     print('--------------------------------',id)
#     session = get_session()
#     try: 
#         user = session.query(User).filter(User.id == id).first()
#         if not user: 
#             logger.error("User with id %d not found. Skipping body_embedding removal.", id)
#             return {"status": "error", "message": f"user id {id} not found" }
#         user.body_embedding = None
#         user.last_embedding_update_ts = datetime.now(timezone.utc)  
#         session.commit()
#         session.refresh(user)  
#         logger.warning("Embeddings removed for user id=%d", id)
#         return user
#     except Exception as e:
#         session.rollback()    
#         logger.exception("remove embeddings: DB error during user lookup.")
#         return{"status":"error", "message":"DB error"}
#     finally:
#         session.close()    

# def get_status():
#     """Return current extract_service status."""
#     return {
#         "running": is_running,
#         "num_cams": len(RTSP_STREAMS),
#         "rtsp_streams": RTSP_STREAMS,
#         "id": current_person_id,
#     }

# File: mvision/services/extract_service.py
# File: mvision/services/extract_service.py
# File: mvision/services/extract_service.py
# File: mvision/services/extract_service.py
# File: mvision/services/extract_service.py
# File: mvision/services/extract_service.py
"""
Multi-RTSP extraction & embeddings service (Postgres + pgvector).

- YOLO: person detection.
- TorchReID: BODY 512-D float32 L2-normalized embeddings (RGB crop).
- InsightFace: FACE 512-D float32 L2-normalized embeddings.
- Robust face↔person link: center-in-box OR IoU OR inter/face_area.
- CSV: body_embedding, face_embedding.
- DB: User.body_embedding, User.face_embedding (+ last_embedding_update_ts).
"""
from __future__ import annotations

import csv
import sys
import time
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple

import logging

import cv2
import numpy as np
import torch

from sqlalchemy.sql import func  # kept if used elsewhere

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

# ------------------------- optional libs -------------------------
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# Use same TorchReID extractor type as your other pipeline
try:
    from torchreid.utils import FeatureExtractor as TorchreidExtractor
except Exception as e:
    print("TorchReID import error:", e)
    TorchreidExtractor = None

# InsightFace + ONNX Runtime
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
EMB_UPDATE_MIN_INTERVAL = 2.0  # seconds

# Face config
USE_FACE = True
FACE_MODEL = "buffalo_l"
FACE_DET_SIZE = (640, 640)         # (w, h)
FACE_PROVIDER = "auto"             # "auto" | "cuda" | "cpu"

# Link thresholds (choose one to succeed)
FACE_IOU_LINK = 0.05               # small face vs large person => IoU tiny; keep low
FACE_OVER_FACE_LINK = 0.60         # inter_area / face_area
# center-in-box is always accepted

# ------------------------- logging -------------------------
logger = logging.getLogger("mvision.extract_service")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# ------------------------- globals -------------------------
yolo_model: YOLO | None = None
reid_extractor: TorchreidExtractor | None = None
face_app: FaceAnalysis | None = None

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

_last_user_update_ts: Dict[int, float] = {}

# ------------------ DB helpers ------------------
def init_db():
    # Base.metadata.create_all(bind=engine)
    logger.info("DB initialized...")

def _to_float_list(x) -> Optional[List[float]]:
    if x is None:
        return None
    if isinstance(x, list) and all(isinstance(v, (float, int)) for v in x):
        return [float(v) for v in x]
    try:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        return [float(v) for v in arr.tolist()]
    except Exception:
        return None

def _dim_ok(vec: Optional[List[float]]) -> bool:
    return (vec is None) or (len(vec) == EXPECTED_EMBED_DIM)

def upsert_user_embeddings_by_id(user_id: int,
                                 body: Optional[List[float]] = None,
                                 face: Optional[List[float]] = None):
    """
    Update any provided embeddings for a user:
      - User.body_embedding (if `body` provided)
      - User.face_embedding (if `face` provided)
      - Always bump last_embedding_update_ts on any change.
    """
    body = _to_float_list(body)
    face = _to_float_list(face)

    if not _dim_ok(body) or not _dim_ok(face):
        logger.warning("Embedding dim mismatch for id=%s (expected %d)", user_id, EXPECTED_EMBED_DIM)
        return None

    if body is None and face is None:
        return None

    with db_lock:
        session = get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                logger.warning("User ID %s not found. Skipping update.", user_id)
                return None

            changed = False
            if body is not None:
                user.body_embedding = body
                changed = True
            if face is not None:
                user.face_embedding = face
                changed = True
            if changed:
                user.last_embedding_update_ts = datetime.now(timezone.utc)
                session.commit()
                session.refresh(user)
                logger.info("Updated embeddings for user id=%s (body=%s, face=%s)",
                            user_id, body is not None, face is not None)
            return user if changed else None
        except Exception:
            session.rollback()
            logger.exception("DB error while updating embeddings for id=%s", user_id)
            return None
        finally:
            session.close()

# ------------------ utils ------------------
def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n == 0 or not np.isfinite(n):
        return v
    return v / n

def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    # TorchReID expects RGB input
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
    # Intersection area divided by face box area (b)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    face_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    return inter / face_area if face_area > 0 else 0.0

def face_center_in(a: Tuple[int, int, int, int], b: Tuple[float, float, float, float]) -> bool:
    # Is face center inside person box?
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
    # Correct attribute names across insightface versions
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
    return emb

def _cuda_ep_loadable() -> bool:
    if ort is None:
        return False
    try:
        if sys.platform.startswith("darwin"):
            return False
        from pathlib import Path as _P
        import os as _os, ctypes as _ct
        capi_dir = _P(ort.__file__).parent / "capi"
        name = "onnxruntime_providers_cuda.dll" if _os.name == "nt" else "libonnxruntime_providers_cuda.so"
        lib_path = capi_dir / name
        if not lib_path.exists():
            return False
        _ct.CDLL(str(lib_path))
        return True
    except Exception:
        return False

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
            else:
                logger.info("Requested CUDA EP, but not loadable. Using CPU.")
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
        return app
    except Exception:
        logger.exception("InsightFace init failed")
        return None

# ------------------ models init ------------------
def init_models():
    """Load YOLO + TorchReID + InsightFace once on startup."""
    global yolo_model, reid_extractor, face_app

    gpu = torch.cuda.is_available() and ("cuda" in DEVICE.lower())
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

    # InsightFace
    face_app = init_face_engine(
        USE_FACE, DEVICE, FACE_MODEL, FACE_DET_SIZE[0], FACE_DET_SIZE[1], FACE_PROVIDER
    )

# ------------------ CSV + crops helpers ------------------
def ensure_csv_header():
    if not Path(EMB_CSV).exists():
        with csv_lock:
            with open(EMB_CSV, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "user_id",
                        "user_name",
                        "ts",
                        "cam_idx",
                        "frame_idx",
                        "det_idx",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                        "conf",
                        "body_embedding",
                        "face_embedding",
                        "crop_path",
                    ]
                )
        logger.info("Created CSV header: %s", EMB_CSV)

def ensure_crops_dir(cam_idx: int) -> Path:
    cam_dir = Path(CROPS_ROOT) / f"{current_person_name}" / f"cam{cam_idx}"
    cam_dir.mkdir(parents=True, exist_ok=True)
    return cam_dir

# ------------------ throttling ------------------
def _should_update_user(id: int) -> bool:
    now = time.time()
    last = _last_user_update_ts.get(id)
    if last is None or (now - last) >= EMB_UPDATE_MIN_INTERVAL:
        _last_user_update_ts[id] = now
        return True
    return False

# ------------------ face on frame ------------------
def detect_faces_and_embeddings(frame: np.ndarray) -> List[Dict]:
    """Return [{'bbox': (x1,y1,x2,y2), 'emb': np.ndarray}]"""
    outs: List[Dict] = []
    if face_app is None:
        return outs
    try:
        faces = face_app.get(frame)
        if faces:
            logger.debug("[Face] detected=%d", len(faces))
        for f in safe_iter_faces(faces):
            bbox = getattr(f, "bbox", None)
            if bbox is None:
                continue
            b = np.asarray(bbox).reshape(-1)
            if b.size < 4:
                continue
            x1, y1, x2, y2 = map(float, b[:4])
            emb = extract_face_embedding(f)
            if emb is None:
                continue
            emb = l2_normalize(np.asarray(emb, dtype=np.float32))  # float32 L2
            outs.append({"bbox": (x1, y1, x2, y2), "emb": emb})
    except Exception:
        logger.exception("FaceAnalysis error")
    return outs

def link_face_to_person(face_bbox: Tuple[float, float, float, float],
                        person_bbox: Tuple[int, int, int, int]) -> bool:
    """Return True if the face bbox belongs to the person bbox."""
    # Accept if center of face is inside person bbox
    if face_center_in(tuple(map(int, face_bbox)), person_bbox):
        return True
    # Else check IoU
    if iou_xyxy(person_bbox, face_bbox) >= FACE_IOU_LINK:
        return True
    # Else check that most of the face is inside the person box
    if inter_over_face(person_bbox, face_bbox) >= FACE_OVER_FACE_LINK:
        return True
    return False

# ------------------ worker loop ------------------
def embedding_loop_for_cam(cam_idx: int, rtsp_url: str):
    """
    - Detect persons via YOLO.
    - Extract body (TorchReID, RGB) and face (InsightFace) embeddings.
    - Link face→person (center-in-box OR IoU OR inter/face_area); write both to CSV.
    - DB: update both embeddings (when available) with throttling.
    """
    global current_person_id, current_person_name

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
            detections: List[tuple[float, float, float, float, float]] = []
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

                        keep = (cls == 0)  # person
                        xyxy, confs = xyxy[keep], confs[keep]

                        for (x1, y1, x2, y2), c in zip(xyxy, confs):
                            x1f = float(max(0, min(W - 1, x1)))
                            y1f = float(max(0, min(H - 1, y1)))
                            x2f = float(max(0, min(W - 1, x2)))
                            y2f = float(max(0, min(H - 1, y2)))
                            detections.append((x1f, y1f, x2f, y2f, float(c)))
                except Exception:
                    logger.exception("[Loop cam%d] YOLO error", cam_idx)

            # --- Faces on the frame (once per frame) ---
            faces = detect_faces_and_embeddings(frame) if USE_FACE else []

            # --- Embedding extraction + CSV + DB ---
            if detections and current_person_id:
                det_idx = 0
                for (x1, y1, x2, y2, conf) in detections:
                    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                    x1i, y1i = max(0, x1i), max(0, y1i)
                    x2i, y2i = min(W, x2i), min(H, y2i)
                    if x2i <= x1i or y2i <= y1i:
                        continue

                    cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                    label = f"{current_person_id} cam{cam_idx}"
                    cv2.putText(
                        vis, label, (x1i, max(0, y1i - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )

                    crop_bgr = frame[y1i:y2i, x1i:x2i]

                    # save crop
                    crop_filename = f"frame_{current_person_id}_{current_person_name}_{frame_idx:06d}_det{det_idx}.jpg"
                    crop_path = cam_dir / crop_filename
                    try:
                        cv2.imwrite(str(crop_path), crop_bgr)
                    except Exception:
                        logger.exception("[Loop cam%d] crop save error", cam_idx)
                        continue

                    # Body embedding (TorchReID, RGB -> L2 float32)
                    body_feat = None
                    if reid_extractor is not None:
                        try:
                            crop_rgb = _to_rgb(crop_bgr)
                            with reid_lock:
                                feats = reid_extractor([crop_rgb])
                            f0 = feats[0]
                            if hasattr(f0, "detach"):
                                f0 = f0.detach().cpu().numpy()
                            elif not isinstance(f0, np.ndarray):
                                f0 = np.asarray(f0)
                            body_feat = l2_normalize(f0.astype(np.float32, copy=False))
                        except Exception:
                            logger.exception("[Loop cam%d] body embedding error", cam_idx)

                    # Face embedding: robust link
                    face_feat = None
                    if faces:
                        # pick best candidate based on heuristics
                        best_idx = -1
                        for idx, fm in enumerate(faces):
                            fb = fm["bbox"]
                            if link_face_to_person(fb, (x1i, y1i, x2i, y2i)):
                                best_idx = idx
                                break
                        if best_idx < 0:
                            # fallback: choose max IoU
                            best_iou, best_idx = 0.0, -1
                            t_xyxy = (x1i, y1i, x2i, y2i)
                            for idx, fm in enumerate(faces):
                                fiou = iou_xyxy(t_xyxy, fm["bbox"])
                                if fiou > best_iou:
                                    best_iou, best_idx = fiou, idx
                            if best_idx >= 0 and best_iou < FACE_IOU_LINK:
                                best_idx = -1
                        if best_idx >= 0:
                            face_feat = faces[best_idx]["emb"]

                    # CSV strings
                    body_str = " ".join(f"{v:.6f}" for v in (body_feat.tolist() if body_feat is not None else []))
                    face_str = " ".join(f"{v:.6f}" for v in (face_feat.tolist() if face_feat is not None else []))
                    ts = time.time()

                    # Append to CSV
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
                                    body_str,
                                    face_str,
                                    str(crop_path),
                                ]
                            )

                    # DB: write both if available (throttled)
                    try:
                        if _should_update_user(current_person_id):
                            upsert_user_embeddings_by_id(
                                current_person_id,
                                body=(body_feat.tolist() if body_feat is not None else None),
                                face=(face_feat.tolist() if face_feat is not None else None),
                            )
                    except Exception:
                        logger.exception(
                            "[Loop cam%d] failed updating user embeddings for '%s'",
                            cam_idx, current_person_id,
                        )

                    det_idx += 1
            else:
                # visualization only
                for (x1, y1, x2, y2, _) in detections:
                    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                    x1i, y1i = max(0, x1i), max(0, y1i)
                    x2i, y2i = min(W, x2i), min(H, y2i)
                    if x2i <= x1i or y2i <= y1i:
                        continue
                    cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                    cv2.putText(
                        vis, f"cam{cam_idx}", (x1i, max(0, y1i - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )

            # show/update
            with frames_lock:
                latest_frames[cam_idx] = vis

            if frame_idx % 50 == 0:
                logger.info("[Loop cam%d] frame %d, detections=%d, faces=%d",
                            cam_idx, frame_idx, len(detections), len(faces))

    finally:
        cap.release()
        logger.info("[Loop cam%d] stopped", cam_idx)

# ------------------ viewer ------------------
def viewer_loop():
    logger.info("[Viewer] started")
    window_name = "Multi-RTSP Viewer"
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
            if key in (ord("q"), 27):
                logger.info("[Viewer] key pressed, stopping...")
                stop_event.set()
                break

            time.sleep(0.01)
    finally:
        cv2.destroyAllWindows()
        logger.info("[Viewer] stopped")

# ------------------ control API ------------------
def start_extraction(id: int):
    """
    Start extraction threads for all RTSP streams and viewer thread.
    """
    global extract_threads, viewer_thread, is_running
    global current_person_id, current_person_name

    if is_running:
        logger.warning("Extraction already running for '%s'", current_person_name)
        return {"status": "already_running", "id": current_person_id, "name": current_person_name}

    if not RTSP_STREAMS:
        logger.error("RTSP_STREAMS is empty.")
        return {"status": "error", "message": "RTSP_STREAMS empty"}

    session = get_session()
    try:
        user = session.query(User).filter(User.id == id).first()
        if not user:
            logger.error("User with id %d not found. Extraction aborted.", id)
            return {"status": "error", "message": f"user id {id} not found"}

        current_person_id = id
        current_person_name = user.name

        logger.info("Extraction will run for user: id=%d name='%s'", current_person_id, current_person_name)
    except Exception:
        session.rollback()
        logger.exception("DB error during user lookup.")
        return {"status": "error", "message": "DB error"}
    finally:
        session.close()

    stop_event.clear()
    extract_threads = {}
    with frames_lock:
        latest_frames.clear()

    # init models if not already
    if yolo_model is None or (USE_FACE and face_app is None) or (reid_extractor is None):
        init_models()

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
    current_person_id = None
    logger.info("Extraction stopped.")
    return {"status": "stopped"}

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
        session.refresh(user)
        logger.warning("Embeddings removed for user id=%d", id)
        return user
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
