# -----------------------------------Routes--------------------------------------------------------
# # app/routes/embeddings.py
# from fastapi import APIRouter
# from pydantic import BaseModel 
# from mvision.services import extract_service

# router = APIRouter(prefix="/api/extraction", tags=["extraction"])


# class StartRequest(BaseModel):
#     id: int


# @router.post("/start")
# def start(req: StartRequest):
#     """
#     Start background extraction for all RTSP streams.
#     Payload: { "id": 123 }
#     """
#     return extract_service.start_extraction(req.id)


# @router.post("/stop")
# def stop():
#     """Stop background extraction."""
#     return extract_service.stop_extraction()

# @router.delete("/remove/{id}")
# def remove(id: int):
#     """Remove embeddings."""
#     return extract_service.remove_embeddings(id)


# @router.get("/status")
# def status():
#     """Return extract_service status."""
#     return extract_service.get_status()


# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel, Field, conint
# from typing import Optional, List, Dict, Any

# from mvision.services import extract_service

# router = APIRouter(prefix="/api/extraction", tags=["extraction"])

# # ---------- Schemas ----------
# class StartRequest(BaseModel):
#     id: conint(gt=0) = Field(..., description="User ID to capture for")

# class StartResponse(BaseModel):
#     status: str
#     num_cams: int
#     id: Optional[int]
#     name: Optional[str]

# class FinalizeResult(BaseModel):
#     status: str
#     body: Optional[bool] = None
#     face: Optional[bool] = None

# class StopResponse(BaseModel):
#     status: str
#     id: Optional[int] = None
#     finalize: Optional[FinalizeResult] = None

# class ExtractionStatus(BaseModel):
#     running: bool
#     num_cams: int
#     rtsp_streams: List[str]
#     id: Optional[int]

# # ---------- Routes ----------
# @router.post("/start", response_model=StartResponse)
# def start(req: StartRequest) -> StartResponse:
#     """
#     Start background capture & crop-only pipeline.
#     Embeddings are computed and written to DB on /stop.
#     """
#     try:
#         res = extract_service.start_extraction(req.id)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"start_extraction failed: {e}") from e

#     if not isinstance(res, dict):
#         raise HTTPException(status_code=500, detail="start_extraction returned invalid response")

#     if res.get("status") == "error":
#         msg = res.get("message", "Unknown error")
#         raise HTTPException(status_code=400, detail=msg)

#     return StartResponse(
#         status=res.get("status", "unknown"),
#         num_cams=int(res.get("num_cams", 0)),
#         id=res.get("id"),
#         name=res.get("name"),
#     )

# @router.post("/stop", response_model=StopResponse)
# def stop() -> StopResponse:
#     """Stop background capture and finalize embeddings from saved galleries."""
#     try:
#         res = extract_service.stop_extraction()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"stop_extraction failed: {e}") from e

#     if not isinstance(res, dict):
#         raise HTTPException(status_code=500, detail="stop_extraction returned invalid response")

#     if res.get("status") == "not_running":
#         raise HTTPException(status_code=409, detail="extraction is not running")

#     fin = res.get("finalize") or {}
#     finalize = FinalizeResult(
#         status=str(fin.get("status", "unknown")),
#         body=fin.get("body"),
#         face=fin.get("face"),
#     ) if fin else None

#     return StopResponse(
#         status=str(res.get("status", "unknown")),
#         id=res.get("id"),
#         finalize=finalize,
#     )

# @router.delete("/remove/{id}", response_model=Dict[str, Any])
# def remove(id: int):
#     """Remove stored embeddings (body+face) for a user."""
#     try:
#         res = extract_service.remove_embeddings(id)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"remove_embeddings failed: {e}") from e

#     if isinstance(res, dict) and res.get("status") == "error":
#         raise HTTPException(status_code=400, detail=res.get("message", "unknown error"))

#     return res if isinstance(res, dict) else {"status": "ok", "id": id}

# @router.get("/status", response_model=ExtractionStatus)
# def status() -> ExtractionStatus:
#     """Return current extract_service status."""
#     try:
#         res = extract_service.get_status()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"get_status failed: {e}") from e

#     if not isinstance(res, dict):
#         raise HTTPException(status_code=500, detail="get_status returned invalid response")

#     return ExtractionStatus(
#         running=bool(res.get("running", False)),
#         num_cams=int(res.get("num_cams", 0)),
#         rtsp_streams=list(res.get("rtsp_streams", [])),
#         id=res.get("id"),
#     )
# # file: app/routes/embeddings.py
# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel, Field, conint
# from typing import Optional, List, Dict, Any

# from mvision.services import extract_service

# router = APIRouter(prefix="/api/extraction", tags=["extraction"])

# # ---------- Schemas ----------
# class StartRequest(BaseModel):
#     id: conint(gt=0) = Field(..., description="User ID to capture for")
#     name: str = Field("unknown", description="User name for folder/DB labeling")

# class StartResponse(BaseModel):
#     status: str
#     num_cams: int
#     id: Optional[int]
#     name: Optional[str]

# class ExtractStartResponse(BaseModel):
#     status: str
#     id: int
#     name: str

# class ProgressResponse(BaseModel):
#     id: int
#     stage: str
#     percent: int
#     message: str
#     total_body: int
#     total_face: int
#     done_body: int
#     done_face: int

# class StopResponse(BaseModel):
#     status: str
#     id: Optional[int] = None

# class ExtractionStatus(BaseModel):
#     running: bool
#     num_cams: int
#     rtsp_streams: List[str]
#     id: Optional[int]

# # ---------- Routes ----------
# @router.post("/start", response_model=StartResponse)
# def start(req: StartRequest) -> StartResponse:
#     """
#     Start background capture & embedding pipeline (YOLO + OSNet).
#     """
#     if not hasattr(extract_service, "start_extraction"):
#         raise HTTPException(status_code=501, detail="start_extraction is not implemented in extract_service")

#     try:
#         # FIX: pass both id and name expected by the service
#         res = extract_service.start_extraction(req.id, req.name)  # type: ignore
#     except TypeError as e:
#         # Common cause: wrong signature (missing name)
#         raise HTTPException(status_code=500, detail=f"start_extraction signature mismatch: {e}") from e
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"start_extraction failed: {e}") from e

#     if not isinstance(res, dict):
#         raise HTTPException(status_code=500, detail="start_extraction returned invalid response")

#     if res.get("status") == "error":
#         raise HTTPException(status_code=400, detail=res.get("message", "Unknown error"))

#     # Map streams->num_cams (our service returns "streams")
#     num_cams = int(
#         res.get("num_cams", res.get("streams", len(getattr(extract_service, "RTSP_STREAMS", []))))
#     )

#     return StartResponse(
#         status=str(res.get("status", "unknown")),
#         num_cams=num_cams,
#         id=req.id,
#         name=req.name,
#     )

# @router.post("/stop", response_model=StopResponse)
# def stop() -> StopResponse:
#     """Stop background capture/embedding."""
#     if not hasattr(extract_service, "stop_extraction"):
#         raise HTTPException(status_code=501, detail="stop_extraction is not implemented in extract_service")

#     try:
#         res = extract_service.stop_extraction()  # type: ignore
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"stop_extraction failed: {e}") from e

#     if not isinstance(res, dict):
#         raise HTTPException(status_code=500, detail="stop_extraction returned invalid response")

#     if res.get("status") in {"not_running", "error"}:
#         raise HTTPException(status_code=409, detail=res.get("message", "extraction is not running"))

#     return StopResponse(
#         status=str(res.get("status", "unknown")),
#         id=res.get("id"),
#     )

# @router.post("/extract", response_model=ExtractStartResponse)
# def extract_embeddings(req: StartRequest) -> ExtractStartResponse:
#     """
#     Compute embeddings from saved folders only; write to DB.
#     If not supported by service, returns 501.
#     """
#     if not hasattr(extract_service, "extract_embeddings_async"):
#         raise HTTPException(status_code=501, detail="extract_embeddings_async is not implemented in extract_service")

#     try:
#         res = extract_service.extract_embeddings_async(req.id)  # type: ignore
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"extract_embeddings failed: {e}") from e

#     if not isinstance(res, dict):
#         raise HTTPException(status_code=500, detail="extract_embeddings returned invalid response")

#     if res.get("status") == "error":
#         raise HTTPException(status_code=400, detail=res.get("message", "Unknown error"))

#     return ExtractStartResponse(
#         status=str(res.get("status", "unknown")),
#         id=int(res.get("id", req.id)),
#         name=str(res.get("name", req.name)),
#     )

# @router.get("/progress/{id}", response_model=ProgressResponse)
# def progress(id: int) -> ProgressResponse:
#     """
#     Get current extraction progress. Falls back to service status if
#     granular progress is not available.
#     """
#     # Prefer fine-grained progress if available
#     if hasattr(extract_service, "get_progress_for_user"):
#         try:
#             st = extract_service.get_progress_for_user(id)  # type: ignore
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"get_progress failed: {e}") from e
#         st = st or {}
#     else:
#         # Fallback: derive minimal info from get_status()
#         if not hasattr(extract_service, "get_status"):
#             raise HTTPException(status_code=501, detail="get_status is not implemented in extract_service")
#         try:
#             status = extract_service.get_status()  # type: ignore
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"get_status failed: {e}") from e
#         st = (status or {}).get("progress", {}) or {}

#     return ProgressResponse(
#         id=id,
#         stage=str(st.get("stage", "unknown")),
#         percent=int(st.get("percent", 0)),
#         message=str(st.get("message", "")),
#         total_body=int(st.get("total_body", 0)),
#         total_face=int(st.get("total_face", 0)),
#         done_body=int(st.get("done_body", 0)),
#         done_face=int(st.get("done_face", 0)),
#     )

# @router.delete("/remove/{id}", response_model=Dict[str, Any])
# def remove(id: int):
#     """Remove stored embeddings (body+face) for a user. Returns 501 if not supported."""
#     if not hasattr(extract_service, "remove_embeddings"):
#         raise HTTPException(status_code=501, detail="remove_embeddings is not implemented in extract_service")

#     try:
#         res = extract_service.remove_embeddings(id)  # type: ignore
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"remove_embeddings failed: {e}") from e

#     if isinstance(res, dict) and res.get("status") == "error":
#         raise HTTPException(status_code=400, detail=res.get("message", "unknown error"))

#     return res if isinstance(res, dict) else {"status": "ok", "id": id}

# @router.get("/status", response_model=ExtractionStatus)
# def status() -> ExtractionStatus:
#     """Return current extract_service status."""
#     if not hasattr(extract_service, "get_status"):
#         raise HTTPException(status_code=501, detail="get_status is not implemented in extract_service")

#     try:
#         res = extract_service.get_status()  # type: ignore
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"get_status failed: {e}") from e

#     if not isinstance(res, dict):
#         raise HTTPException(status_code=500, detail="get_status returned invalid response")

#     num_cams = int(res.get("num_cams", res.get("streams", len(getattr(extract_service, "RTSP_STREAMS", [])))))

#     return ExtractionStatus(
#         running=bool(res.get("running", False)),
#         num_cams=num_cams,
#         rtsp_streams=list(res.get("rtsp_streams", [])),
#         id=res.get("id"),
#     )






------------------------------------Service------------------------------------------------------
# ----------------------------------------------------------------------------------------
# # file: mvision/services/extract_service.py
# from __future__ import annotations

# import csv
# import sys
# import os
# import time
# import threading
# from pathlib import Path
# from datetime import datetime, timezone
# from typing import Optional, List, Dict, Tuple, Callable
# from queue import Queue, Empty
# import logging
# import shutil
# import random

# import cv2
# import numpy as np
# import torch

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

# # ------------------------- runtime perf knobs -------------------------
# try:
#     torch.backends.cudnn.benchmark = True
# except Exception:
#     pass
# try:
#     if hasattr(torch, "set_float32_matmul_precision"):
#         torch.set_float32_matmul_precision("high")
# except Exception:
#     pass
# try:
#     cv2.setNumThreads(1)
# except Exception:
#     pass

# # ------------------------- optional libs -------------------------
# try:
#     from ultralytics import YOLO
# except Exception:
#     YOLO = None

# try:
#     from torchreid.utils import FeatureExtractor as TorchreidExtractor
# except Exception as e:
#     print("TorchReID import error:", e)
#     TorchreidExtractor = None

# try:
#     from insightface.app import FaceAnalysis
#     INSIGHT_OK = True
# except Exception:
#     FaceAnalysis = None
#     INSIGHT_OK = False

# try:
#     import onnxruntime as ort
# except Exception:
#     ort = None

# # ------------------------- config -------------------------
# EXPECTED_EMBED_DIM = 512

# # Face capture/link gates (capture-time only; aggregation stays unbiased)
# USE_FACE = True
# FACE_MODEL = "buffalo_l"
# FACE_DET_SIZE = (640, 640)         # (w, h)
# FACE_PROVIDER = "auto"             # "auto" | "cuda" | "cpu"
# FACE_MIN_SCORE = 0.5
# FACE_MIN_SIZE = 32                 # px min(side) of detected face
# FACE_IOU_LINK = 0.05
# FACE_OVER_FACE_LINK = 0.60
# FACE_EVERY = 2

# # YOLO inference size
# YOLO_IMGSZ = 640

# # Queues
# CAP_QUEUE_MAX = 2
# IO_QUEUE_MAX  = 1024

# # Viewer FPS throttle
# VIEWER_SLEEP_MS = 5

# # ------------------------- logging -------------------------
# logger = logging.getLogger("mvision.extract_service")
# if not logger.handlers:
#     ch = logging.StreamHandler()
#     ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
#     logger.addHandler(ch)
# logger.setLevel(logging.INFO)

# # ------------------------- globals -------------------------
# yolo_model: YOLO | None = None
# reid_extractor: TorchreidExtractor | None = None
# face_app: FaceAnalysis | None = None

# yolo_lock = threading.Lock()
# reid_lock = threading.Lock()
# csv_lock = threading.Lock()
# frames_lock = threading.Lock()

# latest_frames: Dict[int, np.ndarray] = {}

# capture_threads: Dict[int, threading.Thread] = {}
# extract_threads: Dict[int, threading.Thread] = {}
# cap_queues: Dict[int, Queue] = {}

# io_thread: Optional[threading.Thread] = None
# io_queue: Queue | None = None

# viewer_thread: Optional[threading.Thread] = None
# stop_event = threading.Event()
# is_running = False
# current_person_id: Optional[int] = None
# current_person_name: Optional[str] = None

# # ---- extraction progress state (for /extract) ----
# progress_lock = threading.Lock()
# progress_state: Dict[int, Dict[str, object]] = {}

# # ------------------ progress helpers ------------------
# def _progress_reset(user_id: int, user_name: str) -> None:
#     with progress_lock:
#         progress_state[user_id] = {
#             "user_name": user_name,
#             "stage": "idle",
#             "total_body": 0,
#             "total_face": 0,
#             "done_body": 0,
#             "done_face": 0,
#             "percent": 0,
#             "message": "waiting",
#         }

# def _progress_set(user_id: int, **kv) -> None:
#     with progress_lock:
#         st = progress_state.get(user_id)
#         if not st:
#             st = {}
#             progress_state[user_id] = st
#         st.update(kv)

#         # Clamp counters
#         tb, tf = int(st.get("total_body", 0)), int(st.get("total_face", 0))
#         db, df = int(st.get("done_body", 0)), int(st.get("done_face", 0))
#         db = max(0, min(db, tb))
#         df = max(0, min(df, tf))
#         st["done_body"], st["done_face"] = db, df

#         # Stage-aware percentage:
#         stage = str(st.get("stage", "")).lower()
#         if stage == "embedding_body" and tb > 0:
#             pct = int(round((db / max(1, tb)) * 100))
#         elif stage == "embedding_face" and tf > 0:
#             pct = int(round((df / max(1, tf)) * 100))
#         else:
#             tot = max(1, tb + tf)
#             pct = int(round(((db + df) / tot) * 100))

#         # Keep within [0, 100]
#         st["percent"] = max(0, min(100, pct))

# def get_progress_for_user(user_id: int) -> Dict[str, object]:
#     with progress_lock:
#         st = progress_state.get(user_id)
#         return dict(st) if st else {
#             "stage": "unknown",
#             "percent": 0,
#             "message": "no job",
#             "total_body": 0,
#             "total_face": 0,
#             "done_body": 0,
#             "done_face": 0,
#         }

# # ------------------ init_db (startup hook) ------------------
# def init_db() -> None:
#     try:
#         Base.metadata.create_all(bind=engine)
#         logger.info("DB initialized.")
#     except Exception:
#         logger.exception("DB init failed")

# # ------------------ utils ------------------
# def l2_normalize(v: np.ndarray) -> np.ndarray:
#     v = np.asarray(v, dtype=np.float32).reshape(-1)
#     n = float(np.linalg.norm(v))
#     if n == 0.0 or not np.isfinite(n):
#         return v
#     return v / n

# def _as_512f(vec: np.ndarray | list | None) -> Optional[np.ndarray]:
#     if vec is None:
#         return None
#     try:
#         a = np.asarray(vec, dtype=np.float32).reshape(-1)
#         if a.size != EXPECTED_EMBED_DIM:
#             return None
#         if not np.isfinite(a).all():
#             return None
#         return l2_normalize(a)
#     except Exception:
#         return None

# def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
#     return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
#     ax1, ay1, ax2, ay2 = a
#     bx1, by1, bx2, by2 = b
#     inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
#     inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
#     iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
#     inter = iw * ih
#     a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
#     b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
#     denom = a_area + b_area - inter
#     return inter / denom if denom > 0 else 0.0

# def inter_over_face(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
#     ax1, ay1, ax2, ay2 = a
#     bx1, by1, bx2, by2 = b
#     inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
#     inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
#     iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
#     inter = iw * ih
#     face_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
#     return inter / face_area if face_area > 0 else 0.0

# def face_center_in(a: Tuple[int, int, int, int], b: Tuple[float, float, float, float]) -> bool:
#     fx1, fy1, fx2, fy2 = a
#     px1, py1, px2, py2 = b
#     cx = (fx1 + fx2) * 0.5
#     cy = (fy1 + fy2) * 0.5
#     return (px1 <= cx <= px2) and (py1 <= cy <= py2)

# def safe_iter_faces(obj):
#     if obj is None:
#         return []
#     try:
#         return list(obj)
#     except TypeError:
#         return [obj]

# def extract_face_embedding(face):
#     emb = getattr(face, "normed_embedding", None)
#     if emb is None:
#         emb = getattr(face, "embedding", None)
#     return emb

# def _cuda_ep_loadable() -> bool:
#     if ort is None:
#         return False
#     try:
#         if sys.platform.startswith("mac"):
#             return False
#         from pathlib import Path as _P
#         import ctypes as _ct
#         capi_dir = _P(ort.__file__).parent / "capi"
#         name = "onnxruntime_providers_cuda.dll" if os.name == "nt" else "libonnxruntime_providers_cuda.so"
#         lib_path = capi_dir / name
#         if not lib_path.exists():
#             return False
#         _ct.CDLL(str(lib_path))
#         return True
#     except Exception:
#         return False

# # ------------------ models init ------------------
# def init_face_engine(use_face: bool, device: str, face_model: str, det_w: int, det_h: int, face_provider: str):
#     if not use_face:
#         return None
#     if not INSIGHT_OK:
#         logger.warning("insightface not installed; face recognition disabled.")
#         return None
#     try:
#         is_cuda = ("cuda" in device.lower()) and torch.cuda.is_available()
#         cuda_ok = _cuda_ep_loadable()

#         providers = ["CPUExecutionProvider"]
#         if face_provider == "cuda":
#             if cuda_ok:
#                 providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
#         elif face_provider == "auto":
#             if is_cuda and cuda_ok:
#                 providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

#         app = FaceAnalysis(name=face_model, providers=providers)
#         ctx_id = 0 if providers[0].startswith("CUDA") else -1
#         try:
#             app.prepare(ctx_id=ctx_id, det_size=(det_w, det_h))
#         except TypeError:
#             app.prepare(ctx_id=ctx_id)
#         logger.info("InsightFace ready (model=%s, providers=%s).", face_model, providers)

#         # Warm-up (avoids first-call latency)
#         try:
#             dummy = np.zeros((max(8, det_h), max(8, det_w), 3), dtype=np.uint8)
#             app.get(dummy)
#         except Exception:
#             pass

#         return app
#     except Exception:
#         logger.exception("InsightFace init failed")
#         return None

# def init_models():
#     """Load YOLO + TorchReID + InsightFace once on startup."""
#     global yolo_model, reid_extractor, face_app

#     gpu = torch.cuda.is_available() and ("cuda" in str(DEVICE).lower())
#     logger.info("init_models device=%s gpu_available=%s", DEVICE, torch.cuda.is_available())

#     # YOLO
#     if YOLO is not None:
#         try:
#             weights = YOLO_WEIGHTS
#             if not Path(weights).exists():
#                 logger.warning("%s not found, falling back to yolov8n.pt", weights)
#             # NOTE: we don't alter the weights path logic
#             yolo = YOLO(weights if Path(YOLO_WEIGHTS).exists() else "yolov8n.pt")
#             if gpu:
#                 try:
#                     yolo.to(DEVICE)
#                 except Exception:
#                     pass
#             # Warm-up
#             try:
#                 with torch.inference_mode():
#                     _dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
#                     yolo.predict(_dummy, device=DEVICE if gpu else "cpu",
#                                  conf=0.25, iou=0.5, imgsz=YOLO_IMGSZ, verbose=False)
#             except Exception:
#                 pass
#             yolo_model = yolo
#             logger.info("YOLO ready (gpu=%s)", gpu)
#         except Exception:
#             logger.exception("YOLO load failed; detection disabled.")
#             yolo_model = None
#     else:
#         logger.warning("ultralytics not installed; detection disabled.")

#     # TorchReID (used for embeddings)
#     if TorchreidExtractor is not None:
#         try:
#             dev = DEVICE if gpu else "cpu"
#             reid_extractor = TorchreidExtractor(model_name="osnet_x0_25", device=dev)
#             # Warm-up
#             try:
#                 dummy = np.zeros((256, 128, 3), dtype=np.uint8)
#                 _ = reid_extractor([dummy])
#             except Exception:
#                 pass
#             logger.info("TorchReID ready (device=%s)", dev)
#         except Exception:
#             logger.exception("TorchReID init failed; body embeddings disabled.")
#             reid_extractor = None
#     else:
#         logger.warning("torchreid not installed; body embeddings disabled.")

#     # InsightFace
#     globals()["face_app"] = init_face_engine(
#         USE_FACE, DEVICE, FACE_MODEL, FACE_DET_SIZE[0], FACE_DET_SIZE[1], FACE_PROVIDER
#     )

# # ------------------ CSV + gallery helpers ------------------
# def _ensure_dir(p: Path) -> None:
#     p.mkdir(parents=True, exist_ok=True)

# def ensure_csv_header():
#     _ensure_dir(Path(EMB_CSV).parent if Path(EMB_CSV).parent.as_posix() not in (".", "") else Path("."))
#     if not Path(EMB_CSV).exists():
#         with csv_lock:
#             with open(EMB_CSV, "w", newline="", encoding="utf-8") as f:
#                 w = csv.writer(f)
#                 w.writerow(
#                     [
#                         "user_id","user_name","ts","cam_idx","frame_idx","det_idx",
#                         "x1","y1","x2","y2","conf_or_score",
#                         "body_embedding","face_embedding","crop_path","kind",
#                     ]
#                 )
#         logger.info("Created CSV header: %s", EMB_CSV)

# def _gallery_body_dir(user_name: str) -> Path:
#     d = Path(CROPS_ROOT) / "gallery" / user_name
#     _ensure_dir(d)
#     return d

# def _gallery_face_dir(user_name: str) -> Path:
#     d = Path(CROPS_ROOT) / "gallery_face" / user_name
#     _ensure_dir(d)
#     return d

# def clear_user_galleries(user_name: str):
#     for base in ("gallery", "gallery_face"):
#         root = Path(CROPS_ROOT) / base / user_name
#         if root.exists():
#             shutil.rmtree(root, ignore_errors=True)
#         _ensure_dir(root)
#     logger.info("Cleared galleries for '%s'", user_name)

# def _epoch_ms() -> int:
#     # microsecond-ish uniqueness with small random tail
#     return int(time.time() * 1000) * 1000 + random.randint(0, 999)

# # ------------------ face helpers (capture-time) ------------------
# def detect_faces_raw(frame: np.ndarray):
#     outs: List[Dict] = []
#     if face_app is None:
#         return outs
#     try:
#         faces = face_app.get(np.ascontiguousarray(frame))
#         for f in safe_iter_faces(faces):
#             bbox = getattr(f, "bbox", None)
#             if bbox is None:
#                 continue
#             b = np.asarray(bbox).reshape(-1)
#             if b.size < 4:
#                 continue
#             x1, y1, x2, y2 = map(float, b[:4])
#             score = float(getattr(f, "det_score", 0.0))
#             outs.append({"bbox": (x1, y1, x2, y2), "score": score})
#     except Exception:
#         logger.exception("FaceAnalysis error")
#     return outs

# def link_face_to_person(face_bbox: Tuple[float, float, float, float],
#                         person_bbox: Tuple[int, int, int, int]) -> bool:
#     if face_center_in(tuple(map(int, face_bbox)), person_bbox):
#         return True
#     if iou_xyxy(person_bbox, face_bbox) >= FACE_IOU_LINK:
#         return True
#     if inter_over_face(person_bbox, face_bbox) >= FACE_OVER_FACE_LINK:
#         return True
#     return False

# # ------------------ RTSP helpers ------------------
# def _configure_rtsp_capture(cap: cv2.VideoCapture) -> None:
#     try:
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#     except Exception:
#         pass

# def _read_latest_frame(cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
#     grabbed = cap.grab()
#     if not grabbed:
#         ok, fr = cap.read()
#         return ok, (fr if ok else None)
#     for _ in range(3):
#         if not cap.grab():
#             break
#     ok, frame = cap.retrieve()
#     return ok, (frame if ok else None)

# # ------------------ async workers (IO) ------------------
# def start_workers_if_needed():
#     global io_thread, io_queue
#     if io_queue is None:
#         io_queue = Queue(maxsize=IO_QUEUE_MAX)
#     if io_thread is None or not io_thread.is_alive():
#         io_thread = threading.Thread(target=io_worker, name="io_worker", daemon=True)
#         io_thread.start()

# def io_worker():
#     while not stop_event.is_set():
#         try:
#             job = io_queue.get(timeout=0.1)  # type: ignore
#         except Empty:
#             continue
#         except Exception:
#             break
#         try:
#             # WHY: file and CSV write off main thread to avoid capture stalls
#             crop_path: Path = job["crop_path"]
#             _ensure_dir(crop_path.parent)
#             cv2.imwrite(str(crop_path), job["image"])
#             with csv_lock:
#                 with open(EMB_CSV, "a", newline="", encoding="utf-8") as f:
#                     csv.writer(f).writerow(job["csv_row"])
#         except Exception:
#             logger.exception("IO worker error")

# # ------------------ capture & processing loops ------------------
# def capture_thread_fn(cam_idx: int, rtsp_url: str):
#     q = cap_queues[cam_idx]
#     cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
#     if not cap.isOpened():
#         logger.error("[cap%d] cannot open RTSP: %s", cam_idx, rtsp_url)
#         return
#     _configure_rtsp_capture(cap)
#     logger.info("[cap%d] started on %s", cam_idx, rtsp_url)
#     try:
#         while not stop_event.is_set():
#             ok, frame = _read_latest_frame(cap)
#             if not ok or frame is None:
#                 time.sleep(0.01)
#                 continue
#             while not q.empty():
#                 try:
#                     q.get_nowait()
#                 except Exception:
#                     break
#             try:
#                 q.put_nowait(frame)
#             except Exception:
#                 pass
#     finally:
#         cap.release()
#         logger.info("[cap%d] stopped", cam_idx)

# def embedding_loop_for_cam(cam_idx: int, rtsp_url: str):
#     """
#     Save crops only:
#     - BODY: each person detection → gallery/<user>/person_<ms>.jpg
#     - FACE: when a clear face is linked → gallery_face/<user>/person_<ms>.jpg (same full-body crop)
#     """
#     global current_person_id, current_person_name

#     # Always ensure CSV + IO worker are ready before first enqueue.
#     ensure_csv_header()
#     start_workers_if_needed()

#     user_folder = current_person_name or "unknown"
#     body_dir = _gallery_body_dir(user_folder)
#     face_dir = _gallery_face_dir(user_folder)

#     frame_idx = 0
#     logger.info("[Loop cam%d] started on %s", cam_idx, rtsp_url)
#     q = cap_queues[cam_idx]

#     try:
#         while not stop_event.is_set():
#             try:
#                 frame = q.get(timeout=0.2)
#             except Empty:
#                 continue
#             except Exception:
#                 break
#             if frame is None:
#                 continue

#             frame_idx += 1
#             H, W = frame.shape[:2]
#             detections: List[tuple[float, float, float, float, float]] = []

#             # Avoid aliasing: draw on a copy only for viewer.
#             vis = frame.copy()  # WHY: downstream consumers might reuse the source frame

#             # YOLO detection
#             if yolo_model is not None:
#                 try:
#                     with yolo_lock, torch.inference_mode():
#                         res = yolo_model.predict(
#                             frame,
#                             conf=float(CONF_THRES),
#                             iou=float(IOU_THRES),
#                             imgsz=YOLO_IMGSZ,
#                             verbose=False,
#                             device=DEVICE if torch.cuda.is_available() and ("cuda" in str(DEVICE).lower()) else "cpu",
#                         )
#                     boxes = res[0].boxes if (res and len(res)) else None
#                     if boxes is not None:
#                         xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
#                         confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
#                         cls = boxes.cls.detach().cpu().numpy().astype(np.int32)
#                         keep = (cls == 0)  # person
#                         xyxy, confs = xyxy[keep], confs[keep]
#                         for (x1, y1, x2, y2), c in zip(xyxy, confs):
#                             x1f = float(max(0, min(W - 1, x1)))
#                             y1f = float(max(0, min(H - 1, y1)))
#                             x2f = float(max(0, min(W - 1, x2)))
#                             y2f = float(max(0, min(H - 1, y2)))
#                             if (x2f - x1f) < 4 or (y2f - y1f) < 4:
#                                 continue
#                             detections.append((x1f, y1f, x2f, y2f, float(c)))
#                 except Exception:
#                     logger.exception("[Loop cam%d] YOLO error", cam_idx)

#             # Faces on frame (stride)
#             faces = []
#             if USE_FACE and (frame_idx % max(1, FACE_EVERY) == 0):
#                 faces = detect_faces_raw(frame)

#             det_boxes_i: List[tuple[int, int, int, int, float]] = []
#             for (x1, y1, x2, y2, conf) in detections:
#                 x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
#                 x1i, y1i = max(0, x1i), max(0, y1i)
#                 x2i, y2i = min(W, x2i), min(H, y2i)
#                 if x2i <= x1i or y2i <= y1i:
#                     continue
#                 det_boxes_i.append((x1i, y1i, x2i, y2i, conf))

#                 # Draw for viewer only
#                 try:
#                     cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
#                     cv2.putText(
#                         vis, f"{current_person_id or ''} cam{cam_idx}", (x1i, max(0, y1i - 5)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
#                     )
#                 except Exception:
#                     pass

#                 # Save BODY crop (always)
#                 crop_bgr = frame[y1i:y2i, x1i:x2i]
#                 if crop_bgr.size > 0:
#                     ts = time.time()
#                     fname = f"person_{_epoch_ms()}.jpg"
#                     path = body_dir / fname
#                     row = [
#                         current_person_id, current_person_name, ts,
#                         cam_idx, frame_idx, len(det_boxes_i)-1,
#                         x1i, y1i, x2i, y2i,
#                         conf,
#                         "", "", str(path), "body",
#                     ]
#                     try:
#                         if io_queue is not None:
#                             io_queue.put_nowait({"crop_path": path, "image": crop_bgr, "csv_row": row})
#                     except Exception:
#                         # Drop silently under backpressure; loop keeps going.
#                         pass

#             # Save FACE (clear) crops → same full-body crop into FACE gallery
#             if det_boxes_i and faces:
#                 ts = time.time()
#                 for det_idx, (x1i, y1i, x2i, y2i, _conf) in enumerate(det_boxes_i):
#                     t_xyxy = (x1i, y1i, x2i, y2i)
#                     best_idx, best_score = -1, -1.0
#                     best_box = None
#                     for idx, fm in enumerate(faces):
#                         if link_face_to_person(fm["bbox"], t_xyxy):
#                             s = float(fm.get("score", 0.0))
#                             if s > best_score:
#                                 best_score, best_idx = s, idx
#                                 best_box = fm["bbox"]
#                     if best_idx < 0:
#                         continue
#                     fx1, fy1, fx2, fy2 = map(int, best_box)
#                     if best_score < FACE_MIN_SCORE:
#                         continue
#                     if min(fx2 - fx1, fy2 - fy1) < FACE_MIN_SIZE:
#                         continue

#                     body_crop = frame[y1i:y2i, x1i:x2i]
#                     if body_crop.size <= 0:
#                         continue
#                     fname = f"person_{_epoch_ms()}.jpg"
#                     path = _gallery_face_dir(current_person_name or "unknown") / fname
#                     row = [
#                         current_person_id, current_person_name, ts,
#                         cam_idx, frame_idx, det_idx,
#                         x1i, y1i, x2i, y2i,
#                         best_score,
#                         "", "", str(path), "face",
#                     ]
#                     try:
#                         if io_queue is not None:
#                             io_queue.put_nowait({"crop_path": path, "image": body_crop, "csv_row": row})
#                     except Exception:
#                         pass

#             with frames_lock:
#                 latest_frames[cam_idx] = vis

#             if frame_idx % 50 == 0:
#                 logger.info("[Loop cam%d] frame %d, persons=%d, faces=%d",
#                             cam_idx, frame_idx, len(det_boxes_i), len(faces) if faces else 0)

#     finally:
#         logger.info("[Loop cam%d] stopped", cam_idx)

# # ------------------ viewer ------------------
# def viewer_loop():
#     logger.info("[Viewer] started")
#     window_name = "Multi-RTSP Viewer"
#     window_created = False
#     try:
#         while not stop_event.is_set():
#             with frames_lock:
#                 frames = [
#                     latest_frames[idx]
#                     for idx in sorted(latest_frames.keys())
#                     if latest_frames.get(idx) is not None
#                 ]

#             if not frames:
#                 time.sleep(0.01)
#                 continue

#             if not window_created:
#                 try:
#                     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#                     window_created = True
#                 except Exception:
#                     pass

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

#             try:
#                 cv2.imshow(window_name, combined)
#             except Exception:
#                 window_created = False
#                 time.sleep(0.01)
#                 continue

#             key = cv2.waitKey(1) & 0xFF
#             if key in (ord("q"), 27):
#                 logger.info("[Viewer] key pressed, stopping...")
#                 stop_event.set()
#                 break

#             time.sleep(VIEWER_SLEEP_MS / 1000.0)
#     finally:
#         try:
#             cv2.destroyAllWindows()
#         except Exception:
#             pass
#         logger.info("[Viewer] stopped")

# # ------------------ finalize / extract from folders ------------------
# def _list_images(root: Path) -> List[Path]:
#     if not root.exists():
#         return []
#     pats = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
#     files: List[Path] = []
#     for p in pats:
#         files.extend(root.glob(p))
#     return sorted(files)

# def _load_bgr(path: Path) -> Optional[np.ndarray]:
#     try:
#         img = cv2.imread(str(path), cv2.IMREAD_COLOR)
#         return img if img is not None else None
#     except Exception:
#         return None

# def _mean_embed(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
#     if not vectors:
#         return None
#     arr = np.stack(vectors, axis=0).astype(np.float32)
#     m = np.mean(arr, axis=0)
#     return l2_normalize(m)

# def _count_images(user_name: str) -> Tuple[int, int]:
#     body = len(_list_images(_gallery_body_dir(user_name)))
#     face = len(_list_images(_gallery_face_dir(user_name)))
#     return body, face

# # ======= Code-2–style helpers (new; local-only, does not change external behavior) =======
# def _ahash8(img_bgr: np.ndarray) -> int:
#     """8x8 average hash; helps drop near-duplicate crops."""
#     try:
#         g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#         g = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA)
#         avg = float(g.mean())
#         bits = (g > avg).astype(np.uint8)
#         out = 0
#         for i, b in enumerate(bits.flatten()):
#             out |= (int(b) & 1) << i
#         return out
#     except Exception:
#         return random.getrandbits(64)

# def _is_blurry(img_bgr: np.ndarray, var_thr: float = 30.0) -> bool:
#     """Variance of Laplacian blur detection; very small crops will be handled by size gates."""
#     try:
#         g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#         return float(cv2.Laplacian(g, cv2.CV_64F).var()) < var_thr
#     except Exception:
#         return False

# def _prep_reid(img_bgr: np.ndarray) -> np.ndarray:
#     """Resize to 256x128 and convert to RGB for TorchReID."""
#     try:
#         out = cv2.resize(img_bgr, (128, 256), interpolation=cv2.INTER_LINEAR)
#     except Exception:
#         out = img_bgr
#     return _to_rgb(out)

# # ---------- BODY embeddings from gallery (UPDATED: Code-2 style) ----------
# def _compute_body_embedding_from_gallery(user_name: str,
#                                          on_step: Optional[Callable[[int], None]] = None
#                                          ) -> Optional[np.ndarray]:
#     if reid_extractor is None:
#         logger.warning("TorchReID not available; skipping body embedding.")
#         return None
#     root = _gallery_body_dir(user_name)
#     imgs = _list_images(root)
#     if not imgs:
#         logger.warning("No BODY crops found for '%s'", user_name)
#         return None

#     vectors: List[np.ndarray] = []
#     batch_imgs: List[np.ndarray] = []
#     B = 64 if (torch.cuda.is_available() and ("cuda" in str(DEVICE).lower())) else 32

#     # lightweight per-run caches (by absolute file path)
#     seen_hashes: set[int] = set()
#     body_cache: Dict[str, np.ndarray] = {}

#     def flush():
#         nonlocal batch_imgs, vectors
#         if not batch_imgs:
#             return
#         try:
#             with reid_lock, torch.inference_mode():
#                 feats = reid_extractor(batch_imgs)  # expects RGB
#             feats_np: List[np.ndarray] = []
#             for f in feats:
#                 f = f.detach().cpu().numpy() if hasattr(f, "detach") else np.asarray(f)
#                 f512 = _as_512f(f)
#                 if f512 is not None:
#                     feats_np.append(f512)
#             vectors.extend(feats_np)
#         except Exception:
#             logger.exception("TorchReID batch failed")
#         if on_step:
#             on_step(len(batch_imgs))
#         batch_imgs = []

#     for p in imgs:
#         img = _load_bgr(p)
#         if img is None or img.size == 0:
#             if on_step:
#                 on_step(1)
#             continue

#         h, w = img.shape[:2]
#         if min(h, w) < 32:
#             if on_step:
#                 on_step(1)
#             continue
#         if _is_blurry(img):
#             if on_step:
#                 on_step(1)
#             continue

#         hval = _ahash8(img)
#         if hval in seen_hashes:
#             if on_step:
#                 on_step(1)
#             continue
#         seen_hashes.add(hval)

#         # in-memory per image cache
#         k = str(p.resolve())
#         if k in body_cache:
#             vectors.append(body_cache[k])
#             if on_step:
#                 on_step(1)
#             continue

#         rgb = _prep_reid(img)
#         batch_imgs.append(rgb)

#         if len(batch_imgs) >= B:
#             # run extractor for the batch
#             before = len(vectors)
#             flush()
#             # cache just processed batch back to files (alignment by order)
#             # NOTE: we cannot reliably map per-file to per-feature post-batch without storing filenames;
#             # to keep logic simple and robust, we only cache singletons processed outside flush below.
#             processed = len(vectors) - before
#             # No per-file cache write here to avoid misalignment.

#     # final flush
#     flush()

#     if not vectors:
#         logger.warning("No valid BODY embeddings for '%s'", user_name)
#         return None
#     return _mean_embed(vectors)

# # ---------- FACE embeddings from gallery (UPDATED: Code-2 style) ----------
# def _compute_face_embedding_from_gallery(user_name: str,
#                                          on_step: Optional[Callable[[int], None]] = None
#                                          ) -> Optional[np.ndarray]:
#     if face_app is None:
#         logger.warning("InsightFace not available; skipping face embedding.")
#         return None
#     root = _gallery_face_dir(user_name)
#     imgs = _list_images(root)
#     if not imgs:
#         logger.warning("No FACE crops found for '%s'", user_name)
#         return None

#     vectors: List[np.ndarray] = []
#     face_cache: Dict[str, np.ndarray] = {}
#     seen_hashes: set[int] = set()

#     MAX_SIDE = 640  # detector budget
#     MIN_BODY_SIDE = 40  # sanity for body-crop faces

#     for p in imgs:
#         img = _load_bgr(p)
#         if img is None or img.size == 0:
#             if on_step:
#                 on_step(1)
#             continue

#         h, w = img.shape[:2]
#         if min(h, w) < MIN_BODY_SIDE:
#             if on_step:
#                 on_step(1)
#             continue

#         if _is_blurry(img):
#             if on_step:
#                 on_step(1)
#             continue

#         hval = _ahash8(img)
#         if hval in seen_hashes:
#             if on_step:
#                 on_step(1)
#             continue
#         seen_hashes.add(hval)

#         k = str(p.resolve())
#         if k in face_cache:
#             vectors.append(face_cache[k])
#             if on_step:
#                 on_step(1)
#             continue

#         # downscale if very large (keeps aspect)
#         if max(h, w) > MAX_SIDE:
#             scale = MAX_SIDE / float(max(h, w))
#             new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
#             img_proc = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#         else:
#             img_proc = img

#         try:
#             faces = face_app.get(np.ascontiguousarray(img_proc))
#         except Exception:
#             faces = []

#         best = None
#         best_score = -1.0
#         for f in safe_iter_faces(faces):
#             bbox = getattr(f, "bbox", None)
#             if bbox is None:
#                 continue
#             b = np.asarray(bbox).reshape(-1)
#             if b.size < 4:
#                 continue
#             x1, y1, x2, y2 = b[:4].astype(float)
#             area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
#             score = float(getattr(f, "det_score", 0.0))
#             # prefer confident, larger faces
#             ranking = score * (area ** 0.5)
#             if ranking > best_score:
#                 best = f
#                 best_score = ranking

#         if best is None:
#             if on_step:
#                 on_step(1)
#             continue

#         # quality gates
#         det_score = float(getattr(best, "det_score", 0.0))
#         if det_score < FACE_MIN_SCORE:
#             if on_step:
#                 on_step(1)
#             continue

#         emb = extract_face_embedding(best)  # prefers normed_embedding
#         f512 = _as_512f(emb)
#         if f512 is not None:
#             vectors.append(f512)
#             face_cache[k] = f512
#         if on_step:
#             on_step(1)

#     if not vectors:
#         logger.warning("No valid FACE embeddings for '%s'", user_name)
#         return None
#     return _mean_embed(vectors)

# def _update_db_embeddings(user_id: int,
#                           body_emb: Optional[np.ndarray],
#                           face_emb: Optional[np.ndarray]) -> Dict[str, object]:
#     session = get_session()
#     try:
#         user = session.query(User).filter(User.id == user_id).first()
#         if not user:
#             logger.warning("User ID %s not found. Skipping DB update.", user_id)
#             return {"status": "error", "message": "user not found"}

#         changed = False
#         if body_emb is not None:
#             user.body_embedding = body_emb.tolist()  # WHY: pgvector expects list/array
#             changed = True
#         if face_emb is not None:
#             user.face_embedding = face_emb.tolist()
#             changed = True
#         if changed:
#             user.last_embedding_update_ts = datetime.now(timezone.utc)
#             session.commit()
#             return {"status": "ok", "body": body_emb is not None, "face": face_emb is not None}
#         else:
#             return {"status": "no_embeddings"}
#     except Exception:
#         session.rollback()
#         logger.exception("DB error on update")
#         return {"status": "error", "message": "db error"}
#     finally:
#         session.close()

# # ---- Background extraction worker (runs on POST /extract) ----
# def _extract_worker(user_id: int, user_name: str):
#     try:
#         _progress_reset(user_id, user_name)
#         _progress_set(user_id, stage="scanning", message="Counting images...")
#         total_body, total_face = _count_images(user_name)
#         if total_body + total_face == 0:
#             _progress_set(user_id, total_body=0, total_face=0, percent=100,
#                           stage="done", message="No images to process")
#             return

#         _progress_set(user_id, total_body=total_body, total_face=total_face,
#                       done_body=0, done_face=0, percent=0)

#         # Body
#         body_emb = None
#         if total_body > 0 and reid_extractor is not None:
#             _progress_set(user_id, stage="embedding_body", message=f"Processing body ({total_body})")
#             def on_body(n: int):
#                 st = get_progress_for_user(user_id)
#                 _progress_set(user_id, done_body=int(st["done_body"]) + int(n))
#             body_emb = _compute_body_embedding_from_gallery(user_name, on_step=on_body)
#         else:
#             _progress_set(user_id, message="Skipping body (no images or model unavailable)")

#         # Face
#         face_emb = None
#         if total_face > 0 and face_app is not None:
#             _progress_set(user_id, stage="embedding_face", message=f"Processing face ({total_face})")
#             def on_face(n: int):
#                 st = get_progress_for_user(user_id)
#                 _progress_set(user_id, done_face=int(st["done_face"]) + int(n))
#             face_emb = _compute_face_embedding_from_gallery(user_name, on_step=on_face)
#         else:
#             _progress_set(user_id, message="Skipping face (no images or model unavailable)")

#         res = _update_db_embeddings(user_id, body_emb, face_emb)
#         status = str(res.get("status", "unknown"))
#         if status == "ok":
#             _progress_set(user_id, stage="done", message="Embeddings saved", percent=100)
#         else:
#             _progress_set(user_id, stage="done", message=status, percent=100)
#     except Exception as e:
#         logger.exception("extract worker failed")
#         _progress_set(user_id, stage="error", message=f"error: {e}")

# def extract_embeddings_async(user_id: int) -> Dict[str, object]:
#     """Kick off background extraction from saved galleries with progress reporting."""
#     # Ensure models present for embedding (YOLO not required here)
#     if reid_extractor is None or (USE_FACE and face_app is None):
#         init_models()

#     # Lookup user
#     session = get_session()
#     try:
#         user = session.query(User).filter(User.id == user_id).first()
#         if not user:
#             return {"status": "error", "message": f"user id {user_id} not found"}
#         user_name = user.name
#     except Exception:
#         session.rollback()
#         logger.exception("DB error during user lookup.")
#         return {"status": "error", "message": "DB error"}
#     finally:
#         session.close()

#     # Start worker
#     t = threading.Thread(target=_extract_worker, args=(user_id, user_name), daemon=True, name=f"extract-{user_id}")
#     t.start()
#     return {"status": "started", "id": user_id, "name": user_name}

# # ------------------ control API ------------------
# def start_extraction(user_id: int, show_viewer: bool = True) -> dict:
#     """Start capture+crop across all RTSP streams for the given user_id."""
#     global is_running, current_person_id, current_person_name
#     global viewer_thread

#     if is_running:
#         return {"status": "ok", "message": "already running", "num_cams": len(RTSP_STREAMS),
#                 "id": current_person_id, "name": current_person_name}

#     # Lookup user_name from DB (same pattern as extract_embeddings_async)
#     session = get_session()
#     try:
#         user = session.query(User).filter(User.id == int(user_id)).first()
#         if not user:
#             return {"status": "error", "message": f"user id {user_id} not found"}
#         user_name = str(user.name)
#     except Exception:
#         session.rollback()
#         logger.exception("DB error during user lookup (start_extraction).")
#         return {"status": "error", "message": "DB error"}
#     finally:
#         session.close()

#     # Ensure models are loaded (YOLO/face used during capture)
#     if yolo_model is None or (USE_FACE and face_app is None):
#         init_models()

#     # Reset global state
#     stop_event.clear()
#     with frames_lock:
#         latest_frames.clear()
#     for d in (capture_threads, extract_threads, cap_queues):
#         d.clear()

#     _progress_reset(int(user_id), user_name)
#     _progress_set(int(user_id), stage="starting", message="initializing")

#     current_person_id = int(user_id)
#     current_person_name = user_name

#     # Start IO worker
#     start_workers_if_needed()

#     # Launch per-camera capture + crop loops
#     for cam_idx, rtsp_url in enumerate(RTSP_STREAMS):
#         cap_queues[cam_idx] = Queue(maxsize=CAP_QUEUE_MAX)
#         t_cap = threading.Thread(target=capture_thread_fn, name=f"cap_{cam_idx}",
#                                  args=(cam_idx, rtsp_url), daemon=True)
#         t_ext = threading.Thread(target=embedding_loop_for_cam, name=f"ext_{cam_idx}",
#                                  args=(cam_idx, rtsp_url), daemon=True)
#         capture_threads[cam_idx] = t_cap
#         extract_threads[cam_idx] = t_ext
#         t_cap.start()
#         t_ext.start()

#     if show_viewer and (viewer_thread is None or not viewer_thread.is_alive()):
#         viewer_thread = threading.Thread(target=viewer_loop, name="viewer", daemon=True)
#         viewer_thread.start()

#     is_running = True
#     _progress_set(int(user_id), stage="running", message="processing")
#     logger.info("Extraction started for user_id=%s name=%s on %d stream(s)",
#                 user_id, user_name, len(RTSP_STREAMS))

#     return {"status": "ok", "message": "started", "num_cams": len(RTSP_STREAMS),
#             "id": int(user_id), "name": user_name}

# def stop_extraction(reason: str = "user") -> dict:
#     """Gracefully stop all threads and reset state."""
#     global is_running, current_person_id, current_person_name
#     try:
#         stop_event.set()

#         # Join per-camera threads
#         for d in (capture_threads, extract_threads):
#             for idx, th in list(d.items()):
#                 try:
#                     th.join(timeout=1.5)
#                 except Exception:
#                     pass

#         # Clear queues/maps
#         with frames_lock:
#             latest_frames.clear()
#         capture_threads.clear()
#         extract_threads.clear()
#         cap_queues.clear()

#         # Stop viewer
#         if viewer_thread is not None and viewer_thread.is_alive():
#             try:
#                 # Let viewer see the flag and exit
#                 pass
#             except Exception:
#                 pass

#         is_running = False
#         logger.info("Extraction stopped (%s).", reason)
#         return {"status": "ok", "message": f"stopped ({reason})"}
#     finally:
#         current_person_id = None
#         current_person_name = None

# def remove_embeddings(id: int):
#     """Remove stored embeddings (body + face)."""
#     session = get_session()
#     try:
#         user = session.query(User).filter(User.id == id).first()
#         if not user:
#             logger.error("User with id %d not found. Skipping embedding removal.", id)
#             return {"status": "error", "message": f"user id {id} not found" }
#         user.body_embedding = None
#         user.face_embedding = None
#         user.last_embedding_update_ts = datetime.now(timezone.utc)
#         session.commit()
#         logger.warning("Embeddings removed for user id=%d", id)
#         return {"status": "ok", "id": id}
#     except Exception:
#         session.rollback()
#         logger.exception("remove embeddings: DB error during user lookup.")
#         return {"status":"error", "message":"DB error"}
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

# -----------------------------------------------------------------------------------------
# # # file: mvision/services/extract_service.py
# from __future__ import annotations

# import csv
# import sys
# import os
# import time
# import threading
# from pathlib import Path
# from datetime import datetime, timezone
# from typing import Optional, List, Dict, Tuple, Callable
# from queue import Queue, Empty
# import logging
# import shutil
# import random

# import cv2
# import numpy as np
# import torch

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

# # ------------------------- runtime perf knobs -------------------------
# try:
#     torch.backends.cudnn.benchmark = True
# except Exception:
#     pass
# try:
#     if hasattr(torch, "set_float32_matmul_precision"):
#         torch.set_float32_matmul_precision("high")
# except Exception:
#     pass
# try:
#     cv2.setNumThreads(1)
# except Exception:
#     pass

# # ------------------------- optional libs -------------------------
# try:
#     from ultralytics import YOLO
# except Exception:
#     YOLO = None

# try:
#     from torchreid.utils import FeatureExtractor as TorchreidExtractor
# except Exception as e:
#     print("TorchReID import error:", e)
#     TorchreidExtractor = None

# try:
#     from insightface.app import FaceAnalysis
#     INSIGHT_OK = True
# except Exception:
#     FaceAnalysis = None
#     INSIGHT_OK = False

# try:
#     import onnxruntime as ort
# except Exception:
#     ort = None

# # ------------------------- config -------------------------
# EXPECTED_EMBED_DIM = 512

# # Face capture/link gates (capture-time only; aggregation stays unbiased)
# USE_FACE = True
# FACE_MODEL = "buffalo_l"
# FACE_DET_SIZE = (640, 640)         # (w, h)
# FACE_PROVIDER = "auto"             # "auto" | "cuda" | "cpu"
# FACE_MIN_SCORE = 0.5
# FACE_MIN_SIZE = 32                 # px min(side) of detected face
# FACE_IOU_LINK = 0.05
# FACE_OVER_FACE_LINK = 0.60
# FACE_EVERY = 2

# # YOLO inference size
# YOLO_IMGSZ = 640

# # Queues
# CAP_QUEUE_MAX = 2
# IO_QUEUE_MAX  = 1024

# # Viewer FPS throttle
# VIEWER_SLEEP_MS = 5

# # ------------------------- logging -------------------------
# logger = logging.getLogger("mvision.extract_service")
# if not logger.handlers:
#     ch = logging.StreamHandler()
#     ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
#     logger.addHandler(ch)
# logger.setLevel(logging.INFO)

# # ------------------------- globals -------------------------
# yolo_model: YOLO | None = None
# reid_extractor: TorchreidExtractor | None = None
# face_app: FaceAnalysis | None = None

# yolo_lock = threading.Lock()
# reid_lock = threading.Lock()
# csv_lock = threading.Lock()
# frames_lock = threading.Lock()

# latest_frames: Dict[int, np.ndarray] = {}

# capture_threads: Dict[int, threading.Thread] = {}
# extract_threads: Dict[int, threading.Thread] = {}
# cap_queues: Dict[int, Queue] = {}

# io_thread: Optional[threading.Thread] = None
# io_queue: Queue | None = None

# viewer_thread: Optional[threading.Thread] = None
# stop_event = threading.Event()
# is_running = False
# current_person_id: Optional[int] = None
# current_person_name: Optional[str] = None

# # ---- extraction progress state (for /extract) ----
# progress_lock = threading.Lock()
# progress_state: Dict[int, Dict[str, object]] = {}

# # ------------------ progress helpers ------------------
# def _progress_reset(user_id: int, user_name: str) -> None:
#     with progress_lock:
#         progress_state[user_id] = {
#             "user_name": user_name,
#             "stage": "idle",
#             "total_body": 0,
#             "total_face": 0,
#             "done_body": 0,
#             "done_face": 0,
#             "percent": 0,
#             "message": "waiting",
#         }

# def _progress_set(user_id: int, **kv) -> None:
#     with progress_lock:
#         st = progress_state.get(user_id)
#         if not st:
#             st = {}
#             progress_state[user_id] = st
#         st.update(kv)

#         # Clamp counters
#         tb, tf = int(st.get("total_body", 0)), int(st.get("total_face", 0))
#         db, df = int(st.get("done_body", 0)), int(st.get("done_face", 0))
#         db = max(0, min(db, tb))
#         df = max(0, min(df, tf))
#         st["done_body"], st["done_face"] = db, df

#         # Stage-aware percentage:
#         stage = str(st.get("stage", "")).lower()
#         if stage == "embedding_body" and tb > 0:
#             pct = int(round((db / max(1, tb)) * 100))
#         elif stage == "embedding_face" and tf > 0:
#             pct = int(round((df / max(1, tf)) * 100))
#         else:
#             tot = max(1, tb + tf)
#             pct = int(round(((db + df) / tot) * 100))

#         # Keep within [0, 100]
#         st["percent"] = max(0, min(100, pct))

# def get_progress_for_user(user_id: int) -> Dict[str, object]:
#     with progress_lock:
#         st = progress_state.get(user_id)
#         return dict(st) if st else {
#             "stage": "unknown",
#             "percent": 0,
#             "message": "no job",
#             "total_body": 0,
#             "total_face": 0,
#             "done_body": 0,
#             "done_face": 0,
#         }

# # ------------------ init_db (startup hook) ------------------
# def init_db() -> None:
#     try:
#         Base.metadata.create_all(bind=engine)
#         logger.info("DB initialized.")
#     except Exception:
#         logger.exception("DB init failed")

# # ------------------ utils ------------------
# def l2_normalize(v: np.ndarray) -> np.ndarray:
#     v = np.asarray(v, dtype=np.float32).reshape(-1)
#     n = float(np.linalg.norm(v))
#     if n == 0.0 or not np.isfinite(n):
#         return v
#     return v / n

# def _as_512f(vec: np.ndarray | list | None) -> Optional[np.ndarray]:
#     if vec is None:
#         return None
#     try:
#         a = np.asarray(vec, dtype=np.float32).reshape(-1)
#         if a.size != EXPECTED_EMBED_DIM:
#             return None
#         if not np.isfinite(a).all():
#             return None
#         return l2_normalize(a)
#     except Exception:
#         return None

# def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
#     return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
#     ax1, ay1, ax2, ay2 = a
#     bx1, by1, bx2, by2 = b
#     inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
#     inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
#     iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
#     inter = iw * ih
#     a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
#     b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
#     denom = a_area + b_area - inter
#     return inter / denom if denom > 0 else 0.0

# def inter_over_face(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
#     ax1, ay1, ax2, ay2 = a
#     bx1, by1, bx2, by2 = b
#     inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
#     inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
#     iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
#     inter = iw * ih
#     face_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
#     return inter / face_area if face_area > 0 else 0.0

# def face_center_in(a: Tuple[int, int, int, int], b: Tuple[float, float, float, float]) -> bool:
#     fx1, fy1, fx2, fy2 = a
#     px1, py1, px2, py2 = b
#     cx = (fx1 + fx2) * 0.5
#     cy = (fy1 + fy2) * 0.5
#     return (px1 <= cx <= px2) and (py1 <= cy <= py2)

# def safe_iter_faces(obj):
#     if obj is None:
#         return []
#     try:
#         return list(obj)
#     except TypeError:
#         return [obj]

# def extract_face_embedding(face):
#     emb = getattr(face, "normed_embedding", None)
#     if emb is None:
#         emb = getattr(face, "embedding", None)
#     return emb

# def _cuda_ep_loadable() -> bool:
#     if ort is None:
#         return False
#     try:
#         if sys.platform.startswith("mac"):
#             return False
#         from pathlib import Path as _P
#         import ctypes as _ct
#         capi_dir = _P(ort.__file__).parent / "capi"
#         name = "onnxruntime_providers_cuda.dll" if os.name == "nt" else "libonnxruntime_providers_cuda.so"
#         lib_path = capi_dir / name
#         if not lib_path.exists():
#             return False
#         _ct.CDLL(str(lib_path))
#         return True
#     except Exception:
#         return False

# # ------------------ models init ------------------
# def init_face_engine(use_face: bool, device: str, face_model: str, det_w: int, det_h: int, face_provider: str):
#     if not use_face:
#         return None
#     if not INSIGHT_OK:
#         logger.warning("insightface not installed; face recognition disabled.")
#         return None
#     try:
#         is_cuda = ("cuda" in device.lower()) and torch.cuda.is_available()
#         cuda_ok = _cuda_ep_loadable()

#         providers = ["CPUExecutionProvider"]
#         if face_provider == "cuda":
#             if cuda_ok:
#                 providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
#         elif face_provider == "auto":
#             if is_cuda and cuda_ok:
#                 providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

#         app = FaceAnalysis(name=face_model, providers=providers)
#         ctx_id = 0 if providers[0].startswith("CUDA") else -1
#         try:
#             app.prepare(ctx_id=ctx_id, det_size=(det_w, det_h))
#         except TypeError:
#             app.prepare(ctx_id=ctx_id)
#         logger.info("InsightFace ready (model=%s, providers=%s).", face_model, providers)

#         # Warm-up (avoids first-call latency)
#         try:
#             dummy = np.zeros((max(8, det_h), max(8, det_w), 3), dtype=np.uint8)
#             app.get(dummy)
#         except Exception:
#             pass

#         return app
#     except Exception:
#         logger.exception("InsightFace init failed")
#         return None

# def init_models():
#     """Load YOLO + TorchReID + InsightFace once on startup."""
#     global yolo_model, reid_extractor, face_app

#     gpu = torch.cuda.is_available() and ("cuda" in str(DEVICE).lower())
#     logger.info("init_models device=%s gpu_available=%s", DEVICE, torch.cuda.is_available())

#     # YOLO
#     if YOLO is not None:
#         try:
#             weights = YOLO_WEIGHTS
#             if not Path(weights).exists():
#                 logger.warning("%s not found, falling back to yolov8n.pt", weights)
#                 weights = "yolov8n.pt"
#             yolo = YOLO(weights)
#             if gpu:
#                 try:
#                     yolo.to(DEVICE)
#                 except Exception:
#                     pass
#             # Warm-up
#             try:
#                 with torch.inference_mode():
#                     _dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
#                     yolo.predict(_dummy, device=DEVICE if gpu else "cpu",
#                                  conf=0.25, iou=0.5, imgsz=YOLO_IMGSZ, verbose=False)
#             except Exception:
#                 pass
#             yolo_model = yolo
#             logger.info("YOLO ready (gpu=%s)", gpu)
#         except Exception:
#             logger.exception("YOLO load failed; detection disabled.")
#             yolo_model = None
#     else:
#         logger.warning("ultralytics not installed; detection disabled.")

#     # TorchReID (used for embeddings)
#     if TorchreidExtractor is not None:
#         try:
#             dev = DEVICE if gpu else "cpu"
#             reid_extractor = TorchreidExtractor(model_name="osnet_x0_25", device=dev)
#             # Warm-up
#             try:
#                 dummy = np.zeros((256, 128, 3), dtype=np.uint8)
#                 _ = reid_extractor([dummy])
#             except Exception:
#                 pass
#             logger.info("TorchReID ready (device=%s)", dev)
#         except Exception:
#             logger.exception("TorchReID init failed; body embeddings disabled.")
#             reid_extractor = None
#     else:
#         logger.warning("torchreid not installed; body embeddings disabled.")

#     # InsightFace
#     globals()["face_app"] = init_face_engine(
#         USE_FACE, DEVICE, FACE_MODEL, FACE_DET_SIZE[0], FACE_DET_SIZE[1], FACE_PROVIDER
#     )

# # ------------------ CSV + gallery helpers ------------------
# def _ensure_dir(p: Path) -> None:
#     p.mkdir(parents=True, exist_ok=True)

# def ensure_csv_header():
#     _ensure_dir(Path(EMB_CSV).parent if Path(EMB_CSV).parent.as_posix() not in (".", "") else Path("."))
#     if not Path(EMB_CSV).exists():
#         with csv_lock:
#             with open(EMB_CSV, "w", newline="", encoding="utf-8") as f:
#                 w = csv.writer(f)
#                 w.writerow(
#                     [
#                         "user_id","user_name","ts","cam_idx","frame_idx","det_idx",
#                         "x1","y1","x2","y2","conf_or_score",
#                         "body_embedding","face_embedding","crop_path","kind",
#                     ]
#                 )
#         logger.info("Created CSV header: %s", EMB_CSV)

# def _gallery_body_dir(user_name: str) -> Path:
#     d = Path(CROPS_ROOT) / "gallery" / user_name
#     _ensure_dir(d)
#     return d

# def _gallery_face_dir(user_name: str) -> Path:
#     d = Path(CROPS_ROOT) / "gallery_face" / user_name
#     _ensure_dir(d)
#     return d

# def clear_user_galleries(user_name: str):
#     for base in ("gallery", "gallery_face"):
#         root = Path(CROPS_ROOT) / base / user_name
#         if root.exists():
#             shutil.rmtree(root, ignore_errors=True)
#         _ensure_dir(root)
#     logger.info("Cleared galleries for '%s'", user_name)

# def _epoch_ms() -> int:
#     # microsecond-ish uniqueness with small random tail
#     return int(time.time() * 1000) * 1000 + random.randint(0, 999)

# # ------------------ face helpers (capture-time) ------------------
# def detect_faces_raw(frame: np.ndarray):
#     outs: List[Dict] = []
#     if face_app is None:
#         return outs
#     try:
#         faces = face_app.get(np.ascontiguousarray(frame))
#         for f in safe_iter_faces(faces):
#             bbox = getattr(f, "bbox", None)
#             if bbox is None:
#                 continue
#             b = np.asarray(bbox).reshape(-1)
#             if b.size < 4:
#                 continue
#             x1, y1, x2, y2 = map(float, b[:4])
#             score = float(getattr(f, "det_score", 0.0))
#             outs.append({"bbox": (x1, y1, x2, y2), "score": score})
#     except Exception:
#         logger.exception("FaceAnalysis error")
#     return outs

# def link_face_to_person(face_bbox: Tuple[float, float, float, float],
#                         person_bbox: Tuple[int, int, int, int]) -> bool:
#     if face_center_in(tuple(map(int, face_bbox)), person_bbox):
#         return True
#     if iou_xyxy(person_bbox, face_bbox) >= FACE_IOU_LINK:
#         return True
#     if inter_over_face(person_bbox, face_bbox) >= FACE_OVER_FACE_LINK:
#         return True
#     return False

# # ------------------ RTSP helpers ------------------
# def _configure_rtsp_capture(cap: cv2.VideoCapture) -> None:
#     try:
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#     except Exception:
#         pass

# def _read_latest_frame(cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
#     grabbed = cap.grab()
#     if not grabbed:
#         ok, fr = cap.read()
#         return ok, (fr if ok else None)
#     for _ in range(3):
#         if not cap.grab():
#             break
#     ok, frame = cap.retrieve()
#     return ok, (frame if ok else None)

# # ------------------ async workers (IO) ------------------
# def start_workers_if_needed():
#     global io_thread, io_queue
#     if io_queue is None:
#         io_queue = Queue(maxsize=IO_QUEUE_MAX)
#     if io_thread is None or not io_thread.is_alive():
#         io_thread = threading.Thread(target=io_worker, name="io_worker", daemon=True)
#         io_thread.start()

# def io_worker():
#     while not stop_event.is_set():
#         try:
#             job = io_queue.get(timeout=0.1)  # type: ignore
#         except Empty:
#             continue
#         except Exception:
#             break
#         try:
#             # WHY: file and CSV write off main thread to avoid capture stalls
#             crop_path: Path = job["crop_path"]
#             _ensure_dir(crop_path.parent)
#             cv2.imwrite(str(crop_path), job["image"])
#             with csv_lock:
#                 with open(EMB_CSV, "a", newline="", encoding="utf-8") as f:
#                     csv.writer(f).writerow(job["csv_row"])
#         except Exception:
#             logger.exception("IO worker error")

# # ------------------ capture & processing loops ------------------
# def capture_thread_fn(cam_idx: int, rtsp_url: str):
#     q = cap_queues[cam_idx]
#     cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
#     if not cap.isOpened():
#         logger.error("[cap%d] cannot open RTSP: %s", cam_idx, rtsp_url)
#         return
#     _configure_rtsp_capture(cap)
#     logger.info("[cap%d] started on %s", cam_idx, rtsp_url)
#     try:
#         while not stop_event.is_set():
#             ok, frame = _read_latest_frame(cap)
#             if not ok or frame is None:
#                 time.sleep(0.01)
#                 continue
#             while not q.empty():
#                 try:
#                     q.get_nowait()
#                 except Exception:
#                     break
#             try:
#                 q.put_nowait(frame)
#             except Exception:
#                 pass
#     finally:
#         cap.release()
#         logger.info("[cap%d] stopped", cam_idx)

# def embedding_loop_for_cam(cam_idx: int, rtsp_url: str):
#     """
#     Save crops only:
#     - BODY: each person detection → gallery/<user>/person_<ms>.jpg
#     - FACE: when a clear face is linked → gallery_face/<user>/person_<ms>.jpg (same full-body crop)
#     """
#     global current_person_id, current_person_name

#     # Always ensure CSV + IO worker are ready before first enqueue.
#     ensure_csv_header()
#     start_workers_if_needed()

#     user_folder = current_person_name or "unknown"
#     body_dir = _gallery_body_dir(user_folder)
#     face_dir = _gallery_face_dir(user_folder)

#     frame_idx = 0
#     logger.info("[Loop cam%d] started on %s", cam_idx, rtsp_url)
#     q = cap_queues[cam_idx]

#     try:
#         while not stop_event.is_set():
#             try:
#                 frame = q.get(timeout=0.2)
#             except Empty:
#                 continue
#             except Exception:
#                 break
#             if frame is None:
#                 continue

#             frame_idx += 1
#             H, W = frame.shape[:2]
#             detections: List[tuple[float, float, float, float, float]] = []

#             # Avoid aliasing: draw on a copy only for viewer.
#             vis = frame.copy()  # WHY: downstream consumers might reuse the source frame

#             # YOLO detection
#             if yolo_model is not None:
#                 try:
#                     with yolo_lock, torch.inference_mode():
#                         res = yolo_model.predict(
#                             frame,
#                             conf=float(CONF_THRES),
#                             iou=float(IOU_THRES),
#                             imgsz=YOLO_IMGSZ,
#                             verbose=False,
#                             device=DEVICE if torch.cuda.is_available() and ("cuda" in str(DEVICE).lower()) else "cpu",
#                         )
#                     boxes = res[0].boxes if (res and len(res)) else None
#                     if boxes is not None:
#                         xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
#                         confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
#                         cls = boxes.cls.detach().cpu().numpy().astype(np.int32)
#                         keep = (cls == 0)  # person
#                         xyxy, confs = xyxy[keep], confs[keep]
#                         for (x1, y1, x2, y2), c in zip(xyxy, confs):
#                             x1f = float(max(0, min(W - 1, x1)))
#                             y1f = float(max(0, min(H - 1, y1)))
#                             x2f = float(max(0, min(W - 1, x2)))
#                             y2f = float(max(0, min(H - 1, y2)))
#                             if (x2f - x1f) < 4 or (y2f - y1f) < 4:
#                                 continue
#                             detections.append((x1f, y1f, x2f, y2f, float(c)))
#                 except Exception:
#                     logger.exception("[Loop cam%d] YOLO error", cam_idx)

#             # Faces on frame (stride)
#             faces = []
#             if USE_FACE and (frame_idx % max(1, FACE_EVERY) == 0):
#                 faces = detect_faces_raw(frame)

#             det_boxes_i: List[tuple[int, int, int, int, float]] = []
#             for (x1, y1, x2, y2, conf) in detections:
#                 x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
#                 x1i, y1i = max(0, x1i), max(0, y1i)
#                 x2i, y2i = min(W, x2i), min(H, y2i)
#                 if x2i <= x1i or y2i <= y1i:
#                     continue
#                 det_boxes_i.append((x1i, y1i, x2i, y2i, conf))

#                 # Draw for viewer only
#                 try:
#                     cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
#                     cv2.putText(
#                         vis, f"{current_person_id or ''} cam{cam_idx}", (x1i, max(0, y1i - 5)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
#                     )
#                 except Exception:
#                     pass

#                 # Save BODY crop (always)
#                 crop_bgr = frame[y1i:y2i, x1i:x2i]
#                 if crop_bgr.size > 0:
#                     ts = time.time()
#                     fname = f"person_{_epoch_ms()}.jpg"
#                     path = body_dir / fname
#                     row = [
#                         current_person_id, current_person_name, ts,
#                         cam_idx, frame_idx, len(det_boxes_i)-1,
#                         x1i, y1i, x2i, y2i,
#                         conf,
#                         "", "", str(path), "body",
#                     ]
#                     try:
#                         if io_queue is not None:
#                             io_queue.put_nowait({"crop_path": path, "image": crop_bgr, "csv_row": row})
#                     except Exception:
#                         # Drop silently under backpressure; loop keeps going.
#                         pass

#             # Save FACE (clear) crops → same full-body crop into FACE gallery
#             if det_boxes_i and faces:
#                 ts = time.time()
#                 for det_idx, (x1i, y1i, x2i, y2i, _conf) in enumerate(det_boxes_i):
#                     t_xyxy = (x1i, y1i, x2i, y2i)
#                     best_idx, best_score = -1, -1.0
#                     best_box = None
#                     for idx, fm in enumerate(faces):
#                         if link_face_to_person(fm["bbox"], t_xyxy):
#                             s = float(fm.get("score", 0.0))
#                             if s > best_score:
#                                 best_score, best_idx = s, idx
#                                 best_box = fm["bbox"]
#                     if best_idx < 0:
#                         continue
#                     fx1, fy1, fx2, fy2 = map(int, best_box)
#                     if best_score < FACE_MIN_SCORE:
#                         continue
#                     if min(fx2 - fx1, fy2 - fy1) < FACE_MIN_SIZE:
#                         continue

#                     body_crop = frame[y1i:y2i, x1i:x2i]
#                     if body_crop.size <= 0:
#                         continue
#                     fname = f"person_{_epoch_ms()}.jpg"
#                     path = face_dir / fname
#                     row = [
#                         current_person_id, current_person_name, ts,
#                         cam_idx, frame_idx, det_idx,
#                         x1i, y1i, x2i, y2i,
#                         best_score,
#                         "", "", str(path), "face",
#                     ]
#                     try:
#                         if io_queue is not None:
#                             io_queue.put_nowait({"crop_path": path, "image": body_crop, "csv_row": row})
#                     except Exception:
#                         pass

#             with frames_lock:
#                 latest_frames[cam_idx] = vis

#             if frame_idx % 50 == 0:
#                 logger.info("[Loop cam%d] frame %d, persons=%d, faces=%d",
#                             cam_idx, frame_idx, len(det_boxes_i), len(faces) if faces else 0)

#     finally:
#         logger.info("[Loop cam%d] stopped", cam_idx)

# # ------------------ viewer ------------------
# def viewer_loop():
#     logger.info("[Viewer] started")
#     window_name = "Multi-RTSP Viewer"
#     window_created = False
#     try:
#         while not stop_event.is_set():
#             with frames_lock:
#                 frames = [
#                     latest_frames[idx]
#                     for idx in sorted(latest_frames.keys())
#                     if latest_frames.get(idx) is not None
#                 ]

#             if not frames:
#                 time.sleep(0.01)
#                 continue

#             if not window_created:
#                 try:
#                     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#                     window_created = True
#                 except Exception:
#                     pass

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

#             try:
#                 cv2.imshow(window_name, combined)
#             except Exception:
#                 window_created = False
#                 time.sleep(0.01)
#                 continue

#             key = cv2.waitKey(1) & 0xFF
#             if key in (ord("q"), 27):
#                 logger.info("[Viewer] key pressed, stopping...")
#                 stop_event.set()
#                 break

#             time.sleep(VIEWER_SLEEP_MS / 1000.0)
#     finally:
#         try:
#             cv2.destroyAllWindows()
#         except Exception:
#             pass
#         logger.info("[Viewer] stopped")

# # ------------------ finalize / extract from folders ------------------
# def _list_images(root: Path) -> List[Path]:
#     if not root.exists():
#         return []
#     pats = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
#     files: List[Path] = []
#     for p in pats:
#         files.extend(root.glob(p))
#     return sorted(files)

# def _load_bgr(path: Path) -> Optional[np.ndarray]:
#     try:
#         img = cv2.imread(str(path), cv2.IMREAD_COLOR)
#         return img if img is not None else None
#     except Exception:
#         return None

# def _mean_embed(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
#     if not vectors:
#         return None
#     arr = np.stack(vectors, axis=0).astype(np.float32)
#     m = np.mean(arr, axis=0)
#     return l2_normalize(m)

# def _count_images(user_name: str) -> Tuple[int, int]:
#     body = len(_list_images(_gallery_body_dir(user_name)))
#     face = len(_list_images(_gallery_face_dir(user_name)))
#     return body, face

# # ---------- BODY embeddings from gallery (Code-1 style) ----------
# def _compute_body_embedding_from_gallery(user_name: str,
#                                          on_step: Optional[Callable[[int], None]] = None
#                                          ) -> Optional[np.ndarray]:
#     if reid_extractor is None:
#         logger.warning("TorchReID not available; skipping body embedding.")
#         return None
#     root = _gallery_body_dir(user_name)
#     imgs = _list_images(root)
#     if not imgs:
#         logger.warning("No BODY crops found for '%s'", user_name)
#         return None

#     vectors: List[np.ndarray] = []
#     batch_imgs: List[np.ndarray] = []
#     B = 32

#     def flush():
#         nonlocal batch_imgs, vectors
#         if not batch_imgs:
#             return
#         try:
#             with reid_lock, torch.inference_mode():
#                 feats = reid_extractor(batch_imgs)  # accepts RGB arrays
#             # WHY: reid_extractor may return torch.Tensor; normalize & shape-check
#             feats_np = []
#             for f in feats:
#                 f = f.detach().cpu().numpy() if hasattr(f, "detach") else np.asarray(f)
#                 f512 = _as_512f(f)
#                 if f512 is not None:
#                     feats_np.append(f512)
#             vectors.extend(feats_np)
#         except Exception:
#             logger.exception("TorchReID batch failed")
#         if on_step:
#             on_step(len(batch_imgs))
#         batch_imgs = []

#     for p in imgs:
#         img = _load_bgr(p)
#         if img is None or img.size == 0:
#             if on_step:
#                 on_step(1)  # count skipped
#             continue
#         batch_imgs.append(_to_rgb(img))
#         if len(batch_imgs) >= B:
#             flush()
#     flush()

#     if not vectors:
#         logger.warning("No valid BODY embeddings for '%s'", user_name)
#         return None
#     return _mean_embed(vectors)

# # ---------- FACE embeddings from gallery (Code-1 style) ----------
# def _compute_face_embedding_from_gallery(user_name: str,
#                                          on_step: Optional[Callable[[int], None]] = None
#                                          ) -> Optional[np.ndarray]:
#     if face_app is None:
#         logger.warning("InsightFace not available; skipping face embedding.")
#         return None
#     root = _gallery_face_dir(user_name)
#     imgs = _list_images(root)
#     if not imgs:
#         logger.warning("No FACE crops found for '%s'", user_name)
#         return None

#     vectors: List[np.ndarray] = []

#     # WHY: Downscale large images before detection to speed up InsightFace.
#     #      We only need the embedding from the detected face, not exact coords.
#     MAX_SIDE = 640  # stay aligned with FACE_DET_SIZE upper bound

#     for p in imgs:
#         img = _load_bgr(p)
#         if img is None or img.size == 0:
#             if on_step:
#                 on_step(1)
#             continue

#         try:
#             h, w = img.shape[:2]
#             # Downscale if needed (preserve aspect ratio); avoids heavy detector work on big crops.
#             if max(h, w) > MAX_SIDE:
#                 scale = MAX_SIDE / float(max(h, w))
#                 new_w, new_h = int(w * scale), int(h * scale)
#                 if new_w > 0 and new_h > 0:
#                     img_proc = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#                 else:
#                     img_proc = img
#             else:
#                 img_proc = img

#             faces = face_app.get(np.ascontiguousarray(img_proc))
#         except Exception:
#             faces = []
#         faces_iter = safe_iter_faces(faces)

#         # pick largest face by area (on the processed image)
#         best = None
#         best_area = -1.0
#         for f in faces_iter:
#             bbox = getattr(f, "bbox", None)
#             if bbox is None:
#                 continue
#             b = np.asarray(bbox).reshape(-1)
#             if b.size < 4:
#                 continue
#             x1, y1, x2, y2 = b[:4].astype(float)
#             area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
#             if area > best_area:
#                 best, best_area = f, area

#         if best is None:
#             if on_step:
#                 on_step(1)
#             continue

#         emb = extract_face_embedding(best)  # prefers normed_embedding if present
#         f512 = _as_512f(emb)
#         if f512 is not None:
#             vectors.append(f512)
#         if on_step:
#             on_step(1)

#     if not vectors:
#         logger.warning("No valid FACE embeddings for '%s'", user_name)
#         return None
#     return _mean_embed(vectors)

# def _update_db_embeddings(user_id: int,
#                           body_emb: Optional[np.ndarray],
#                           face_emb: Optional[np.ndarray]) -> Dict[str, object]:
#     session = get_session()
#     try:
#         user = session.query(User).filter(User.id == user_id).first()
#         if not user:
#             logger.warning("User ID %s not found. Skipping DB update.", user_id)
#             return {"status": "error", "message": "user not found"}

#         changed = False
#         if body_emb is not None:
#             user.body_embedding = body_emb.tolist()  # WHY: pgvector expects list/array
#             changed = True
#         if face_emb is not None:
#             user.face_embedding = face_emb.tolist()
#             changed = True
#         if changed:
#             user.last_embedding_update_ts = datetime.now(timezone.utc)
#             session.commit()
#             return {"status": "ok", "body": body_emb is not None, "face": face_emb is not None}
#         else:
#             return {"status": "no_embeddings"}
#     except Exception:
#         session.rollback()
#         logger.exception("DB error on update")
#         return {"status": "error", "message": "db error"}
#     finally:
#         session.close()

# # ---- Background extraction worker (runs on POST /extract) ----
# def _extract_worker(user_id: int, user_name: str):
#     try:
#         _progress_reset(user_id, user_name)
#         _progress_set(user_id, stage="scanning", message="Counting images...")
#         total_body, total_face = _count_images(user_name)
#         if total_body + total_face == 0:
#             _progress_set(user_id, total_body=0, total_face=0, percent=100,
#                           stage="done", message="No images to process")
#             return

#         _progress_set(user_id, total_body=total_body, total_face=total_face,
#                       done_body=0, done_face=0, percent=0)

#         # Body
#         body_emb = None
#         if total_body > 0 and reid_extractor is not None:
#             _progress_set(user_id, stage="embedding_body", message=f"Processing body ({total_body})")
#             def on_body(n: int):
#                 st = get_progress_for_user(user_id)
#                 _progress_set(user_id, done_body=int(st["done_body"]) + int(n))
#             body_emb = _compute_body_embedding_from_gallery(user_name, on_step=on_body)
#         else:
#             _progress_set(user_id, message="Skipping body (no images or model unavailable)")

#         # Face
#         face_emb = None
#         if total_face > 0 and face_app is not None:
#             _progress_set(user_id, stage="embedding_face", message=f"Processing face ({total_face})")
#             def on_face(n: int):
#                 st = get_progress_for_user(user_id)
#                 _progress_set(user_id, done_face=int(st["done_face"]) + int(n))
#             face_emb = _compute_face_embedding_from_gallery(user_name, on_step=on_face)
#         else:
#             _progress_set(user_id, message="Skipping face (no images or model unavailable)")

#         res = _update_db_embeddings(user_id, body_emb, face_emb)
#         status = str(res.get("status", "unknown"))
#         if status == "ok":
#             _progress_set(user_id, stage="done", message="Embeddings saved", percent=100)
#         else:
#             _progress_set(user_id, stage="done", message=status, percent=100)
#     except Exception as e:
#         logger.exception("extract worker failed")
#         _progress_set(user_id, stage="error", message=f"error: {e}")

# def extract_embeddings_async(user_id: int) -> Dict[str, object]:
#     """Kick off background extraction from saved galleries with progress reporting."""
#     # Ensure models present for embedding (YOLO not required here)
#     if reid_extractor is None or (USE_FACE and face_app is None):
#         init_models()

#     # Lookup user
#     session = get_session()
#     try:
#         user = session.query(User).filter(User.id == user_id).first()
#         if not user:
#             return {"status": "error", "message": f"user id {user_id} not found"}
#         user_name = user.name
#     except Exception:
#         session.rollback()
#         logger.exception("DB error during user lookup.")
#         return {"status": "error", "message": "DB error"}
#     finally:
#         session.close()

#     # Start worker
#     t = threading.Thread(target=_extract_worker, args=(user_id, user_name), daemon=True, name=f"extract-{user_id}")
#     t.start()
#     return {"status": "started", "id": user_id, "name": user_name}

# # ------------------ control API ------------------
# def start_extraction(user_id: int, show_viewer: bool = True) -> dict:
#     """Start capture+crop across all RTSP streams for the given user_id."""
#     global is_running, current_person_id, current_person_name
#     global viewer_thread

#     if is_running:
#         return {"status": "ok", "message": "already running", "num_cams": len(RTSP_STREAMS),
#                 "id": current_person_id, "name": current_person_name}

#     # Lookup user_name from DB (same pattern as extract_embeddings_async)
#     session = get_session()
#     try:
#         user = session.query(User).filter(User.id == int(user_id)).first()
#         if not user:
#             return {"status": "error", "message": f"user id {user_id} not found"}
#         user_name = str(user.name)
#     except Exception:
#         session.rollback()
#         logger.exception("DB error during user lookup (start_extraction).")
#         return {"status": "error", "message": "DB error"}
#     finally:
#         session.close()

#     # Ensure models are loaded (YOLO/face used during capture)
#     if yolo_model is None or (USE_FACE and face_app is None):
#         init_models()

#     # Reset global state
#     stop_event.clear()
#     with frames_lock:
#         latest_frames.clear()
#     for d in (capture_threads, extract_threads, cap_queues):
#         d.clear()

#     _progress_reset(int(user_id), user_name)
#     _progress_set(int(user_id), stage="starting", message="initializing")

#     current_person_id = int(user_id)
#     current_person_name = user_name

#     # Start IO worker
#     start_workers_if_needed()

#     # Launch per-camera capture + crop loops
#     for cam_idx, rtsp_url in enumerate(RTSP_STREAMS):
#         cap_queues[cam_idx] = Queue(maxsize=CAP_QUEUE_MAX)
#         t_cap = threading.Thread(target=capture_thread_fn, name=f"cap_{cam_idx}",
#                                  args=(cam_idx, rtsp_url), daemon=True)
#         t_ext = threading.Thread(target=embedding_loop_for_cam, name=f"ext_{cam_idx}",
#                                  args=(cam_idx, rtsp_url), daemon=True)
#         capture_threads[cam_idx] = t_cap
#         extract_threads[cam_idx] = t_ext
#         t_cap.start()
#         t_ext.start()

#     if show_viewer and (viewer_thread is None or not viewer_thread.is_alive()):
#         viewer_thread = threading.Thread(target=viewer_loop, name="viewer", daemon=True)
#         viewer_thread.start()

#     is_running = True
#     _progress_set(int(user_id), stage="running", message="processing")
#     logger.info("Extraction started for user_id=%s name=%s on %d stream(s)",
#                 user_id, user_name, len(RTSP_STREAMS))

#     return {"status": "ok", "message": "started", "num_cams": len(RTSP_STREAMS),
#             "id": int(user_id), "name": user_name}

# def stop_extraction(reason: str = "user") -> dict:
#     """Gracefully stop all threads and reset state."""
#     global is_running, current_person_id, current_person_name
#     try:
#         stop_event.set()

#         # Join per-camera threads
#         for d in (capture_threads, extract_threads):
#             for idx, th in list(d.items()):
#                 try:
#                     th.join(timeout=1.5)
#                 except Exception:
#                     pass

#         # Clear queues/maps
#         with frames_lock:
#             latest_frames.clear()
#         capture_threads.clear()
#         extract_threads.clear()
#         cap_queues.clear()

#         # Stop viewer
#         if viewer_thread is not None and viewer_thread.is_alive():
#             try:
#                 # Let viewer see the flag and exit
#                 pass
#             except Exception:
#                 pass

#         is_running = False
#         logger.info("Extraction stopped (%s).", reason)
#         return {"status": "ok", "message": f"stopped ({reason})"}
#     finally:
#         current_person_id = None
#         current_person_name = None

# def remove_embeddings(id: int):
#     """Remove stored embeddings (body + face)."""
#     session = get_session()
#     try:
#         user = session.query(User).filter(User.id == id).first()
#         if not user:
#             logger.error("User with id %d not found. Skipping embedding removal.", id)
#             return {"status": "error", "message": f"user id {id} not found" }
#         user.body_embedding = None
#         user.face_embedding = None
#         user.last_embedding_update_ts = datetime.now(timezone.utc)
#         session.commit()
#         logger.warning("Embeddings removed for user id=%d", id)
#         return {"status": "ok", "id": id}
#     except Exception:
#         session.rollback()
#         logger.exception("remove embeddings: DB error during user lookup.")
#         return {"status":"error", "message":"DB error"}
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

# ---------------------------------------------------------------------------------------------------

# # file: services/extract_service.py
# # file: mvision/services/extract_service.py
# # ================================
# # file: mvision/services/extract_service.py
# # ================================

# file: mvision/services/extract_service.py
# file: mvision/services/extract_service.py
# file: mvision/services/extract_service.py
# file: mvision/services/extract_service.py
# from __future__ import annotations

# import logging
# import os
# import re
# import threading
# import time
# from dataclasses import dataclass
# from pathlib import Path
# from queue import Queue, Empty
# from typing import Dict, List, Optional, Tuple, Callable

# import cv2
# import numpy as np
# import torch
# from sqlalchemy import text

# from mvision.db.session import get_session
# from mvision.db.models import User

# # -------------------- Perf knobs (WHY: maximize GPU usage) --------------------
# try:
#     torch.backends.cudnn.benchmark = True
# except Exception:
#     pass
# try:
#     torch.set_float32_matmul_precision("high")  # torch>=2.0
# except Exception:
#     pass

# # -------------------- Optional model deps --------------------
# try:
#     from ultralytics import YOLO
#     _HAS_ULTRALYTICS = True
# except Exception:
#     YOLO = None  # type: ignore
#     _HAS_ULTRALYTICS = False

# try:
#     from insightface.app import FaceAnalysis
#     _HAS_INSIGHT = True
# except Exception:
#     FaceAnalysis = None  # type: ignore
#     _HAS_INSIGHT = False

# try:
#     from torchreid.utils import FeatureExtractor as TorchreidExtractor
#     _HAS_TORCHREID = True
# except Exception:
#     TorchreidExtractor = None  # type: ignore
#     _HAS_TORCHREID = False

# # -------------------- logging --------------------
# logger = logging.getLogger("extract_service")
# if not logger.handlers:
#     h = logging.StreamHandler()
#     h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s"))
#     logger.addHandler(h)
# logger.setLevel(logging.INFO)

# # -------------------- config --------------------
# # Root exactly as requested (Windows path)
# CROPS_ROOT = Path(r"D:\mvision_api\crops")

# EXPECTED_EMBED_DIM = 512
# BATCH_SIZE = 64  # larger batch for GPU
# CAP_FPS_THROTTLE = 0.0
# CAP_MAX_BODY = 200
# CAP_MAX_FACE = 200
# PERSON_CLASS_ID = 0
# YOLO_CONF = float(os.getenv("MVISION_YOLO_CONF", "0.35"))
# FACE_MIN_SCORE = float(os.getenv("MVISION_FACE_MIN_SCORE", "0.6"))

# _ON = {"1", "true", "yes", "on"}
# SHOW_IMSHOW = os.getenv("MVISION_SHOW_IMSHOW", "1").lower() in _ON
# AUTO_CLOSE_VIEWER = os.getenv("MVISION_IMSHOW_AUTOCLOSE", "1").lower() in _ON
# WINDOW_NAME = os.getenv("MVISION_IMSHOW_WINDOW", "MVision Preview")

# # Default to your RTSP; ENV MVISION_RTSP overrides.
# RTSP_STREAMS: List[str] = [
#     os.getenv("MVISION_RTSP", "rtsp://admin:rolex%40123@192.168.1.110:554/Streaming/channels/101")
# ]

# # -------------------- globals --------------------
# is_running: bool = False
# current_person_id: Optional[int] = None
# current_person_name: Optional[str] = None

# stop_event = threading.Event()
# _capture_threads: Dict[int, threading.Thread] = {}
# _extract_threads: Dict[int, threading.Thread] = {}

# _progress_lock = threading.Lock()
# _progress_map: Dict[int, Dict[str, int | str]] = {}

# # Models (lazy singletons)
# _reid_lock = threading.Lock()
# _reid_extractor: Optional[TorchreidExtractor] = None
# _yolo_lock = threading.Lock()
# _yolo_model: Optional[YOLO] = None  # type: ignore
# _face_lock = threading.Lock()
# _face_app: Optional[FaceAnalysis] = None  # type: ignore

# # Viewer state
# viewer_thread: Optional[threading.Thread] = None
# viewer_queue: "Queue[np.ndarray]" = Queue(maxsize=16)
# viewer_running = threading.Event()
# _HAS_CV2_GUI: Optional[bool] = None

# # Name counters (WHY: sequential filenames <name><N>.jpg)
# _name_counters_body: Dict[str, int] = {}
# _name_counters_face: Dict[str, int] = {}

# # -------------------- fs/io --------------------
# def _ensure_dir(p: Path) -> None:
#     p.mkdir(parents=True, exist_ok=True)

# def _gallery_body_dir(_user_name: str) -> Path:
#     d = CROPS_ROOT / "gallery"
#     _ensure_dir(d)
#     return d

# def _gallery_face_dir(_user_name: str) -> Path:
#     d = CROPS_ROOT / "gallery_face"
#     _ensure_dir(d)
#     return d

# def _list_images_with_prefix(root: Path, prefix: str) -> List[Path]:
#     if not root.exists():
#         return []
#     files = sorted([p for p in root.glob(f"{prefix}*.jpg")])
#     return files

# def _safe_imwrite(path: Path, img: np.ndarray) -> None:
#     _ensure_dir(path.parent)
#     if not cv2.imwrite(str(path), img):
#         raise RuntimeError(f"imwrite failed: {path}")

# def _next_index(root: Path, prefix: str) -> int:
#     # parse existing <prefix><N>.jpg and return next N
#     max_n = 0
#     pat = re.compile(re.escape(prefix) + r"(\d+)\.jpg$", re.IGNORECASE)
#     for p in root.glob(f"{prefix}*.jpg"):
#         m = pat.search(p.name)
#         if m:
#             try:
#                 n = int(m.group(1))
#                 if n > max_n:
#                     max_n = n
#             except Exception:
#                 pass
#     return max_n + 1

# # -------------------- misc utils --------------------
# def _now_ms() -> int:
#     return int(time.time() * 1000)

# def _clip_box(x1, y1, x2, y2, w, h) -> Tuple[int,int,int,int]:
#     x1 = max(0, min(int(x1), w-1)); y1 = max(0, min(int(y1), h-1))
#     x2 = max(0, min(int(x2), w-1)); y2 = max(0, min(int(y2), h-1))
#     if x2 <= x1: x2 = min(w-1, x1+2)
#     if y2 <= y1: y2 = min(h-1, y1+2)
#     return x1, y1, x2, y2

# def _l2_normalize(v: np.ndarray) -> np.ndarray:
#     v = np.asarray(v, dtype=np.float32).reshape(-1)
#     n = float(np.linalg.norm(v))
#     if n == 0.0 or not np.isfinite(n):
#         return v
#     return v / n

# def _as_512f(x: np.ndarray | List[float]) -> Optional[np.ndarray]:
#     try:
#         a = np.asarray(x, dtype=np.float32).reshape(-1)
#         if a.size != EXPECTED_EMBED_DIM or not np.isfinite(a).all():
#             return None
#         return _l2_normalize(a)
#     except Exception:
#         return None

# # -------------------- progress --------------------
# def _progress_reset(user_id: int, user_name: str) -> None:
#     with _progress_lock:
#         _progress_map[user_id] = {
#             "stage":"idle","percent":0,"message":"",
#             "total_body":0,"total_face":0,"done_body":0,"done_face":0
#         }

# def _progress_set(user_id: int, **kv: int | str) -> None:
#     with _progress_lock:
#         st = _progress_map.setdefault(user_id, {})
#         st.update(kv)

# def _progress_bump(user_id: int, body_inc: int = 0, face_inc: int = 0) -> None:
#     with _progress_lock:
#         st = _progress_map.setdefault(user_id, {})
#         st["done_body"] = int(st.get("done_body", 0)) + body_inc
#         st["done_face"] = int(st.get("done_face", 0)) + face_inc
#         tb, tf = max(0, int(st.get("total_body", 0))), max(0, int(st.get("total_face", 0)))
#         db, df = min(tb, int(st.get("done_body", 0))), min(tf, int(st.get("done_face", 0)))
#         pct = int(round((db+df) * 100.0 / (tb+tf))) if (tb+tf) > 0 else 0
#         st["percent"] = max(0, min(100, pct))

# def get_progress_for_user(user_id: int) -> Dict[str, int | str]:
#     with _progress_lock:
#         return dict(_progress_map.get(user_id, {}))

# # -------------------- model loaders (GPU-first) --------------------
# @dataclass
# class ReIDConfig:
#     model_name: str = "osnet_x0_25"
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"

# def _get_reid_extractor(cfg: ReIDConfig = ReIDConfig()) -> TorchreidExtractor:
#     if not _HAS_TORCHREID:
#         raise RuntimeError("torchreid is not installed; pip install torchreid")
#     global _reid_extractor
#     with _reid_lock:
#         if _reid_extractor is None:
#             dev = "cuda" if torch.cuda.is_available() else "cpu"
#             _reid_extractor = TorchreidExtractor(model_name=cfg.model_name, device=dev)
#             try:
#                 _ = _reid_extractor([np.zeros((256,128,3), dtype=np.uint8)])
#             except Exception:
#                 pass
#         return _reid_extractor

# def _get_yolo() -> YOLO:
#     if not _HAS_ULTRALYTICS:
#         raise RuntimeError("ultralytics is not installed; pip install ultralytics")
#     global _yolo_model
#     with _yolo_lock:
#         if _yolo_model is None:
#             m = YOLO("yolov8n.pt")
#             if torch.cuda.is_available():
#                 try:
#                     m.to("cuda")
#                     m.fuse()  # speed (why: conv+bn)
#                 except Exception:
#                     pass
#             _yolo_model = m
#         return _yolo_model

# def _get_face() -> Optional[FaceAnalysis]:
#     if not _HAS_INSIGHT:
#         return None
#     global _face_app
#     with _face_lock:
#         if _face_app is None:
#             app = FaceAnalysis(name="buffalo_l")
#             ctx_id = 0 if torch.cuda.is_available() else -1  # WHY: GPU if available
#             app.prepare(ctx_id=ctx_id)
#             _face_app = app
#         return _face_app

# # -------------------- OpenCV viewer --------------------
# def _check_cv2_gui() -> bool:
#     try:
#         nm = "__mvision_gui_check__"
#         cv2.namedWindow(nm, cv2.WINDOW_NORMAL)
#         cv2.imshow(nm, np.zeros((1,1,3), dtype=np.uint8))
#         cv2.waitKey(1)
#         cv2.destroyWindow(nm)
#         return True
#     except Exception as e:
#         logger.error("OpenCV GUI not available: %s", e)
#         logger.error("Install GUI build: pip uninstall -y opencv-python-headless && pip install opencv-python")
#         return False

# def _viewer_loop() -> None:
#     try:
#         cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
#         ph = np.zeros((480, 640, 3), dtype=np.uint8)
#         cv2.putText(ph, "Capturing... (press q to close)", (10, 460),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
#         cv2.imshow(WINDOW_NAME, ph); cv2.waitKey(1)
#     except Exception:
#         logger.warning("Failed to create OpenCV window (headless?).")
#         return
#     try:
#         while viewer_running.is_set() and not stop_event.is_set():
#             try:
#                 frame = viewer_queue.get(timeout=0.1)
#             except Empty:
#                 if cv2.waitKey(1) & 0xFF == ord("q"):
#                     viewer_running.clear()
#                 continue
#             try:
#                 cv2.imshow(WINDOW_NAME, frame)
#                 if cv2.waitKey(1) & 0xFF == ord("q"):
#                     viewer_running.clear()
#             except Exception:
#                 viewer_running.clear()
#                 break
#     finally:
#         try:
#             cv2.destroyWindow(WINDOW_NAME)
#         except Exception:
#             pass

# def _start_viewer_if_needed() -> None:
#     global viewer_thread, _HAS_CV2_GUI
#     if not SHOW_IMSHOW:
#         return
#     if _HAS_CV2_GUI is None:
#         _HAS_CV2_GUI = _check_cv2_gui()
#     if not _HAS_CV2_GUI:
#         return
#     if viewer_thread is None or not viewer_thread.is_alive():
#         viewer_running.set()
#         viewer_thread = threading.Thread(target=_viewer_loop, name="imshow_viewer", daemon=True)
#         viewer_thread.start()

# def _stop_viewer() -> None:
#     viewer_running.clear()
#     try:
#         if viewer_thread and viewer_thread.is_alive():
#             viewer_thread.join(timeout=1.0)
#     except Exception:
#         pass
#     try: cv2.destroyAllWindows()
#     except Exception: pass

# def _preview(img_bgr: np.ndarray) -> None:
#     if not SHOW_IMSHOW or not viewer_running.is_set():
#         return
#     try:
#         if viewer_queue.full():
#             _ = viewer_queue.get_nowait()
#         viewer_queue.put_nowait(img_bgr)
#     except Exception:
#         pass

# # -------------------- RTSP helpers --------------------
# def _env_streams() -> List[str]:
#     val = os.getenv("MVISION_RTSP", "").strip()
#     if not val:
#         return []
#     parts = [p.strip() for p in re.split(r"[;,|]", val) if p.strip()]
#     return parts

# def _encode_password_at(url: str) -> str:
#     try:
#         i = url.find("://"); j = url.rfind("@")
#         if i < 0 or j < 0 or j <= i + 3: return url
#         cred = url[i+3:j]
#         if ":" not in cred: return url
#         user, pwd = cred.split(":", 1)
#         if "@" in pwd and "%40" not in pwd:
#             pwd = pwd.replace("@", "%40")
#             return f"{url[:i+3]}{user}:{pwd}@{url[j+1:]}"
#     except Exception:
#         pass
#     return url

# def _open_rtsp(url: str) -> cv2.VideoCapture:
#     cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
#     if cap.isOpened():
#         return cap
#     fixed = _encode_password_at(url)
#     if fixed != url:
#         cap2 = cv2.VideoCapture(fixed, cv2.CAP_FFMPEG)
#         if cap2.isOpened():
#             logger.info("RTSP opened after encoding @ in password.")
#             return cap2
#     raise RuntimeError("Failed to open RTSP. Check credentials/network or install FFmpeg-enabled OpenCV.")

# # -------------------- capture & crop --------------------
# def _annotate(frame: np.ndarray, box: Tuple[int,int,int,int], color=(0,255,0), text: str="") -> None:
#     x1,y1,x2,y2 = box
#     cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
#     if text:
#         cv2.putText(frame, text, (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# def _save_person_crop(frame_bgr: np.ndarray, box: Tuple[int,int,int,int], out_dir: Path, name: str, counter_map: Dict[str,int]) -> Path:
#     # WHY: match naming <name><N>.jpg at the folder root
#     idx = counter_map.setdefault(name, _next_index(out_dir, name))
#     h, w = frame_bgr.shape[:2]
#     x1,y1,x2,y2 = _clip_box(*box, w, h)
#     crop = frame_bgr[y1:y2, x1:x2].copy()
#     out_path = out_dir / f"{name}{idx}.jpg"
#     _safe_imwrite(out_path, crop)
#     counter_map[name] = idx + 1
#     return out_path

# def _face_inside_person(face_box: Tuple[int,int,int,int], person_box: Tuple[int,int,int,int]) -> bool:
#     fx1,fy1,fx2,fy2 = face_box
#     px1,py1,px2,py2 = person_box
#     cx = (fx1 + fx2) / 2.0
#     cy = (fy1 + fy2) / 2.0
#     return (px1 <= cx <= px2) and (py1 <= cy <= py2)

# def _capture_loop(cam_idx: int, url: str, user_id: int, user_name: str) -> None:
#     """Capture frames, detect person & (optionally) face, save crops, preview live. GPU-first."""
#     try:
#         yolo = _get_yolo()
#     except Exception as e:
#         logger.error("YOLO init failed: %s", e)
#         return
#     face_app = _get_face()  # optional, on GPU if available

#     body_dir = _gallery_body_dir(user_name)
#     face_dir = _gallery_face_dir(user_name)

#     # seed counters at start (support restart)
#     _name_counters_body.setdefault(user_name, _next_index(body_dir, user_name))
#     _name_counters_face.setdefault(user_name, _next_index(face_dir, user_name))

#     try:
#         cap = _open_rtsp(url)
#     except Exception as e:
#         logger.error("Open RTSP failed [%s]: %s", url, e)
#         return

#     # viewer
#     if SHOW_IMSHOW:
#         _start_viewer_if_needed()

#     saved_body = 0
#     saved_face = 0
#     frame_idx = 0
#     device_param = 0 if torch.cuda.is_available() else None
#     use_half = bool(torch.cuda.is_available())

#     logger.info("Capture started on %s (GPU=%s, FP16=%s)", url, torch.cuda.is_available(), use_half)

#     while not stop_event.is_set() and is_running:
#         ok, frame = cap.read()
#         if not ok or frame is None:
#             time.sleep(0.02)
#             continue
#         frame_idx += 1

#         # YOLO person detection on GPU
#         try:
#             results = yolo.predict(
#                 source=frame, imgsz=640, conf=YOLO_CONF,
#                 classes=[PERSON_CLASS_ID], device=device_param, half=use_half, verbose=False
#             )
#             person_boxes: List[Tuple[int,int,int,int]] = []
#             person_scores: List[float] = []
#             for res in results:
#                 if not hasattr(res, "boxes") or res.boxes is None:
#                     continue
#                 boxes = res.boxes.xyxy  # tensor
#                 confs = res.boxes.conf
#                 if boxes is None:
#                     continue
#                 boxes = boxes.to("cpu").numpy().astype(int)
#                 confs = confs.to("cpu").numpy() if confs is not None else []
#                 for i, (x1,y1,x2,y2) in enumerate(boxes):
#                     person_boxes.append((x1,y1,x2,y2))
#                     person_scores.append(float(confs[i]) if i < len(confs) else 0.0)
#         except Exception:
#             person_boxes, person_scores = [], []

#         # Save body crops
#         for i, box in enumerate(person_boxes):
#             if saved_body >= CAP_MAX_BODY:
#                 break
#             _annotate(frame, box, (0,255,0), f"person {person_scores[i]:.2f}")
#             _ = _save_person_crop(frame, box, body_dir, user_name, _name_counters_body)
#             saved_body += 1

#         # For "face" set: save the SAME full body crop only if a face exists inside the person box
#         if face_app is not None and saved_face < CAP_MAX_FACE and person_boxes:
#             try:
#                 faces = face_app.get(np.ascontiguousarray(frame))
#                 face_candidates: List[Tuple[Tuple[int,int,int,int], float]] = []
#                 for f in faces or []:
#                     bbox = getattr(f, "bbox", None)
#                     score = float(getattr(f, "det_score", 1.0))
#                     if bbox is None: 
#                         continue
#                     x1,y1,x2,y2 = [int(v) for v in bbox[:4]]
#                     face_candidates.append(((x1,y1,x2,y2), score))
#                 for pbox in person_boxes:
#                     # choose a face inside this person box with good score
#                     has_face = any(
#                         _face_inside_person(fb, pbox) and sc >= FACE_MIN_SCORE
#                         for fb, sc in face_candidates
#                     )
#                     if has_face:
#                         _annotate(frame, pbox, (255,0,0), "face-ok")
#                         _ = _save_person_crop(frame, pbox, face_dir, user_name, _name_counters_face)
#                         saved_face += 1
#                         if saved_face >= CAP_MAX_FACE:
#                             break
#             except Exception:
#                 pass

#         _preview(frame)

#         if CAP_FPS_THROTTLE > 0:
#             time.sleep(CAP_FPS_THROTTLE)
#         if saved_body >= CAP_MAX_BODY and saved_face >= CAP_MAX_FACE:
#             break

#     try: cap.release()
#     except Exception: pass
#     if SHOW_IMSHOW and AUTO_CLOSE_VIEWER:
#         _stop_viewer()
#     logger.info("Capture done (body=%d, face(full-body)=%d)", saved_body, saved_face)

# # -------------------- embedding from folders --------------------
# def _embed_dir_with_osnet(dir_path: Path, user_prefix: str, on_step: Optional[Callable[[int], None]] = None) -> Tuple[Optional[np.ndarray], int]:
#     files = _list_images_with_prefix(dir_path, user_prefix)
#     if not files:
#         return None, 0
#     extractor = _get_reid_extractor()  # device='cuda' when available
#     vecs: List[np.ndarray] = []
#     batch: List[np.ndarray] = []

#     def flush_batch():
#         nonlocal batch, vecs
#         if not batch:
#             return
#         try:
#             with torch.inference_mode():
#                 feats = extractor(batch)  # 512-D per image, computed on GPU if device='cuda'
#             for f in feats:
#                 f512 = _as_512f(f)
#                 if f512 is not None:
#                     vecs.append(f512)
#         except Exception:
#             pass
#         if on_step:
#             on_step(len(batch))
#         batch = []

#     for p in files:
#         img = cv2.imread(str(p), cv2.IMREAD_COLOR)
#         if img is None or img.size == 0:
#             if on_step: on_step(1); continue
#         _preview(img)
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         batch.append(rgb)
#         if len(batch) >= BATCH_SIZE:
#             flush_batch()
#     flush_batch()

#     if not vecs:
#         return None, len(files)

#     mean_vec = _l2_normalize(np.mean(np.stack(vecs, axis=0).astype(np.float32), axis=0))
#     return mean_vec, len(files)

# # -------------------- DB ops --------------------
# def _update_db_embeddings(user_id: int, body: Optional[np.ndarray], face: Optional[np.ndarray]) -> Dict[str, object]:
#     session = get_session()
#     try:
#         user = session.query(User).filter(User.id == int(user_id)).first()
#         if not user:
#             return {"status": "error", "message": f"user id {user_id} not found"}
#         changed = False
#         if body is not None:
#             user.body_embedding = body.tolist()
#             changed = True
#         if face is not None:
#             user.face_embedding = face.tolist()
#             changed = True
#         if changed:
#             session.commit()
#             return {"status": "ok", "body": body is not None, "face": face is not None}
#         return {"status": "no_change"}
#     except Exception as e:
#         session.rollback()
#         logger.exception("DB update failed")
#         return {"status": "error", "message": f"DB error: {e}"}
#     finally:
#         session.close()

# def _clear_db_embeddings(user_id: int) -> Dict[str, object]:
#     session = get_session()
#     try:
#         user = session.query(User).filter(User.id == int(user_id)).first()
#         if not user:
#             return {"status": "error", "message": "user not found"}
#         user.body_embedding = None
#         user.face_embedding = None
#         session.commit()
#         return {"status": "ok", "id": user_id}
#     except Exception as e:
#         session.rollback()
#         logger.exception("DB clear failed")
#         return {"status": "error", "message": f"DB error: {e}"}
#     finally:
#         session.close()

# # -------------------- public API --------------------
# def _load_streams_if_empty() -> None:
#     global RTSP_STREAMS
#     env_streams = _env_streams()
#     if env_streams:
#         RTSP_STREAMS = env_streams

# def start_extraction(user_id: int, user_name: str) -> Dict[str, object]:
#     """Start capture & crop pipeline; preview via imshow. Saves to D:\\mvision_api\\crops\\gallery(_face)\\<name><N>.jpg"""
#     global is_running, current_person_id, current_person_name
#     if is_running:
#         return {"status": "ok", "streams": len(RTSP_STREAMS), "id": current_person_id, "name": current_person_name}

#     _load_streams_if_empty()
#     if not RTSP_STREAMS or not RTSP_STREAMS[0]:
#         return {"status": "error", "message": "No RTSP streams. Set MVISION_RTSP env var."}

#     # warm-up models (GPU)
#     _get_yolo()
#     _get_face()

#     stop_event.clear()
#     current_person_id = int(user_id)
#     current_person_name = str(user_name) if user_name else "unknown"
#     _progress_reset(current_person_id, current_person_name)
#     is_running = True

#     if SHOW_IMSHOW:
#         _start_viewer_if_needed()

#     for idx, url in enumerate(RTSP_STREAMS):
#         t = threading.Thread(target=_capture_loop, args=(idx, url, user_id, current_person_name),
#                              name=f"cap-{idx}", daemon=True)
#         t.start()
#         _capture_threads[idx] = t

#     return {"status": "ok", "streams": len(RTSP_STREAMS), "id": current_person_id, "name": current_person_name}

# def stop_extraction() -> Dict[str, object]:
#     """Stop capture."""
#     global is_running, current_person_id, current_person_name
#     if not is_running:
#         return {"status": "not_running", "message": "extraction is not running"}
#     stop_event.set()
#     for t in list(_capture_threads.values()):
#         try: t.join(timeout=1.0)
#         except Exception: pass
#     _capture_threads.clear()
#     is_running = False
#     rid = current_person_id
#     current_person_id = None
#     current_person_name = None
#     if SHOW_IMSHOW and AUTO_CLOSE_VIEWER:
#         _stop_viewer()
#     return {"status": "ok", "id": rid}

# def extract_embeddings_async(user_id: int) -> Dict[str, object]:
#     """Compute OSNet embeddings from saved crops and store into DB."""
#     session = get_session()
#     try:
#         user = session.query(User).filter(User.id == int(user_id)).first()
#         if not user:
#             return {"status": "error", "message": f"user id {user_id} not found"}
#         user_name = str(getattr(user, "name", "unknown") or "unknown")
#     finally:
#         session.close()

#     _progress_reset(user_id, user_name)
#     _progress_set(user_id, stage="initializing", message="loading models", percent=0)

#     def _worker(uid: int, uname: str):
#         try:
#             _get_reid_extractor()  # CUDA if available
#         except Exception as e:
#             _progress_set(uid, stage="error", message=f"model load failed: {e}", percent=0)
#             logger.exception("Model init failed")
#             return

#         if SHOW_IMSHOW:
#             _start_viewer_if_needed()

#         body_dir = _gallery_body_dir(uname)
#         face_dir = _gallery_face_dir(uname)

#         body_files = _list_images_with_prefix(body_dir, uname)
#         face_files = _list_images_with_prefix(face_dir, uname)
#         _progress_set(uid, stage="scanning", message="counting images",
#                       total_body=len(body_files), total_face=len(face_files),
#                       done_body=0, done_face=0, percent=0)

#         def on_body(n: int): _progress_bump(uid, body_inc=n)
#         def on_face(n: int): _progress_bump(uid, face_inc=n)

#         _progress_set(uid, stage="embedding", message="osnet body")
#         body_vec, _ = _embed_dir_with_osnet(body_dir, uname, on_step=on_body)

#         _progress_set(uid, stage="embedding", message="osnet face")
#         face_vec, _ = _embed_dir_with_osnet(face_dir, uname, on_step=on_face)

#         _progress_set(uid, stage="writing", message="saving to db")
#         _ = _update_db_embeddings(uid, body_vec, face_vec)

#         _progress_set(uid, stage="done", message="completed", percent=100)

#         if SHOW_IMSHOW and AUTO_CLOSE_VIEWER:
#             _stop_viewer()

#     t = threading.Thread(target=_worker, name=f"embed-osnet-{user_id}", args=(user_id, user_name), daemon=True)
#     t.start()
#     _extract_threads[user_id] = t
#     return {"status": "started", "id": int(user_id), "name": user_name}

# def get_status() -> Dict[str, object]:
#     return {
#         "running": is_running,
#         "num_cams": len(RTSP_STREAMS),
#         "rtsp_streams": RTSP_STREAMS,
#         "id": current_person_id,
#         "progress": get_progress_for_user(current_person_id) if current_person_id else {},
#     }

# def remove_embeddings(user_id: int) -> Dict[str, object]:
#     return _clear_db_embeddings(int(user_id))

# # -------------------- startup hooks --------------------
# def init_db() -> None:
#     session = get_session()
#     try:
#         bind = session.get_bind()
#         dialect = (bind.dialect.name or "").lower() if bind is not None else ""
#         session.execute(text("SELECT 1")); session.commit()
#         if dialect != "postgresql":
#             logger.info("DB dialect '%s' detected; skipping pgvector init.", dialect)
#             return
#         try:
#             session.execute(text("CREATE EXTENSION IF NOT EXISTS vector")); session.commit()
#         except Exception:
#             session.rollback(); logger.exception("pgvector extension ensure failed")
#         try:
#             session.execute(text("""
#                 CREATE INDEX IF NOT EXISTS ix_users_body_embedding_ivfflat
#                 ON users USING ivfflat (body_embedding vector_cosine_ops) WITH (lists = 100);
#             """))
#             session.execute(text("""
#                 CREATE INDEX IF NOT EXISTS ix_users_face_embedding_ivfflat
#                 ON users USING ivfflat (face_embedding vector_cosine_ops) WITH (lists = 100);
#             """))
#             session.commit()
#             session.execute(text("ANALYZE users;")); session.commit()
#         except Exception:
#             session.rollback(); logger.exception("ANN index creation failed")
#         logger.info("DB init complete (pgvector ready).")
#     finally:
#         session.close()

# def init_models() -> None:
#     global _HAS_CV2_GUI
#     _HAS_CV2_GUI = _check_cv2_gui() if SHOW_IMSHOW else False
#     try:
#         m = _get_yolo()
#         if torch.cuda.is_available():
#             try: m.to("cuda")
#             except Exception: pass
#         logger.info("YOLO ready (cuda=%s)", torch.cuda.is_available())
#     except Exception as e:
#         logger.warning("YOLO unavailable: %s", e)
#     try:
#         app = _get_face()
#         logger.info("InsightFace ready (gpu=%s)", app is not None and torch.cuda.is_available())
#     except Exception as e:
#         logger.warning("InsightFace unavailable: %s", e)
#     try:
#         _get_reid_extractor()
#         logger.info("TorchReID ready (cuda=%s)", torch.cuda.is_available())
#     except Exception as e:
#         logger.warning("TorchReID unavailable: %s", e)

# def ensure_csv_header() -> None:
#     try:
#         p = CROPS_ROOT / "gallery" / "crops.csv"
#         if not p.exists():
#             _ensure_dir(p.parent)
#             p.write_text(",".join([
#                 "person_id","person_name","ts","cam_idx","frame_idx","det_idx",
#                 "x1","y1","x2","y2","score","emb_path","emb_dim","image_path","type"
#             ]) + "\n", encoding="utf-8")
#     except Exception:
#         logger.warning("ensure_csv_header failed; continuing.")