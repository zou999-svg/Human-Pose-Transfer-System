# server.py
import os
import io
import json
import uuid
import base64
import shutil
import asyncio
import datetime
import mimetypes
from typing import Optional, List

import yaml
import imageio
import numpy as np
import torch

from PIL import Image

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
    BackgroundTasks,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from starlette.concurrency import run_in_threadpool

from skimage.transform import resize
from skimage import img_as_ubyte

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Text,
    Boolean,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from passlib.context import CryptContext
from jose import jwt, JWTError

# 复用仓库函数（不再 subprocess）
from demo import load_checkpoints, make_animation

# =========================
# 0) 路径与基础配置
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

APP_NAME = "FOMM Mobile Backend"
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_ME_TO_A_LONG_RANDOM_SECRET")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))

# 文件存储根目录（本地）
STORAGE_ROOT = os.getenv("STORAGE_ROOT", os.path.join(BASE_DIR, "storage"))

# 推理最大并发（GPU 建议 1）
MAX_INFER_CONCURRENCY = int(os.getenv("MAX_INFER_CONCURRENCY", "1"))

os.makedirs(STORAGE_ROOT, exist_ok=True)

# =========================
# 1) 模型配置（扩展到 7 个）
# =========================
MODEL_CONFIG = {
    "vox": {
        "label": "人脸/说话头 (vox)",
        "config_path": os.path.join(CONFIG_DIR, "vox-256.yaml"),
        "checkpoint_path": os.path.join(CKPT_DIR, "vox-cpk.pth.tar"),
    },
    "vox-adv": {
        "label": "人脸/说话头-高清 (vox-adv)",
        "config_path": os.path.join(CONFIG_DIR, "vox-adv-256.yaml"),
        "checkpoint_path": os.path.join(CKPT_DIR, "vox-adv-cpk.pth.tar"),
    },
    "taichi": {
        "label": "全身动作 (taichi)",
        "config_path": os.path.join(CONFIG_DIR, "taichi-256.yaml"),
        "checkpoint_path": os.path.join(CKPT_DIR, "taichi-cpk.pth.tar"),
    },
    "taichi-adv": {
        "label": "全身动作-高清 (taichi-adv)",
        "config_path": os.path.join(CONFIG_DIR, "taichi-adv-256.yaml"),
        "checkpoint_path": os.path.join(CKPT_DIR, "taichi-adv-cpk.pth.tar"),
    },
    "fashion": {
        "label": "时尚/服装 (fashion)",
        "config_path": os.path.join(CONFIG_DIR, "fashion-256.yaml"),
        "checkpoint_path": os.path.join(CKPT_DIR, "fashion.pth.tar"),  # 若你实际文件名不同，请改这里
    },
    "mgif": {
        "label": "GIF/短视频 (mgif)",
        "config_path": os.path.join(CONFIG_DIR, "mgif-256.yaml"),
        "checkpoint_path": os.path.join(CKPT_DIR, "mgif-cpk.pth.tar"),
    },
    "bair": {
        "label": "机器人手臂 (bair)",
        "config_path": os.path.join(CONFIG_DIR, "bair-256.yaml"),
        "checkpoint_path": os.path.join(CKPT_DIR, "bair-cpk.pth.tar"),
    },

}

# =========================
# 2) DB（SQLAlchemy）
# =========================
Base = declarative_base()
engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)

    nickname = Column(String(64), default="")
    # 存 “头像文件相对路径”（在 STORAGE_ROOT 下），如：users/1/avatar/avatar_xxx.png
    avatar_url = Column(String(255), default="")

    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    animations = relationship("AnimationJob", back_populates="user")


class AnimationJob(Base):
    __tablename__ = "animation_jobs"

    id = Column(String(64), primary_key=True, index=True)  # uuid hex
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    model_key = Column(String(32), nullable=False)
    relative = Column(Boolean, default=True)
    adapt_scale = Column(Boolean, default=True)
    use_cpu = Column(Boolean, default=False)

    # 本地文件路径（相对 STORAGE_ROOT）
    source_path = Column(String(255), default="")
    driving_path = Column(String(255), default="")
    result_path = Column(String(255), default="")

    status = Column(String(16), default="queued")  # queued/running/success/failed
    error_message = Column(Text, default="")

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="animations")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =========================
# 3) Auth（JWT + pbkdf2）
# =========================
pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],
    deprecated="auto",
)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def create_access_token(sub: str) -> str:
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": sub, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db=Depends(get_db)) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# =========================
# 4) 推理：模型缓存 + 预处理 + 生成
# =========================
_MODEL_CACHE = {
    "model_key": None,
    "generator": None,
    "kp_detector": None,
    "frame_shape": (256, 256),
    "cpu": False,
}

_INFER_SEM = asyncio.Semaphore(MAX_INFER_CONCURRENCY)


def _load_yaml_frame_shape(config_path: str):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        fs = cfg.get("dataset_params", {}).get("frame_shape", [256, 256, 3])
        return (int(fs[0]), int(fs[1]))
    except Exception:
        return (256, 256)


def _load_model_if_needed(model_key: str, cpu: bool = False):
    global _MODEL_CACHE

    if (
        _MODEL_CACHE["model_key"] == model_key
        and _MODEL_CACHE["generator"] is not None
        and _MODEL_CACHE["cpu"] == cpu
    ):
        return

    # 只缓存一个模型，最稳
    _MODEL_CACHE.update({"model_key": None, "generator": None, "kp_detector": None})
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cfg = MODEL_CONFIG[model_key]
    config_path = cfg["config_path"]
    checkpoint_path = cfg["checkpoint_path"]

    if not os.path.exists(config_path):
        raise RuntimeError(f"Config not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")

    frame_shape = _load_yaml_frame_shape(config_path)
    generator, kp_detector = load_checkpoints(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        cpu=cpu,
    )

    _MODEL_CACHE.update(
        {
            "model_key": model_key,
            "generator": generator,
            "kp_detector": kp_detector,
            "frame_shape": frame_shape,
            "cpu": cpu,
        }
    )


def _read_image_to_np(image_path: str) -> np.ndarray:
    img = imageio.v3.imread(image_path)  # HWC
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] > 3:
        img = img[..., :3]
    return img


def _preprocess_source(source_np, frame_shape):
    img = source_np.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    h, w = frame_shape
    img = resize(img, (h, w), preserve_range=True)[..., :3]
    return img


def _read_and_preprocess_driving(video_path, frame_shape):
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    fps = meta.get("fps", 25)

    frames = []
    h, w = frame_shape
    try:
        for frame in reader:
            f = frame.astype(np.float32)
            if f.max() > 1.5:
                f = f / 255.0
            f = resize(f, (h, w), preserve_range=True)[..., :3]
            frames.append(f)
    finally:
        reader.close()

    if not frames:
        raise RuntimeError("Driving video has no frames / cannot be read.")
    return frames, fps


def _generate_video_sync(
    source_image_path: str,
    driving_video_path: str,
    model_key: str,
    relative: bool,
    adapt_scale: bool,
    use_cpu: bool,
    out_path: str,
) -> None:
    _load_model_if_needed(model_key, cpu=use_cpu)
    gen = _MODEL_CACHE["generator"]
    kp = _MODEL_CACHE["kp_detector"]
    frame_shape = _MODEL_CACHE["frame_shape"]
    cpu = _MODEL_CACHE["cpu"]

    source_np = _read_image_to_np(source_image_path)
    source = _preprocess_source(source_np, frame_shape)
    driving, fps = _read_and_preprocess_driving(driving_video_path, frame_shape)

    preds = make_animation(
        source_image=source,
        driving_video=driving,
        generator=gen,
        kp_detector=kp,
        relative=relative,
        adapt_movement_scale=adapt_scale,
        cpu=cpu,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
    try:
        for frame in preds:
            writer.append_data(img_as_ubyte(frame))
    finally:
        writer.close()


# =========================
# 5) 异步任务队列（不依赖 Redis）
# =========================
_job_queue: "asyncio.Queue[str]" = asyncio.Queue()
_worker_task: Optional[asyncio.Task] = None


async def _worker_loop():
    """
    单机单进程队列 worker：
    - 从队列拿 job_id
    - DB 标记 running
    - 推理生成 result
    - DB 标记 success/failed
    """
    while True:
        job_id = await _job_queue.get()
        db = SessionLocal()
        try:
            job: AnimationJob = db.query(AnimationJob).filter(AnimationJob.id == job_id).first()
            if not job:
                continue

            job.status = "running"
            job.started_at = datetime.datetime.utcnow()
            db.commit()

            async with _INFER_SEM:
                abs_source = os.path.join(STORAGE_ROOT, job.source_path)
                abs_driving = os.path.join(STORAGE_ROOT, job.driving_path)
                abs_result = os.path.join(STORAGE_ROOT, job.result_path)

                try:
                    await run_in_threadpool(
                        _generate_video_sync,
                        abs_source,
                        abs_driving,
                        job.model_key,
                        bool(job.relative),
                        bool(job.adapt_scale),
                        bool(job.use_cpu),
                        abs_result,
                    )
                except Exception as e:
                    job.status = "failed"
                    job.error_message = repr(e)
                    job.finished_at = datetime.datetime.utcnow()
                    db.commit()
                    continue

            job.status = "success"
            job.finished_at = datetime.datetime.utcnow()
            db.commit()

        finally:
            db.close()
            _job_queue.task_done()


# =========================
# 6) FastAPI + Schemas
# =========================
app = FastAPI(title=APP_NAME, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RegisterReq(BaseModel):
    username: str
    password: str


class LoginReq(BaseModel):
    username: str
    password: str


class TokenResp(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MeResp(BaseModel):
    id: int
    username: str
    nickname: str
    avatar_url: str
    created_at: datetime.datetime


class CreateJobResp(BaseModel):
    job_id: str
    status: str
    detail_url: str


class JobItem(BaseModel):
    id: str
    status: str
    model_key: str
    created_at: datetime.datetime
    started_at: Optional[datetime.datetime]
    finished_at: Optional[datetime.datetime]


class JobDetail(BaseModel):
    id: str
    status: str
    model_key: str
    relative: bool
    adapt_scale: bool
    use_cpu: bool
    created_at: datetime.datetime
    started_at: Optional[datetime.datetime]
    finished_at: Optional[datetime.datetime]
    error_message: str
    source_url: str
    driving_url: str
    result_url: str


def _user_dir(user_id: int) -> str:
    return os.path.join("users", str(user_id))


def _safe_relpath(path: str) -> str:
    p = os.path.normpath(path).replace("\\", "/")
    if p.startswith("../") or p.startswith("..\\") or p.startswith("/"):
        raise HTTPException(400, "Invalid path")
    return p


def _public_job_url(job_id: str) -> str:
    return f"/animations/{job_id}"


def _public_file_url(job_id: str, kind: str) -> str:
    return f"/animations/{job_id}/file/{kind}"


def _me_avatar_url(u: User) -> str:
    return "/me/avatar" if (u.avatar_url or "").strip() else ""


def _guess_media_type(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"


@app.on_event("startup")
async def on_startup():
    init_db()
    global _worker_task
    if _worker_task is None:
        _worker_task = asyncio.create_task(_worker_loop())


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/models")
def models():
    out = []
    for k, v in MODEL_CONFIG.items():
        cfg_ok = os.path.exists(v["config_path"])
        ckpt_ok = os.path.exists(v["checkpoint_path"])
        frame_shape = _load_yaml_frame_shape(v["config_path"]) if cfg_ok else (256, 256)
        out.append(
            {
                "key": k,
                "label": v["label"],
                "available": bool(cfg_ok and ckpt_ok),
                "frame_shape": [int(frame_shape[0]), int(frame_shape[1])],
            }
        )
    return {"models": out}


# =========================
# 7) Auth APIs
# =========================
@app.post("/auth/register", response_model=MeResp)
def register(req: RegisterReq, db=Depends(get_db)):
    req.username = req.username.strip()
    if len(req.username) < 3:
        raise HTTPException(400, "username too short")
    if len(req.password) < 6:
        raise HTTPException(400, "password too short")

    exists = db.query(User).filter(User.username == req.username).first()
    if exists:
        raise HTTPException(409, "username already exists")

    u = User(
        username=req.username,
        password_hash=hash_password(req.password),
        nickname=req.username,
        avatar_url="",
    )
    db.add(u)
    db.commit()
    db.refresh(u)

    return MeResp(
        id=u.id,
        username=u.username,
        nickname=u.nickname or "",
        avatar_url=_me_avatar_url(u),
        created_at=u.created_at,
    )


@app.post("/auth/login", response_model=TokenResp)
def login(req: LoginReq, db=Depends(get_db)):
    user = db.query(User).filter(User.username == req.username.strip()).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(401, "invalid username or password")

    token = create_access_token(sub=user.username)
    return TokenResp(access_token=token)


@app.get("/me", response_model=MeResp)
def me(user: User = Depends(get_current_user)):
    return MeResp(
        id=user.id,
        username=user.username,
        nickname=user.nickname or "",
        avatar_url=_me_avatar_url(user),
        created_at=user.created_at,
    )


@app.patch("/me", response_model=MeResp)
def update_me(
    nickname: Optional[str] = Form(None),
    db=Depends(get_db),
    user: User = Depends(get_current_user),
):
    u = db.query(User).filter(User.id == user.id).first()
    if not u:
        raise HTTPException(404, "user not found")

    if nickname is not None:
        u.nickname = nickname[:64]

    db.commit()
    db.refresh(u)

    return MeResp(
        id=u.id,
        username=u.username,
        nickname=u.nickname or "",
        avatar_url=_me_avatar_url(u),
        created_at=u.created_at,
    )


# =========================
# 7.5) Avatar APIs（本地上传头像）
# =========================
@app.post("/me/avatar")
async def upload_my_avatar(
    avatar: UploadFile = File(..., description="avatar image file"),
    db=Depends(get_db),
    user: User = Depends(get_current_user),
):
    ctype = (avatar.content_type or "").lower()
    if not ctype.startswith("image/"):
        raise HTTPException(400, "Only image file is allowed")

    content = await avatar.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(400, "Image too large (>5MB)")

    ext = os.path.splitext(avatar.filename or "")[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
        if "jpeg" in ctype or "jpg" in ctype:
            ext = ".jpg"
        elif "webp" in ctype:
            ext = ".webp"
        else:
            ext = ".png"

    base_dir = _user_dir(user.id)
    avatar_dir = os.path.join(base_dir, "avatar")
    os.makedirs(os.path.join(STORAGE_ROOT, avatar_dir), exist_ok=True)

    filename = f"avatar_{uuid.uuid4().hex}{ext}"
    rel_path = _safe_relpath(os.path.join(avatar_dir, filename))
    abs_path = os.path.join(STORAGE_ROOT, rel_path)

    old_rel = (user.avatar_url or "").strip()
    if old_rel:
        try:
            old_abs = os.path.join(STORAGE_ROOT, _safe_relpath(old_rel))
            if os.path.exists(old_abs):
                os.remove(old_abs)
        except Exception:
            pass

    with open(abs_path, "wb") as f:
        f.write(content)

    u = db.query(User).filter(User.id == user.id).first()
    if not u:
        raise HTTPException(404, "user not found")
    u.avatar_url = rel_path
    db.commit()

    return {"ok": True, "avatar_url": "/me/avatar"}


@app.get("/me/avatar")
def get_my_avatar(user: User = Depends(get_current_user)):
    rel = (user.avatar_url or "").strip()
    if not rel:
        raise HTTPException(404, "No avatar")

    abs_path = os.path.join(STORAGE_ROOT, _safe_relpath(rel))
    if not os.path.exists(abs_path):
        raise HTTPException(404, "Avatar file not found")

    media_type = _guess_media_type(abs_path)
    return FileResponse(
        abs_path,
        media_type=media_type,
        filename=os.path.basename(abs_path),
        headers={"Cache-Control": "no-store"},
    )


# =========================
# 8) Animations APIs（创建任务 / 查询 / 列表 / 下载文件）
# =========================
@app.post("/animations", response_model=CreateJobResp)
async def create_animation(
    source_image: UploadFile = File(..., description="source image file (png/jpg)"),
    driving_video: UploadFile = File(..., description="driving video file (mp4/gif)"),
    model_key: str = Form("vox"),
    relative: bool = Form(True),
    adapt_scale: bool = Form(True),
    use_cpu: bool = Form(False),
    db=Depends(get_db),
    user: User = Depends(get_current_user),
):
    if model_key not in MODEL_CONFIG:
        raise HTTPException(400, f"Unknown model_key: {model_key}. Use GET /models")

    job_id = uuid.uuid4().hex

    base_dir = _user_dir(user.id)
    uploads_dir = os.path.join(base_dir, "uploads")
    results_dir = os.path.join(base_dir, "results")

    os.makedirs(os.path.join(STORAGE_ROOT, uploads_dir), exist_ok=True)
    os.makedirs(os.path.join(STORAGE_ROOT, results_dir), exist_ok=True)

    src_name = f"src_{job_id}_{os.path.basename(source_image.filename)}"
    drv_name = f"drv_{job_id}_{os.path.basename(driving_video.filename)}"
    out_name = f"result_{job_id}.mp4"

    rel_src = _safe_relpath(os.path.join(uploads_dir, src_name))
    rel_drv = _safe_relpath(os.path.join(uploads_dir, drv_name))
    rel_out = _safe_relpath(os.path.join(results_dir, out_name))

    abs_src = os.path.join(STORAGE_ROOT, rel_src)
    abs_drv = os.path.join(STORAGE_ROOT, rel_drv)

    try:
        with open(abs_src, "wb") as f:
            shutil.copyfileobj(source_image.file, f)
        with open(abs_drv, "wb") as f:
            shutil.copyfileobj(driving_video.file, f)
    finally:
        await source_image.close()
        await driving_video.close()

    job = AnimationJob(
        id=job_id,
        user_id=user.id,
        model_key=model_key,
        relative=bool(relative),
        adapt_scale=bool(adapt_scale),
        use_cpu=bool(use_cpu),
        source_path=rel_src,
        driving_path=rel_drv,
        result_path=rel_out,
        status="queued",
    )
    db.add(job)
    db.commit()

    await _job_queue.put(job_id)

    return CreateJobResp(job_id=job_id, status="queued", detail_url=_public_job_url(job_id))


@app.get("/animations", response_model=List[JobItem])
def list_animations(
    page: int = 1,
    page_size: int = 20,
    db=Depends(get_db),
    user: User = Depends(get_current_user),
):
    page = max(page, 1)
    page_size = min(max(page_size, 1), 50)

    q = (
        db.query(AnimationJob)
        .filter(AnimationJob.user_id == user.id)
        .order_by(AnimationJob.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = q.all()
    return [
        JobItem(
            id=j.id,
            status=j.status,
            model_key=j.model_key,
            created_at=j.created_at,
            started_at=j.started_at,
            finished_at=j.finished_at,
        )
        for j in items
    ]


@app.get("/animations/{job_id}", response_model=JobDetail)
def get_animation(job_id: str, db=Depends(get_db), user: User = Depends(get_current_user)):
    job = (
        db.query(AnimationJob)
        .filter(AnimationJob.id == job_id, AnimationJob.user_id == user.id)
        .first()
    )
    if not job:
        raise HTTPException(404, "job not found")

    return JobDetail(
        id=job.id,
        status=job.status,
        model_key=job.model_key,
        relative=bool(job.relative),
        adapt_scale=bool(job.adapt_scale),
        use_cpu=bool(job.use_cpu),
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error_message=job.error_message or "",
        source_url=_public_file_url(job.id, "source"),
        driving_url=_public_file_url(job.id, "driving"),
        result_url=_public_file_url(job.id, "result"),
    )


@app.get("/animations/{job_id}/file/{kind}")
def get_animation_file(
    job_id: str,
    kind: str,
    db=Depends(get_db),
    user: User = Depends(get_current_user),
):
    job = (
        db.query(AnimationJob)
        .filter(AnimationJob.id == job_id, AnimationJob.user_id == user.id)
        .first()
    )
    if not job:
        raise HTTPException(404, "job not found")

    if kind == "source":
        rel = job.source_path
        media_type = "image/*"
        filename = "source"
    elif kind == "driving":
        rel = job.driving_path
        media_type = "video/*"
        filename = "driving"
    elif kind == "result":
        if job.status != "success":
            raise HTTPException(400, f"result not ready, status={job.status}")
        rel = job.result_path
        media_type = "video/mp4"
        filename = "result.mp4"
    else:
        raise HTTPException(400, "kind must be source/driving/result")

    abs_path = os.path.join(STORAGE_ROOT, _safe_relpath(rel))
    if not os.path.exists(abs_path):
        raise HTTPException(404, "file not found on disk")

    return FileResponse(abs_path, media_type=media_type, filename=filename)


@app.delete("/animations/{job_id}")
def delete_animation(job_id: str, db=Depends(get_db), user: User = Depends(get_current_user)):
    job = (
        db.query(AnimationJob)
        .filter(AnimationJob.id == job_id, AnimationJob.user_id == user.id)
        .first()
    )
    if not job:
        raise HTTPException(404, "job not found")

    for rel in [job.source_path, job.driving_path, job.result_path]:
        if rel:
            abs_path = os.path.join(STORAGE_ROOT, _safe_relpath(rel))
            try:
                if os.path.exists(abs_path):
                    os.remove(abs_path)
            except Exception:
                pass

    db.delete(job)
    db.commit()
    return {"ok": True}


# =========================
# 9) Realtime WS（逐帧推理）
# =========================
def _get_normalize_kp():
    # first-order-model 不同版本位置不同，做兼容尝试
    try:
        from animate import normalize_kp
        return normalize_kp
    except Exception:
        pass
    try:
        from modules.util import normalize_kp
        return normalize_kp
    except Exception:
        pass
    return None


NORMALIZE_KP = _get_normalize_kp()


def _np_from_bytes_rgb(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


def _jpg_bytes_from_uint8(img_u8: np.ndarray, quality: int = 80) -> bytes:
    im = Image.fromarray(img_u8, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _to_tensor(img_float_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(img_float_hwc).permute(2, 0, 1).unsqueeze(0).float()
    return t.to(device)


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    y = t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    y = np.clip(y, 0, 1)
    return (y * 255).astype(np.uint8)


def _user_from_token_string(token: str, db):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            return None
    except Exception:
        return None
    return db.query(User).filter(User.username == username).first()


# 强烈建议：限制同时只有 1 个实时会话（避免 GPU 爆）
_REALTIME_SEM = asyncio.Semaphore(1)


@app.websocket("/ws/realtime")
async def ws_realtime(ws: WebSocket):
    await ws.accept()

    if NORMALIZE_KP is None:
        await ws.send_text(
            json.dumps(
                {
                    "ok": False,
                    "error": "normalize_kp not found. 请检查你的 first-order-model 版本，并修正导入位置。",
                }
            )
        )
        await ws.close(code=1011)
        return

    await _REALTIME_SEM.acquire()

    db = SessionLocal()
    try:
        init_text = await ws.receive_text()
        init = json.loads(init_text)

        token = (init.get("token") or "").strip()
        user = _user_from_token_string(token, db)
        if not user:
            await ws.send_text(json.dumps({"ok": False, "error": "unauthorized"}))
            await ws.close(code=1008)
            return

        model_key = init.get("model_key", "vox")
        relative = bool(init.get("relative", True))
        adapt_scale = bool(init.get("adapt_scale", True))
        use_cpu = bool(init.get("use_cpu", False))

        source_b64 = init.get("source_base64", "")
        if not source_b64:
            await ws.send_text(json.dumps({"ok": False, "error": "missing source_base64"}))
            await ws.close(code=1003)
            return

        if "," in source_b64:
            source_b64 = source_b64.split(",", 1)[1]
        source_bytes = base64.b64decode(source_b64)

        if model_key not in MODEL_CONFIG:
            await ws.send_text(json.dumps({"ok": False, "error": f"unknown model_key={model_key}"}))
            await ws.close(code=1003)
            return

        _load_model_if_needed(model_key, cpu=use_cpu)
        gen = _MODEL_CACHE["generator"]
        kp = _MODEL_CACHE["kp_detector"]
        frame_shape = _MODEL_CACHE["frame_shape"]
        cpu = _MODEL_CACHE["cpu"]

        device = torch.device("cpu")
        if torch.cuda.is_available() and not cpu:
            device = torch.device("cuda")

        gen.eval()
        kp.eval()

        src_u8 = _np_from_bytes_rgb(source_bytes)
        src_float = src_u8.astype(np.float32) / 255.0
        h, w = frame_shape
        src_float = resize(src_float, (h, w), preserve_range=True)[..., :3]

        src_t = _to_tensor(src_float, device)

        with torch.no_grad():
            kp_source = kp(src_t)

        kp_driving_initial = None

        await ws.send_text(json.dumps({"ok": True, "msg": "realtime ready", "frame_shape": [h, w]}))

        while True:
            msg = await ws.receive()
            if "bytes" not in msg:
                continue
            frame_bytes = msg["bytes"]

            drv_u8 = _np_from_bytes_rgb(frame_bytes)
            drv_float = drv_u8.astype(np.float32) / 255.0
            drv_float = resize(drv_float, (h, w), preserve_range=True)[..., :3]
            drv_t = _to_tensor(drv_float, device)

            async with _INFER_SEM:
                with torch.no_grad():
                    kp_driving = kp(drv_t)
                    if kp_driving_initial is None:
                        kp_driving_initial = kp_driving

                    kp_norm = NORMALIZE_KP(
                        kp_source=kp_source,
                        kp_driving=kp_driving,
                        kp_driving_initial=kp_driving_initial,
                        use_relative_movement=relative,
                        use_relative_jacobian=relative,
                        adapt_movement_scale=adapt_scale,
                    )

                    out = gen(src_t, kp_source=kp_source, kp_driving=kp_norm)
                    pred = out["prediction"] if isinstance(out, dict) and "prediction" in out else out
                    out_u8 = _tensor_to_uint8(pred)
                    out_jpg = _jpg_bytes_from_uint8(out_u8, quality=80)

            await ws.send_bytes(out_jpg)

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"ok": False, "error": repr(e)}))
        except Exception:
            pass
    finally:
        db.close()
        _REALTIME_SEM.release()
