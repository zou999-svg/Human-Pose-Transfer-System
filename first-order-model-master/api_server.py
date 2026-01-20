import os
import uuid
import shutil
import asyncio
import yaml
import imageio
import numpy as np
import torch

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from skimage.transform import resize
from skimage import img_as_ubyte

# 复用仓库函数（不再 subprocess）
from demo import load_checkpoints, make_animation


# ---------------------------
# 1) 模型配置
# ---------------------------
MODEL_CONFIG = {
    "vox": {
        "label": "人脸模型 (vox)",
        "config_path": "config/vox-256.yaml",
        "checkpoint_path": "checkpoints/vox-cpk.pth.tar"
    },
    "vox-adv": {
        "label": "人脸模型-高清 (vox-adv)",
        "config_path": "config/vox-adv-256.yaml",
        "checkpoint_path": "checkpoints/vox-adv-cpk.pth.tar"
    },
    "taichi": {
        "label": "全身动作 (taichi)",
        "config_path": "config/taichi-256.yaml",
        "checkpoint_path": "checkpoints/taichi-cpk.pth.tar"
    },
    "fashion": {
        "label": "时尚模型 (fashion)",
        "config_path": "config/fashion-256.yaml",
        "checkpoint_path": "checkpoints/fashion-cpk.pth.tar"
    },
    "mgif": {
        "label": "动画模型 (mgif)",
        "config_path": "config/mgif-256.yaml",
        "checkpoint_path": "checkpoints/mgif-cpk.pth.tar"
    }
}

TEMP_ROOT = "api_temp"
UPLOAD_DIR = os.path.join(TEMP_ROOT, "uploads")
RESULT_DIR = os.path.join(TEMP_ROOT, "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ---------------------------
# 2) 全局缓存：模型只加载一次
# ---------------------------
_MODEL_CACHE = {
    "model_key": None,
    "generator": None,
    "kp_detector": None,
    "frame_shape": (256, 256),
    "cpu": False,
}

# 并发限制：GPU 推理建议 1（避免显存/速度崩）
_SEM = asyncio.Semaphore(1)


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
        raise HTTPException(500, f"Config not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise HTTPException(500, f"Checkpoint not found: {checkpoint_path}")

    frame_shape = _load_yaml_frame_shape(config_path)
    generator, kp_detector = load_checkpoints(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        cpu=cpu
    )

    _MODEL_CACHE.update({
        "model_key": model_key,
        "generator": generator,
        "kp_detector": kp_detector,
        "frame_shape": frame_shape,
        "cpu": cpu,
    })


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
        raise HTTPException(400, "Driving video has no frames / cannot be read.")
    return frames, fps


def _generate_video_sync(
    source_image_path: str,
    driving_video_path: str,
    model_key: str,
    relative: bool,
    adapt_scale: bool,
    use_cpu: bool
) -> str:
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
        cpu=cpu
    )

    out_path = os.path.join(RESULT_DIR, f"result_{uuid.uuid4().hex}.mp4")
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
    try:
        for frame in preds:
            writer.append_data(img_as_ubyte(frame))
    finally:
        writer.close()

    return out_path


def _cleanup_files(*paths: str):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


app = FastAPI(title="First Order Motion Model API", version="1.0.0")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/models")
def models():
    return {
        "models": [
            {"key": k, "label": v["label"]}
            for k, v in MODEL_CONFIG.items()
        ]
    }


@app.post("/animate")
async def animate(
    background_tasks: BackgroundTasks,
    source_image: UploadFile = File(..., description="source image file (png/jpg)"),
    driving_video: UploadFile = File(..., description="driving video file (mp4/gif)"),
    model_key: str = Form("vox"),
    relative: bool = Form(True),
    adapt_scale: bool = Form(True),
    use_cpu: bool = Form(False),
):
    if model_key not in MODEL_CONFIG:
        raise HTTPException(400, f"Unknown model_key: {model_key}. Use GET /models")

    # 保存上传文件到本地临时目录
    uid = uuid.uuid4().hex
    src_path = os.path.join(UPLOAD_DIR, f"src_{uid}_{source_image.filename}")
    drv_path = os.path.join(UPLOAD_DIR, f"drv_{uid}_{driving_video.filename}")

    try:
        with open(src_path, "wb") as f:
            shutil.copyfileobj(source_image.file, f)
        with open(drv_path, "wb") as f:
            shutil.copyfileobj(driving_video.file, f)
    finally:
        await source_image.close()
        await driving_video.close()

    # 并发限制：避免多请求同时占满显存
    async with _SEM:
        try:
            out_path = await run_in_threadpool(
                _generate_video_sync,
                src_path, drv_path, model_key, relative, adapt_scale, use_cpu
            )
        except HTTPException:
            background_tasks.add_task(_cleanup_files, src_path, drv_path)
            raise
        except Exception as e:
            background_tasks.add_task(_cleanup_files, src_path, drv_path)
            raise HTTPException(500, f"Inference failed: {repr(e)}")

    # 返回文件，同时后台清理临时上传文件（结果可选择保留或也清理）
    background_tasks.add_task(_cleanup_files, src_path, drv_path)
    # 如果你希望结果也自动删：background_tasks.add_task(_cleanup_files, out_path)

    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename="result.mp4"
    )
