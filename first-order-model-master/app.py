import os
import uuid
import gradio as gr
import yaml
import imageio
import numpy as np
import torch

from skimage.transform import resize
from skimage import img_as_ubyte

# 直接复用仓库里的函数（不再起子进程）
from demo import load_checkpoints, make_animation


# ---------------------------
# 1) 模型配置
# ---------------------------
MODEL_CONFIG = {
    "人脸模型 (vox)": {
        "config_path": "config/vox-256.yaml",
        "checkpoint_path": "checkpoints/vox-cpk.pth.tar"
    },
    "人脸模型-高清 (vox-adv)": {
        "config_path": "config/vox-adv-256.yaml",
        "checkpoint_path": "checkpoints/vox-adv-cpk.pth.tar"
    },
    "全身动作 (taichi)": {
        "config_path": "config/taichi-256.yaml",
        "checkpoint_path": "checkpoints/taichi-cpk.pth.tar"
    },
    "时尚模型 (fashion)": {
        "config_path": "config/fashion-256.yaml",
        "checkpoint_path": "checkpoints/fashion-cpk.pth.tar"
    },
    "动画模型 (mgif)": {
        "config_path": "config/mgif-256.yaml",
        "checkpoint_path": "checkpoints/mgif-cpk.pth.tar"
    }
}

# ---------------------------
# 2) 全局缓存：避免每次点按钮都重新 load 模型
# ---------------------------
_MODEL_CACHE = {
    "model_name": None,
    "generator": None,
    "kp_detector": None,
    "frame_shape": (256, 256),  # (H, W)
    "cpu": False,
}

def _get_video_path(driving_video):
    """兼容不同 gradio 版本的返回格式：str / dict / tuple"""
    if driving_video is None:
        return None
    if isinstance(driving_video, str):
        return driving_video
    if isinstance(driving_video, dict) and "name" in driving_video:
        return driving_video["name"]
    if isinstance(driving_video, (list, tuple)) and len(driving_video) > 0:
        return driving_video[0]
    return str(driving_video)

def _load_yaml_frame_shape(config_path: str):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        fs = cfg.get("dataset_params", {}).get("frame_shape", [256, 256, 3])
        return (int(fs[0]), int(fs[1]))
    except Exception:
        return (256, 256)

def _load_model_if_needed(model_name: str, cpu: bool = False):
    global _MODEL_CACHE

    if (_MODEL_CACHE["model_name"] == model_name) and (_MODEL_CACHE["generator"] is not None) and (_MODEL_CACHE["cpu"] == cpu):
        return

    # 释放旧模型（只保留一个，显存最稳）
    _MODEL_CACHE["model_name"] = None
    _MODEL_CACHE["generator"] = None
    _MODEL_CACHE["kp_detector"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cfg = MODEL_CONFIG[model_name]
    config_path = cfg["config_path"]
    checkpoint_path = cfg["checkpoint_path"]

    if not os.path.exists(config_path):
        raise gr.Error(f"找不到 config: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise gr.Error(f"找不到 checkpoint: {checkpoint_path}")

    frame_shape = _load_yaml_frame_shape(config_path)

    generator, kp_detector = load_checkpoints(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        cpu=cpu
    )

    _MODEL_CACHE.update({
        "model_name": model_name,
        "generator": generator,
        "kp_detector": kp_detector,
        "frame_shape": frame_shape,
        "cpu": cpu,
    })

def _preprocess_source(source_image_np, frame_shape):
    if source_image_np is None:
        raise gr.Error("请提供源图片!")

    img = source_image_np
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0

    h, w = frame_shape
    img = resize(img, (h, w), preserve_range=True)[..., :3]
    return img

def _read_and_preprocess_driving(video_path, frame_shape):
    if video_path is None or not os.path.exists(video_path):
        raise gr.Error("请提供有效的驱动视频!")

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

    if len(frames) == 0:
        raise gr.Error("驱动视频读取失败（没有帧）。")

    return frames, fps

def generate_video(source_image_np, driving_video, model_name, relative=True, adapt_scale=True, use_cpu=False):
    video_path = _get_video_path(driving_video)

    _load_model_if_needed(model_name, cpu=use_cpu)
    generator = _MODEL_CACHE["generator"]
    kp_detector = _MODEL_CACHE["kp_detector"]
    frame_shape = _MODEL_CACHE["frame_shape"]
    cpu = _MODEL_CACHE["cpu"]

    source = _preprocess_source(source_image_np, frame_shape)
    driving, fps = _read_and_preprocess_driving(video_path, frame_shape)

    predictions = make_animation(
        source_image=source,
        driving_video=driving,
        generator=generator,
        kp_detector=kp_detector,
        relative=relative,
        adapt_movement_scale=adapt_scale,
        cpu=cpu
    )

    temp_dir = "gradio_temp"
    os.makedirs(temp_dir, exist_ok=True)
    out_path = os.path.join(temp_dir, f"result_{uuid.uuid4().hex}.mp4")

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
    try:
        for frame in predictions:
            writer.append_data(img_as_ubyte(frame))
    finally:
        writer.close()

    return out_path


# ---------------------------
# 3) 颜值拉满的 UI（CSS + Hero + 卡片布局）
# ---------------------------
CSS = r"""
/* ===== 背景与整体 ===== */
.gradio-container {
  max-width: 1200px !important;
  margin: 0 auto !important;
}
body {
  background: radial-gradient(1200px 600px at 10% 0%, rgba(99,102,241,.25), transparent 60%),
              radial-gradient(900px 500px at 100% 30%, rgba(16,185,129,.18), transparent 55%),
              radial-gradient(1000px 600px at 50% 120%, rgba(236,72,153,.16), transparent 60%),
              linear-gradient(180deg, rgba(15,23,42,1) 0%, rgba(2,6,23,1) 100%) !important;
}

/* ===== 顶部 Hero ===== */
#hero {
  border-radius: 22px;
  padding: 22px 22px 18px 22px;
  background: linear-gradient(135deg, rgba(255,255,255,.10), rgba(255,255,255,.06));
  border: 1px solid rgba(255,255,255,.14);
  box-shadow: 0 18px 60px rgba(0,0,0,.35);
}
#hero h1 {
  margin: 0 !important;
  font-size: 30px !important;
  letter-spacing: .2px;
}
#hero p {
  margin: 8px 0 0 0 !important;
  opacity: .88;
  line-height: 1.5;
}
.badges {
  margin-top: 14px;
  display: flex; gap: 10px; flex-wrap: wrap;
}
.badge {
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(255,255,255,.10);
  border: 1px solid rgba(255,255,255,.14);
  font-size: 12px;
  opacity: .92;
}

/* ===== 卡片面板 ===== */
.panel-card {
  border-radius: 18px !important;
  background: linear-gradient(180deg, rgba(255,255,255,.09), rgba(255,255,255,.05)) !important;
  border: 1px solid rgba(255,255,255,.14) !important;
  box-shadow: 0 16px 50px rgba(0,0,0,.30) !important;
  padding: 14px 14px 8px 14px !important;
}
.panel-title {
  font-weight: 700;
  font-size: 15px;
  margin: 0 0 10px 0;
  opacity: .95;
}

/* ===== 按钮美化 ===== */
button.primary {
  border-radius: 14px !important;
  font-weight: 700 !important;
  letter-spacing: .2px;
  padding: 12px 14px !important;
  box-shadow: 0 10px 30px rgba(99,102,241,.25) !important;
}
button.secondary {
  border-radius: 14px !important;
  padding: 12px 14px !important;
  background: rgba(255,255,255,.08) !important;
  border: 1px solid rgba(255,255,255,.14) !important;
}

/* ===== 状态条 ===== */
#statusbar {
  border-radius: 14px;
  padding: 10px 12px;
  background: rgba(255,255,255,.08);
  border: 1px solid rgba(255,255,255,.14);
}

/* ===== 让视频/图片更像产品 ===== */
video, img {
  border-radius: 14px !important;
}
"""

def pretty_status(kind: str, text: str):
    icon = {"idle":"🟣", "run":"🟡", "ok":"🟢", "err":"🔴"}.get(kind, "ℹ️")
    return f"<div id='statusbar'>{icon} <b>{text}</b></div>"

with gr.Blocks() as demo:
    gr.HTML(
        """
        <div id="hero">
          <h1>🎭 First Order Motion Model · WebUI</h1>
          <p>上传 <b>源图片</b> + <b>驱动视频</b>，一键生成「会动的照片」。界面做成产品级，推理在进程内完成，不再刷命令行。</p>
          <div class="badges">
            <span class="badge">⚡ 模型缓存加速</span>
            <span class="badge">🧠 进程内推理</span>
            <span class="badge">🎬 MP4 输出</span>
            <span class="badge">🛡️ 单并发更稳</span>
          </div>
        </div>
        """
    )

    with gr.Row(equal_height=True):
        # 左侧：输入区
        with gr.Column(scale=5):
            with gr.Column(elem_classes=["panel-card"]):
                gr.Markdown("### ① 源图片", elem_classes=["panel-title"])
                source_image_input = gr.Image(
                    label="Source Image",
                    type="numpy"
                )
                # 可选：如果你有 assets，就放开
                ex1 = os.path.join(os.getcwd(), "assets/source.png")
                ex2 = os.path.join(os.getcwd(), "assets/source_person.png")
                examples = [p for p in [ex1, ex2] if os.path.exists(p)]
                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=source_image_input,
                        label="示例图片（可直接点击）"
                    )
                gr.Markdown(
                    "- 建议：清晰正脸 / 光照均匀 / 避免遮挡\n"
                    "- 你也可以用人物半身照做表情迁移"
                )

            with gr.Column(elem_classes=["panel-card"]):
                gr.Markdown("### ② 驱动视频", elem_classes=["panel-title"])
                driving_video_input = gr.Video(label="Driving Video")
                v1 = os.path.join(os.getcwd(), "assets/driving.mp4")
                v2 = os.path.join(os.getcwd(), "assets/driving_person.mp4")
                v_examples = [p for p in [v1, v2] if os.path.exists(p)]
                if v_examples:
                    gr.Examples(
                        examples=v_examples,
                        inputs=driving_video_input,
                        label="示例视频（可直接点击）"
                    )
                gr.Markdown("- 建议：10~30 秒、镜头稳定、主体清晰（效果更好）")

            with gr.Column(elem_classes=["panel-card"]):
                gr.Markdown("### ③ 选择模型 & 参数", elem_classes=["panel-title"])

                model_selector = gr.Dropdown(
                    choices=list(MODEL_CONFIG.keys()),
                    value="人脸模型 (vox)",
                    label="预训练模型"
                )

                with gr.Row():
                    relative_ck = gr.Checkbox(value=True, label="relative（推荐）")
                    adapt_ck = gr.Checkbox(value=True, label="adapt_scale（推荐）")

                use_cpu_ck = gr.Checkbox(value=False, label="CPU 模式（仅排错 / 很慢）")

                with gr.Row():
                    submit_btn = gr.Button("🚀 开始生成", variant="primary", elem_classes=["primary"])
                    clear_btn = gr.Button("🔄 清空", variant="secondary", elem_classes=["secondary"])

        # 右侧：输出区
        with gr.Column(scale=7):
            with gr.Column(elem_classes=["panel-card"]):
                gr.Markdown("### ④ 结果预览", elem_classes=["panel-title"])
                status = gr.HTML(pretty_status("idle", "待命：请上传素材后点击开始生成"))
                result_video = gr.Video(label="Result", interactive=False)
                gr.Markdown(
                    "小提示：如果口型/动作不自然，试试换一个驱动视频，或换用更匹配的模型（比如全身用 taichi）。"
                )

    # ---------------------------
    # 事件：生成 / 清空
    # ---------------------------
    def on_submit(source_image, driving_video, model_name, relative, adapt_scale, use_cpu):
        # UI：开始时禁用按钮 + 状态提示
        yield (
            gr.update(value=None),
            gr.update(value=pretty_status("run", "处理中：模型推理中…（请不要重复点击）")),
            gr.update(interactive=False)
        )

        try:
            out_path = generate_video(source_image, driving_video, model_name, relative, adapt_scale, use_cpu)
            yield (
                gr.update(value=out_path),
                gr.update(value=pretty_status("ok", "完成：已生成视频 ✅")),
                gr.update(interactive=True)
            )
        except gr.Error as e:
            yield (
                gr.update(value=None),
                gr.update(value=pretty_status("err", f"失败：{str(e)}")),
                gr.update(interactive=True)
            )
        except Exception as e:
            yield (
                gr.update(value=None),
                gr.update(value=pretty_status("err", f"未知错误：{repr(e)}")),
                gr.update(interactive=True)
            )

    def on_clear():
        return None, pretty_status("idle", "已清空：重新上传素材再生成"), gr.update(interactive=True)

    submit_btn.click(
        fn=on_submit,
        inputs=[source_image_input, driving_video_input, model_selector, relative_ck, adapt_ck, use_cpu_ck],
        outputs=[result_video, status, submit_btn]
    )
    clear_btn.click(
        fn=on_clear,
        inputs=[],
        outputs=[result_video, status, submit_btn]
    )


if __name__ == "__main__":
    # 单并发更稳（避免显存被多请求打爆）；老版本不支持就忽略
    try:
        demo.queue(concurrency_count=1, max_size=12)
    except Exception:
        pass

    try:
        demo.launch(
            server_name="0.0.0.0",
            theme=gr.themes.Soft(),
            css=CSS
        )
    except Exception:
        demo.launch(
            server_name="0.0.0.0",
            css=CSS
        )
