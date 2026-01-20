import os, tempfile
import gradio as gr

# 直接调用 demo.py 的 CLI（最省心，少改代码）
import subprocess

CONFIG = "config/vox-256.yaml"
CKPT = "checkpoints/vox-cpk.pth.tar"

def run_fom(source_image, driving_video):
    out = os.path.join(tempfile.mkdtemp(), "result.mp4")
    cmd = [
        "python", "demo.py",
        "--config", CONFIG,
        "--driving_video", driving_video,
        "--source_image", source_image,
        "--checkpoint", CKPT,
        "--relative", "--adapt_scale",
        "--result_video", out
    ]
    subprocess.check_call(cmd)
    return out

demo = gr.Interface(
    fn=run_fom,
    inputs=[
        gr.Image(type="filepath", label="Source Image (png/jpg)"),
        gr.Video(label="Driving Video (mp4)")
    ],
    outputs=gr.Video(label="Result"),
    title="First Order Motion Model Demo"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
