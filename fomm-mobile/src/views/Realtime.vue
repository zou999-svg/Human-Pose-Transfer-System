<template>
  <div class="page">
    <van-nav-bar title="实时摄像头驱动" left-text="返回" left-arrow @click-left="$router.back()" />

    <div class="card">
      <div class="h">1) 选择源图片（要动的那张）</div>

      <input class="file" type="file" accept="image/*" @change="onPickSource" />
      <div v-if="sourcePreview" class="row">
        <img :src="sourcePreview" class="img" />
      </div>

      <div class="h">2) 模型 & 参数</div>

      <van-field label="模型" v-model="modelKey" placeholder="vox" />
      <van-switch v-model="relative" size="20px">relative</van-switch>
      <van-switch v-model="adaptScale" size="20px">adapt_scale</van-switch>
      <van-switch v-model="useCpu" size="20px">CPU(很慢)</van-switch>

      <div class="h">3) 摄像头</div>

      <van-button type="primary" block :disabled="cameraOn" @click="openCamera">打开摄像头</van-button>
      <van-button type="warning" block :disabled="!cameraOn || running" @click="startRealtime">开始实时</van-button>
      <van-button type="danger" block :disabled="!running" @click="stopRealtime">停止实时</van-button>
      <van-button block :disabled="!cameraOn" @click="closeCamera">关闭摄像头</van-button>

      <div class="small" v-if="status">{{ status }}</div>

      <div class="row">
        <video ref="videoEl" class="video" autoplay playsinline muted></video>
        <img v-if="outImg" :src="outImg" class="video" />
      </div>

      <div class="small">建议把发送帧率调低：{{ sendFps }} fps（越高越卡）</div>
      <van-slider v-model="sendFps" min="2" max="12" />
    </div>

    <!-- 用于抓帧 -->
    <canvas ref="canvasEl" class="hidden"></canvas>
  </div>
</template>

<script setup>
import { ref, computed, onBeforeUnmount } from "vue";
import { showToast } from "vant";
import { useAuthStore } from "../stores/auth";

const auth = useAuthStore();

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
const WS_BASE = API_BASE.replace(/^http/, "ws"); // http->ws, https->wss

const videoEl = ref(null);
const canvasEl = ref(null);

const cameraOn = ref(false);
const running = ref(false);
const status = ref("");

const modelKey = ref("vox");
const relative = ref(true);
const adaptScale = ref(true);
const useCpu = ref(false);

const sourceFile = ref(null);
const sourcePreview = ref("");
const outImg = ref("");

const sendFps = ref(8); // 2~12

let stream = null;
let ws = null;
let rafId = 0;
let lastTs = 0;
let busy = false; // 防止堆积：上一帧没返回就不发下一帧

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function onPickSource(e) {
  const f = e.target.files?.[0];
  if (!f) return;
  sourceFile.value = f;
  sourcePreview.value = URL.createObjectURL(f);
}

async function openCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    videoEl.value.srcObject = stream;
    cameraOn.value = true;
    status.value = "摄像头已打开";
  } catch (e) {
    showToast("打开摄像头失败（需要 https 或 localhost）");
    status.value = "打开摄像头失败：" + String(e);
  }
}

function closeCamera() {
  stopRealtime();
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  cameraOn.value = false;
  status.value = "摄像头已关闭";
}

function stopRealtime() {
  running.value = false;
  busy = false;
  if (rafId) cancelAnimationFrame(rafId);
  rafId = 0;
  lastTs = 0;

  if (ws) {
    try { ws.close(); } catch {}
    ws = null;
  }
  status.value = "已停止实时";
}

async function startRealtime() {
  if (!cameraOn.value) return showToast("请先打开摄像头");
  if (!sourceFile.value) return showToast("请先选择源图片");
  if (!auth.token) return showToast("请先登录");

  // 建议固定 256x256（和模型一致）
  const canvas = canvasEl.value;
  canvas.width = 256;
  canvas.height = 256;

  outImg.value = "";
  status.value = "连接中…";
  running.value = true;

  ws = new WebSocket(`${WS_BASE}/ws/realtime`);
  ws.binaryType = "arraybuffer";

  ws.onopen = async () => {
    const b64 = await fileToBase64(sourceFile.value);

    ws.send(JSON.stringify({
      token: auth.token,
      model_key: modelKey.value,
      relative: relative.value,
      adapt_scale: adaptScale.value,
      use_cpu: useCpu.value,
      source_base64: b64, // 带 data:image/...;base64, 也没事，后端会处理
    }));

    status.value = "已连接，开始推理中…";
    loop();
  };

  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      // 后端发的状态 json
      try {
        const obj = JSON.parse(ev.data);
        if (obj.ok === false) status.value = "后端错误：" + (obj.error || "");
      } catch {}
      return;
    }

    const blob = new Blob([ev.data], { type: "image/jpeg" });
    const url = URL.createObjectURL(blob);
    outImg.value = url;
    busy = false;
  };

  ws.onerror = () => {
    status.value = "WebSocket 连接错误";
    running.value = false;
  };

  ws.onclose = () => {
    status.value = "WebSocket 已断开";
    running.value = false;
  };
}

function loop(ts = 0) {
  if (!running.value || !ws || ws.readyState !== 1) return;

  const interval = 1000 / sendFps.value;
  if (!lastTs) lastTs = ts;

  if ((ts - lastTs >= interval) && !busy) {
    lastTs = ts;
    busy = true;

    const canvas = canvasEl.value;
    const ctx = canvas.getContext("2d");

    // 把 video 截到 256x256
    ctx.drawImage(videoEl.value, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      if (!blob || !ws || ws.readyState !== 1) {
        busy = false;
        return;
      }
      const buf = await blob.arrayBuffer();
      ws.send(buf);
      // busy 会在收到后端图片后变回 false
    }, "image/jpeg", 0.7);
  }

  rafId = requestAnimationFrame(loop);
}

onBeforeUnmount(() => {
  closeCamera();
});
</script>

<style scoped>
.page { padding-bottom: 16px; }
.card {
  margin: 12px;
  padding: 12px;
  border-radius: 14px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  color: rgba(255,255,255,0.92);
}
.h { font-weight: 700; margin: 10px 0 8px; }
.small { margin-top: 8px; opacity: 0.85; font-size: 12px; }
.row { display: flex; gap: 10px; margin-top: 10px; align-items: center; }
.img { width: 120px; height: 120px; object-fit: cover; border-radius: 12px; border: 1px solid rgba(255,255,255,0.12); }
.video { width: 48%; border-radius: 12px; border: 1px solid rgba(255,255,255,0.12); }
.hidden { display: none; }
.file { margin: 8px 0 6px; width: 100%; color: rgba(255,255,255,0.9); }
</style>
