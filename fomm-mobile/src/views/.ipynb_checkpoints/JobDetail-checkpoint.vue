<template>
  <div style="padding:16px 16px 20px 16px;">
    <div class="glass" style="padding:14px; display:flex; align-items:center; justify-content:space-between;">
      <div>
        <div style="font-size:16px; font-weight:900;">任务详情</div>
        <div class="hint">ID：{{ id }}</div>
      </div>
      <van-button class="btn-ghost" size="small" @click="router.back()">返回</van-button>
    </div>

    <div v-if="detail" style="margin-top:14px;" class="glass">
      <div style="padding:14px; display:flex; gap:10px; align-items:center;">
        <div style="font-weight:900;">状态</div>
        <van-tag :type="tagType(detail.status)" plain>{{ detail.status }}</van-tag>
        <van-tag type="primary" plain>{{ modelLabel(detail.model_key) }}</van-tag>
      </div>

      <div style="padding:0 14px 14px 14px;">
        <van-steps :active="stepIndex(detail.status)">
          <van-step>排队</van-step>
          <van-step>运行</van-step>
          <van-step>完成</van-step>
        </van-steps>

        <div v-if="detail.status !== 'success'" style="margin-top:12px;">
          <van-notice-bar
            :text="detail.status === 'failed' ? ('失败：' + (detail.error_message || '未知错误')) : '生成中：自动刷新状态…'"
            left-icon="info-o"
            color="rgba(255,255,255,.86)"
            background="rgba(255,255,255,.06)"
          />
        </div>
      </div>
    </div>

    <div v-if="detail" style="margin-top:14px;" class="glass">
      <div style="padding:14px; font-weight:900;">📈 指标</div>
      <div style="padding:0 14px 14px 14px;">
        <van-cell-group inset class="glass" style="box-shadow:none; border:1px solid rgba(255,255,255,.10);">
          <van-cell title="已用时" :value="elapsedText" />
          <van-cell title="预计耗时" :value="etaText" />
          <van-cell title="驱动时长" :value="drivingDurationText" />
          <van-cell title="帧数" :value="framesText" />
          <van-cell title="FPS" :value="fpsText" />
        </van-cell-group>
        <div class="hint" style="margin-top:8px;">
          备注：预计耗时为粗略估算（仅供参考）；帧数/FPS 若后端未提供会显示 “—”。
        </div>
      </div>
    </div>

    <div v-if="detail" style="margin-top:14px;" class="glass">
      <div style="padding:14px; font-weight:900;">素材</div>
      <div style="padding:0 14px 14px 14px;">
        <div v-if="sourceUrl">
          <div class="hint" style="margin-bottom:8px;">源图</div>
          <img :src="sourceUrl" style="width:100%; border-radius:14px;" />
        </div>

        <div v-if="drivingUrl" style="margin-top:12px;">
          <div class="hint" style="margin-bottom:8px;">驱动视频</div>
          <video
            :src="drivingUrl"
            controls
            style="width:100%; border-radius:14px;"
            @loadedmetadata="onDrivingMeta"
          ></video>
        </div>
      </div>
    </div>

    <div v-if="detail" style="margin-top:14px;" class="glass">
      <div style="padding:14px; font-weight:900;">结果</div>
      <div style="padding:0 14px 14px 14px;">
        <template v-if="detail.status === 'success'">
          <video v-if="resultUrl" :src="resultUrl" controls style="width:100%; border-radius:14px;"></video>
          <div style="margin-top:12px;">
            <van-button class="btn-primary" block @click="downloadResult">下载 result.mp4</van-button>
          </div>
        </template>

        <template v-else>
          <div class="hint">等待完成后会自动出现结果视频。</div>
        </template>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from "vue";
import { useRoute, useRouter } from "vue-router";
import { showToast } from "vant";
import api from "../api/client";

const route = useRoute();
const router = useRouter();
const id = route.params.id;

const detail = ref(null);
const sourceUrl = ref("");
const drivingUrl = ref("");
const resultUrl = ref("");

const drivingDuration = ref(null); // seconds
let timer = null;

// 模型 key -> label 映射
const modelsMap = ref({});

async function loadModels() {
  try {
    const { data } = await api.get("/models");
    const arr = data.models || [];
    modelsMap.value = Object.fromEntries(arr.map((m) => [m.key, m]));
  } catch (e) {
    modelsMap.value = {};
  }
}
function modelLabel(key) {
  return modelsMap.value[key]?.label || key;
}

function tagType(status) {
  if (status === "success") return "success";
  if (status === "failed") return "danger";
  if (status === "running") return "primary";
  if (status === "queued") return "warning";
  return "default";
}
function stepIndex(status) {
  if (status === "queued") return 0;
  if (status === "running") return 1;
  if (status === "success") return 2;
  return 1;
}

function fmtSec(sec){
  if (sec == null || Number.isNaN(sec)) return "—";
  sec = Math.max(0, Math.floor(sec));
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  if (m <= 0) return `${s}s`;
  return `${m}m ${s}s`;
}

const elapsedText = computed(() => {
  const d = detail.value;
  if (!d?.created_at) return "—";

  const t0 = new Date(d.created_at).getTime();
  const now = Date.now();

  const isRunning = (d.status === "queued" || d.status === "running");
  const endTs = !isRunning
    ? (d.finished_at ? new Date(d.finished_at).getTime() : now)
    : now;

  return fmtSec((endTs - t0) / 1000);
});

const drivingDurationText = computed(() => {
  if (drivingDuration.value == null) return "—";
  return fmtSec(drivingDuration.value);
});

// 预计耗时：优先用后端提供的字段；没有则用驱动时长做粗略估算
const etaText = computed(() => {
  const d = detail.value;
  if (!d) return "—";
  const serverEta = d.eta_seconds ?? d.expected_seconds ?? d.meta?.eta_seconds;
  if (typeof serverEta === "number") return fmtSec(serverEta);

  if (drivingDuration.value != null) {
    const guess = Math.max(12, drivingDuration.value * 1.2 + 8);
    return fmtSec(guess) + "（估算）";
  }
  return "—";
});

const framesText = computed(() => {
  const d = detail.value;
  const frames = d?.frame_count ?? d?.meta?.frame_count ?? d?.meta?.frames;
  return (typeof frames === "number") ? String(frames) : "—";
});

const fpsText = computed(() => {
  const d = detail.value;
  const fps = d?.fps ?? d?.meta?.fps;
  if (typeof fps === "number") return fps.toFixed(2);

  // 如果有 frames + duration，算一个
  const frames = d?.frame_count ?? d?.meta?.frame_count ?? d?.meta?.frames;
  if (typeof frames === "number" && drivingDuration.value != null && drivingDuration.value > 0) {
    return (frames / drivingDuration.value).toFixed(2) + "（估算）";
  }
  return "—";
});

function onDrivingMeta(e){
  const dur = e?.target?.duration;
  if (typeof dur === "number" && isFinite(dur)) drivingDuration.value = dur;
}

async function fetchDetail() {
  const { data } = await api.get(`/animations/${id}`);
  detail.value = data;
  return data;
}

async function fetchFileAsObjectUrl(url) {
  const resp = await api.get(url, { responseType: "blob" });
  return URL.createObjectURL(resp.data);
}

async function refreshFiles(d) {
  if (!sourceUrl.value && d.source_url) sourceUrl.value = await fetchFileAsObjectUrl(d.source_url);
  if (!drivingUrl.value && d.driving_url) drivingUrl.value = await fetchFileAsObjectUrl(d.driving_url);
  if (d.status === "success" && !resultUrl.value && d.result_url) resultUrl.value = await fetchFileAsObjectUrl(d.result_url);
}

async function loop() {
  try {
    const d = await fetchDetail();
    await refreshFiles(d);
    if (d.status === "queued" || d.status === "running") timer = setTimeout(loop, 2000);
  } catch (e) {
    showToast(e?.response?.data?.detail || "获取任务失败");
  }
}

async function downloadResult() {
  try {
    const resp = await api.get(`/animations/${id}/file/result`, { responseType: "blob" });
    const url = URL.createObjectURL(resp.data);
    const a = document.createElement("a");
    a.href = url;
    a.download = "result.mp4";
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    showToast(e?.response?.data?.detail || "下载失败");
  }
}

onMounted(async () => {
  await loadModels();
  loop();
});

onBeforeUnmount(() => {
  if (timer) clearTimeout(timer);
  // 清理 blob url
  if (sourceUrl.value) URL.revokeObjectURL(sourceUrl.value);
  if (drivingUrl.value) URL.revokeObjectURL(drivingUrl.value);
  if (resultUrl.value) URL.revokeObjectURL(resultUrl.value);
});
</script>
