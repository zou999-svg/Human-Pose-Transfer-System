<template>
  <div style="padding-bottom:80px;">
    <div style="padding:16px;">
      <div class="glass" style="padding:16px;">
        <div style="font-size:18px; font-weight:900;">✨ 生成动画</div>
        <div class="hint">上传源图片 + 驱动视频，一键生成 MP4。</div>

        <div style="margin-top:12px;">
          <van-cell-group inset class="glass" style="box-shadow:none; border:1px solid rgba(255,255,255,.10);">
            <van-cell title="模型" :value="modelLabel" is-link @click="showModels=true" />
            <van-cell title="relative（推荐）">
              <template #right-icon><van-switch v-model="relative" size="20" /></template>
            </van-cell>
            <van-cell title="adapt_scale（推荐）">
              <template #right-icon><van-switch v-model="adaptScale" size="20" /></template>
            </van-cell>
            <van-cell title="CPU 模式（很慢）">
              <template #right-icon><van-switch v-model="useCpu" size="20" /></template>
            </van-cell>
          </van-cell-group>

          <div v-if="modelDesc" style="margin-top:10px;">
            <van-notice-bar
              :text="modelDesc"
              left-icon="info-o"
              wrapable
              :scrollable="false"
              color="rgba(255,255,255,.88)"
              background="rgba(255,255,255,.06)"
            />
          </div>

          <van-action-sheet
            v-model:show="showModels"
            :actions="modelActions"
            cancel-text="取消"
            @select="onPickModel"
          />
        </div>
      </div>

      <div style="margin-top:14px;" class="glass">
        <div style="padding:14px 14px 8px 14px; font-weight:900;">① 源图片</div>
        <div style="padding:0 14px 14px 14px;">
          <van-uploader
            v-model="srcList"
            :max-count="1"
            accept="image/*"
            :after-read="onSrcRead"
            preview-size="96"
            upload-text="从本地选择图片"
          />
          <div v-if="sourceFile" class="hint" style="margin-top:10px;">
            已选择：{{ sourceFile.name }}（{{ prettySize(sourceFile.size) }}）
          </div>
          <div class="hint" style="margin-top:10px;">建议：清晰正脸 / 光照均匀 / 避免遮挡</div>
        </div>
      </div>

      <div style="margin-top:14px;" class="glass">
        <div style="padding:14px 14px 8px 14px; font-weight:900;">② 驱动视频</div>
        <div style="padding:0 14px 14px 14px;">
          <van-uploader
            v-model="videoList"
            :max-count="1"
            accept="video/*"
            :after-read="onVideoRead"
            :preview-image="false"
            upload-text="从本地选择视频"
          />
          <div v-if="drivingFile" class="hint" style="margin-top:10px;">
            已选择：{{ drivingFile.name }}（{{ prettySize(drivingFile.size) }}）
          </div>

          <div v-if="drivingPreview" style="margin-top:10px;">
            <video :src="drivingPreview" controls style="width:100%; border-radius:14px;"></video>
          </div>

          <div class="hint" style="margin-top:10px;">建议：10~30 秒、主体清晰、镜头稳定</div>
        </div>
      </div>

      <div style="margin-top:14px;" class="glass">
        <div style="padding:14px; display:flex; gap:10px; align-items:center;">
          <div style="font-weight:900;">③ 一键生成</div>
          <van-tag type="primary" plain>MP4 输出</van-tag>
          <van-tag type="success" plain>自动记录</van-tag>
        </div>

        <div style="padding:0 14px 14px 14px;">
          <div v-if="loading" style="margin-bottom:10px;">
            <div class="hint" style="margin-bottom:6px;">上传进度：{{ uploadProgress }}%</div>
            <van-progress :percentage="uploadProgress" stroke-width="8" pivot-text="" />
          </div>

          <van-button
            class="btn-primary"
            block
            :loading="loading"
            :disabled="!sourceFile || !drivingFile"
            @click="createJob"
          >
            🚀 开始生成
          </van-button>

          <div class="hint" style="margin-top:10px;">
            生成后会跳转到任务详情页，自动刷新状态，成功后可播放和下载。
          </div>
        </div>
      </div>

      <div style="margin-top:14px;">
        <van-button type="primary" block @click="$router.push('/realtime')">
          实时摄像头驱动
        </van-button>
      </div>
    </div>

    <van-tabbar route>
      <van-tabbar-item replace to="/" icon="play-circle-o">生成</van-tabbar-item>
      <van-tabbar-item replace to="/history" icon="todo-list-o">记录</van-tabbar-item>
      <van-tabbar-item replace to="/me" icon="user-o">我的</van-tabbar-item>
    </van-tabbar>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from "vue";
import { useRouter } from "vue-router";
import { showToast } from "vant";
import api from "../api/client";

const router = useRouter();

const showModels = ref(false);
const models = ref([]);
const modelActions = ref([]);

const modelKey = ref("vox");

const relative = ref(true);
const adaptScale = ref(true);
const useCpu = ref(false);

const loading = ref(false);
const uploadProgress = ref(0);

const srcList = ref([]);
const videoList = ref([]);

const sourceFile = ref(null);
const drivingFile = ref(null);
const drivingPreview = ref("");
let _videoObjUrl = "";

const modelLabel = computed(() => {
  const m = models.value.find((x) => x.key === modelKey.value);
  return m?.label || modelKey.value || "—";
});
const modelDesc = computed(() => {
  const m = models.value.find((x) => x.key === modelKey.value);
  return m?.desc || "";
});

async function fetchModels() {
  try {
    const { data } = await api.get("/models");
    models.value = data.models || [];

    modelActions.value = models.value.map((m) => ({
      name: m.label,
      key: m.key,
      subname: m.key,
      disabled: m.available === false,
      desc: m.desc || "",
    }));

    // 默认选一个可用的
    const firstOk = models.value.find((m) => m.available !== false);
    const stillOk = models.value.some((m) => m.key === modelKey.value && m.available !== false);
    if (!modelKey.value || !stillOk) {
      modelKey.value = firstOk?.key || models.value?.[0]?.key || "vox";
    }
  } catch (e) {
    showToast("获取模型列表失败");
  }
}

onMounted(fetchModels);

function onPickModel(action) {
  if (action.disabled) {
    showToast("这个模型还没放好权重/配置文件");
    return;
  }
  modelKey.value = action.key;
  showModels.value = false;
}

function prettySize(bytes){
  const mb = bytes / 1024 / 1024;
  if (mb >= 1) return mb.toFixed(2) + "MB";
  const kb = bytes / 1024;
  return kb.toFixed(1) + "KB";
}

function onSrcRead(item) {
  const f = item?.file;
  if (!f) return;
  if (f.size > 8 * 1024 * 1024) {
    showToast("图片太大（建议 < 8MB）");
    srcList.value = [];
    sourceFile.value = null;
    return;
  }
  sourceFile.value = f;
}

function onVideoRead(item) {
  const f = item?.file;
  if (!f) return;
  if (f.size > 150 * 1024 * 1024) {
    showToast("视频太大（建议 < 150MB）");
    videoList.value = [];
    drivingFile.value = null;
    return;
  }
  drivingFile.value = f;

  if (_videoObjUrl) URL.revokeObjectURL(_videoObjUrl);
  _videoObjUrl = URL.createObjectURL(f);
  drivingPreview.value = _videoObjUrl;
}

async function createJob() {
  loading.value = true;
  uploadProgress.value = 0;

  try {
    const form = new FormData();
    form.append("source_image", sourceFile.value);
    form.append("driving_video", drivingFile.value);
    form.append("model_key", modelKey.value);
    form.append("relative", String(relative.value));
    form.append("adapt_scale", String(adaptScale.value));
    form.append("use_cpu", String(useCpu.value));

    const { data } = await api.post("/animations", form, {
      headers: { "Content-Type": "multipart/form-data" },
      onUploadProgress: (evt) => {
        if (!evt.total) return;
        uploadProgress.value = Math.min(100, Math.round((evt.loaded / evt.total) * 100));
      },
    });

    showToast("任务已创建");
    router.push(`/job/${data.job_id}`);
  } catch (e) {
    showToast(e?.response?.data?.detail || "创建失败");
  } finally {
    loading.value = false;
  }
}
</script>
