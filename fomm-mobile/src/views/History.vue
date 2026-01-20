<template>
  <div style="padding-bottom:80px;">
    <div style="padding:16px;">
      <div class="glass" style="padding:16px;">
        <div style="font-size:18px; font-weight:900;">📚 生成记录</div>
        <div class="hint">筛选查看：全部 / 成功 / 失败 / 运行中</div>
      </div>

      <div style="margin-top:14px;" class="glass">
        <div style="padding:10px 12px;">
          <van-tabs
            v-model:active="filter"
            swipeable
            color="rgba(255,255,255,.85)"
            title-active-color="rgba(0,0,0,.85)"
            title-inactive-color="rgba(255,255,255,.75)"
          >
            <van-tab title="全部" name="all" />
            <van-tab title="成功" name="success" />
            <van-tab title="失败" name="failed" />
            <van-tab title="运行中" name="running" />
          </van-tabs>
        </div>

        <van-pull-refresh v-model="refreshing" @refresh="reload">
          <div style="padding:10px 14px;">
            <van-list
              v-model:loading="loading"
              :finished="finished"
              finished-text="没有更多了"
              :immediate-check="false"
              @load="loadMore"
            >
              <template v-if="filteredItems.length">
                <van-cell
                  v-for="it in filteredItems"
                  :key="it.id"
                  is-link
                  @click="go(it.id)"
                >
                  <template #title>
                    <div style="display:flex; align-items:center; gap:8px;">
                      <div style="font-weight:900;">{{ modelLabel(it.model_key) }}</div>
                      <van-tag :type="tagType(it.status)" plain>{{ it.status }}</van-tag>
                    </div>
                  </template>
                  <template #label>
                    <div class="hint">{{ formatTime(it.created_at) }}</div>
                  </template>
                </van-cell>
              </template>

              <template v-else-if="finished && !loading">
                <div style="padding:30px 0;">
                  <van-empty description="暂无记录，去生成一个吧～" />
                </div>
              </template>
            </van-list>
          </div>
        </van-pull-refresh>
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
import api from "../api/client";

const router = useRouter();

const items = ref([]);
const loading = ref(false);
const finished = ref(false);
const refreshing = ref(false);
const page = ref(1);
const pageSize = 20;

const filter = ref("all");

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

const filteredItems = computed(() => {
  if (filter.value === "all") return items.value;
  if (filter.value === "running") {
    return items.value.filter(x => x.status === "queued" || x.status === "running");
  }
  return items.value.filter(x => x.status === filter.value);
});

function formatTime(s) {
  return new Date(s).toLocaleString();
}

function tagType(status) {
  if (status === "success") return "success";
  if (status === "failed") return "danger";
  if (status === "running") return "primary";
  if (status === "queued") return "warning";
  return "default";
}

async function loadMore() {
  if (loading.value || finished.value) return;

  loading.value = true;
  try {
    const { data } = await api.get("/animations", { params: { page: page.value, page_size: pageSize } });
    if (!data.length) finished.value = true;
    items.value.push(...data);
    page.value += 1;
  } finally {
    loading.value = false;
  }
}

async function reload() {
  try {
    items.value = [];
    page.value = 1;
    finished.value = false;
    await loadMore();
  } finally {
    refreshing.value = false;
  }
}

function go(id) {
  router.push(`/job/${id}`);
}

onMounted(async () => {
  await loadModels();
  await reload();
});
</script>
