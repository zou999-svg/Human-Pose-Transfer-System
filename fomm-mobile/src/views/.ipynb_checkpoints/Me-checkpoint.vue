<template>
  <div style="padding-bottom:80px;">
    <div style="padding:16px;">
      <div class="glass" style="padding:16px; display:flex; gap:12px; align-items:center;">
        <div
          style="width:54px;height:54px;border-radius:16px; background:rgba(255,255,255,.10); display:flex;align-items:center;justify-content:center; font-size:26px;"
        >
          👤
        </div>
        <div style="flex:1;">
          <div style="font-weight:900; font-size:16px;">{{ me?.username || "-" }}</div>
          <div class="hint">个人资料会同步保存到后端</div>
        </div>
        <van-button class="btn-ghost" size="small" @click="logout">退出</van-button>
      </div>

      <div class="glass" style="margin-top:14px; padding:14px;">
        <div style="font-weight:900; margin-bottom:10px;">编辑资料</div>

        <van-cell-group inset class="glass" style="box-shadow:none; border:1px solid rgba(255,255,255,.10);">
          <van-field v-model="nickname" label="昵称" placeholder="输入昵称" />
          <van-field v-model="avatarUrl" label="头像URL" placeholder="https://..." />
        </van-cell-group>

        <div style="margin-top:12px;">
          <van-button class="btn-primary" block :loading="saving" @click="save">保存</van-button>
        </div>
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
import { ref, onMounted } from "vue";
import { useRouter } from "vue-router";
import { showToast } from "vant";
import api from "../api/client";
import { useAuthStore } from "../stores/auth";

const router = useRouter();
const auth = useAuthStore();

const me = ref(null);
const nickname = ref("");
const avatarUrl = ref("");
const saving = ref(false);

onMounted(async () => {
  const { data } = await api.get("/me");
  me.value = data;
  nickname.value = data.nickname || "";
  avatarUrl.value = data.avatar_url || "";
});

async function save() {
  saving.value = true;
  try {
    const form = new FormData();
    form.append("nickname", nickname.value);
    form.append("avatar_url", avatarUrl.value);
    const { data } = await api.patch("/me", form);
    me.value = data;
    showToast("已保存");
  } catch (e) {
    showToast(e?.response?.data?.detail || "保存失败");
  } finally {
    saving.value = false;
  }
}

function logout() {
  auth.logout();
  router.replace("/login");
}
</script>
