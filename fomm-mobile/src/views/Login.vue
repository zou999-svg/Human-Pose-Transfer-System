<template>
  <div style="padding:16px;">
    <div class="glass" style="padding:18px;">
      <div style="display:flex; align-items:center; gap:10px;">
        <div style="font-size:26px;">🎭</div>
        <div>
          <div style="font-size:18px; font-weight:900;">FOMM · 移动端</div>
          <div class="hint">登录后即可上传素材，生成会动的照片</div>
        </div>
      </div>
    </div>

    <div class="glass" style="margin-top:14px; padding:14px;">
      <van-tabs v-model:active="tab" animated>
        <van-tab title="登录">
          <van-form @submit="onLogin" style="margin-top:12px;">
            <van-field v-model="login.username" name="username" label="账号" placeholder="用户名" required />
            <van-field v-model="login.password" name="password" type="password" label="密码" placeholder="至少 6 位" required />
            <div style="margin-top:14px;">
              <van-button class="btn-primary" block native-type="submit" :loading="loading">登录</van-button>
            </div>
          </van-form>
        </van-tab>

        <van-tab title="注册">
          <van-form @submit="onRegister" style="margin-top:12px;">
            <van-field v-model="reg.username" name="username" label="账号" placeholder="用户名(>=3)" required />
            <van-field v-model="reg.password" name="password" type="password" label="密码" placeholder="至少 6 位" required />
            <div style="margin-top:14px;">
              <van-button class="btn-primary" block native-type="submit" :loading="loading">注册并登录</van-button>
            </div>
          </van-form>
        </van-tab>
      </van-tabs>
    </div>

    <div class="hint" style="margin-top:12px; padding:0 6px;">
      小提示：如果你用的是 AutoDL 的端口映射，确保后端 8000 正在运行；前端 5173 正在运行。
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";
import { useRouter } from "vue-router";
import { showToast } from "vant";
import api from "../api/client";
import { useAuthStore } from "../stores/auth";

const router = useRouter();
const auth = useAuthStore();

const tab = ref(0);
const loading = ref(false);

const login = ref({ username: "", password: "" });
const reg = ref({ username: "", password: "" });

async function onLogin() {
  loading.value = true;
  try {
    const { data } = await api.post("/auth/login", login.value);
    auth.setToken(data.access_token);
    showToast("登录成功");
    router.replace("/");
  } catch (e) {
    showToast(e?.response?.data?.detail || "登录失败");
  } finally {
    loading.value = false;
  }
}

async function onRegister() {
  loading.value = true;
  try {
    await api.post("/auth/register", reg.value);
    const { data } = await api.post("/auth/login", reg.value);
    auth.setToken(data.access_token);
    showToast("注册成功");
    router.replace("/");
  } catch (e) {
    showToast(e?.response?.data?.detail || "注册失败");
  } finally {
    loading.value = false;
  }
}
</script>
