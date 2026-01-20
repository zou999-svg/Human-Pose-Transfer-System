import { createRouter, createWebHistory } from "vue-router";
import { useAuthStore } from "../stores/auth";

import Login from "../views/Login.vue";
import Home from "../views/Home.vue";
import History from "../views/History.vue";
import JobDetail from "../views/JobDetail.vue";
import Me from "../views/Me.vue";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/login", component: Login },
    { path: "/", component: Home, meta: { auth: true } },
    { path: "/history", component: History, meta: { auth: true } },
    { path: "/job/:id", component: JobDetail, meta: { auth: true } },
    { path: "/me", component: Me, meta: { auth: true } },
    { path: "/realtime", component: () => import("../views/Realtime.vue"), meta: { auth: true } },
  ],
});

router.beforeEach((to) => {
  const auth = useAuthStore();
  if (to.meta.auth && !auth.token) return "/login";
  if (to.path === "/login" && auth.token) return "/";
});

export default router;
