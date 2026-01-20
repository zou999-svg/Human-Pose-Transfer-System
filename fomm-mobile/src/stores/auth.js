import { defineStore } from "pinia";

export const useAuthStore = defineStore("auth", {
  state: () => ({
    token: localStorage.getItem("token") || "",
    me: null,
  }),
  actions: {
    setToken(t) {
      this.token = t;
      localStorage.setItem("token", t);
    },
    logout() {
      this.token = "";
      this.me = null;
      localStorage.removeItem("token");
    },
  },
});
