// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: "2024-11-01",
  devtools: { enabled: true },

  modules: ["@nuxt/eslint", "@nuxt/fonts", "@nuxt/icon", "@nuxt/ui"],

  runtimeConfig: {
    apiBaseUrl: "http://localhost:10000",
    serviceWorkers: "run",
    public: {
      apiBaseUrl: "http://localhost:10000",
      ci: "false",
      siteEnv: "dev",
    },
  },
});
