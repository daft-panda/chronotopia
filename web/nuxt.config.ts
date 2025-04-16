import { resolve } from "path";
import wasm from "vite-plugin-wasm";

// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: "2024-11-01",
  devtools: { enabled: true },
  ssr: false,

  modules: [
    "@nuxt/eslint",
    "@nuxt/fonts",
    "@nuxt/icon",
    "@nuxt/ui",
    "nuxt-maplibre",
    "nuxt-workers",
  ],

  css: ["~/assets/css/main.css"],

  runtimeConfig: {
    apiBaseUrl: "http://localhost:10000",
    serviceWorkers: "run",
    public: {
      apiBaseUrl: "http://localhost:10000",
      ci: "false",
      siteEnv: "dev",
    },
  },

  vite: {
    plugins: [wasm()],
    worker: {
      plugins: () => [wasm()],
    },
  },

  nitro: {
    publicAssets: [
      {
        // This makes the resources available at /streets-gl-lib/resources
        dir: resolve("./node_modules/streets-gl-lib/dist/"),
        baseURL: "/streets-gl",
      },
      {
        // This makes the resources available at /streets-gl-lib/resources
        dir: resolve("./node_modules/streets-gl-lib/dist/resources/misc"),
        baseURL: "/misc",
      },
    ],
  },
});
