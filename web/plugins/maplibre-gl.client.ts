import * as MaplibreGL from "maplibre-gl";
import VueMaplibreGl from "@indoorequal/vue-maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

export default defineNuxtPlugin((nuxtApp) => {
  nuxtApp.vueApp.use(VueMaplibreGl);
});
