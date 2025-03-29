<script lang="ts" setup>
import { Timestamp } from '@bufbuild/protobuf';
import { RequestParameters } from './model/chronotopia_pb';
import type { FeatureCollection, LineString } from 'geojson';
import type { LineLayerSpecification, LngLatLike } from 'maplibre-gl';

const layout = {
  'line-join': 'round',
  'line-cap': 'round'
} as LineLayerSpecification['layout'];

const paint = {
  'line-color': '#FF0000',
  'line-width': 8
} as LineLayerSpecification['paint'];

const { $api } = useNuxtApp();
const trips = await $api.getTrips(new RequestParameters({ from: new Timestamp({ seconds: BigInt(1742601500) }) }));
const geojsonSource = {
  data: ref<FeatureCollection<LineString>>({
    type: 'FeatureCollection',
    features: trips.trips.map(trip => {
      return {
        type: 'Feature',
        properties: {},
        geometry: {
          type: 'LineString',
          coordinates:
            trip.points.map(v => [v.lon, v.lat])
        }
      }
    })
  }),
  show: ref(true)
};

const markerCoordinates = ref<LngLatLike>([13.377507, 52.516267]);
</script>

<template>
  <div style="height: 100%;">
    <h1>Hi!</h1>
    <ClientOnly>
      <mgl-map map-style="https://api.maptiler.com/maps/streets/style.json?key=Ic6Mr5qetb5kn90hyEzO" :zoom="6"
        :center="[9.4777420000, 51.3157550000]">
        <mgl-fullscreen-control />
        <mgl-navigation-control />
        <mgl-geolocation-control />
        <mgl-geo-json-source source-id="geojson" :data="geojsonSource.data as any">
          <mgl-line-layer layer-id="geojson" :layout="layout" :paint="paint" />
        </mgl-geo-json-source>
        <mgl-marker :coordinates="markerCoordinates" color="#cc0000" :scale="0.5" />

      </mgl-map>
    </ClientOnly>
  </div>
</template>

<style lang="css">
html,
body {
  height: 100%;
  width: 100%;
  margin: 0;
  padding: 0;
}

div#__nuxt {
  height: 100%;
  width: 100%;
}
</style>
