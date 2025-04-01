<script lang="ts" setup>
import { Timestamp } from '@bufbuild/protobuf';
import { type Point, RequestParameters } from './model/chronotopia_pb';
import { type Map, Popup, type LineLayerSpecification } from 'maplibre-gl';
import { MglFullscreenControl, MglGeolocateControl, MglMap, MglMarker, MglNavigationControl } from '#components';
import type { GeoJSON, GeoJsonProperties, Geometry } from 'geojson';

const layout = {
  'line-join': 'round',
  'line-cap': 'round'
} as LineLayerSpecification['layout'];

// const paint = {
//   'line-color': '#FF0000',
//   'line-width': 8
// } as LineLayerSpecification['paint'];

const { $api } = useNuxtApp();
const trips = await $api.getTrips(new RequestParameters({ from: new Timestamp({ seconds: BigInt(1742601500) }) }));
const geojsonSource = ref({
  data: {
    type: 'FeatureCollection',
    features: trips.trips.map(trip => {
      return {
        type: 'Feature',
        properties: {},
        geometry: {
          type: 'LineString',
          coordinates:
            trip.points.map(v => [v.latlon!.lon, v.latlon!.lat])
        }
      }
    })
  },
});



const routeMatchTrace = await $api.getRouteMatchTrace({});

const selectWindow = (index: number) => {
  geojsonSource.value.data = {
    type: 'FeatureCollection',
    features: routeMatchTrace.windowTraces[index]!.segments.map(segment => {
      return {
        type: 'Feature',
        properties: {},
        geometry: {
          type: 'LineString',
          coordinates:
            segment.coordinates.map(v => [v.lon, v.lat])
        }
      }
    })
  }
}

const debugGeojsonSource: Ref<undefined | GeoJSON<Geometry, GeoJsonProperties>> = ref(undefined);

if (import.meta.client) {
  // DO NOT USE THIS SERVER SIDE, IT WILL BLOW UP VITE WITH THE MOST VAGUE OF ERRORS
  const map = useMglMap();
  const hueStep = 255 / routeMatchTrace.windowTraces.length;
  const layers: string[] = [];

  watch(() => map.isLoaded, () => {
    for (const windowIdx in routeMatchTrace.windowTraces) {
      const window = routeMatchTrace.windowTraces[windowIdx];
      const sourceId = `window-${windowIdx}`;
      layers.push(sourceId);
      const geojson: GeoJSON<Geometry, GeoJsonProperties> = {
        type: 'FeatureCollection',
        features: window.segments.map(segment => {
          return {
            type: 'Feature',
            properties: {
              window: windowIdx,
            },
            geometry: {
              type: 'LineString',
              coordinates:
                segment.coordinates.map(v => [v.lon, v.lat])
            }
          }
        })
      }

      const mmap = map.map as Map;

      mmap.addSource(sourceId, {
        type: 'geojson',
        data: geojson
      });
      mmap.addLayer({
        id: sourceId,
        type: 'line',
        source: sourceId,
        layout: layout,
        paint: {
          'line-color': `hsl(${Math.round(hueStep * Number(windowIdx))}, 50%, 50%)`,
          'line-width': 8
        } as LineLayerSpecification['paint']
      });

      // Create a popup, but don't add it to the map yet.
      const popup = new Popup({
        closeButton: false,
        closeOnClick: false
      });

      // Make sure to detect marker change for overlapping markers
      // and use mousemove instead of mouseenter event
      let currentFeatureCoordinates = "";
      mmap.on('mousemove', layers, (e) => {
        const featureCoordinates = e.lngLat.toString();
        if (currentFeatureCoordinates !== featureCoordinates) {
          currentFeatureCoordinates = featureCoordinates;

          // Change the cursor style as a UI indicator.
          mmap.getCanvas().style.cursor = 'pointer';

          const coordinates = e.lngLat;
          const description = e.features![0].properties.window;

          // Populate the popup and set its coordinates
          // based on the feature found.
          popup.setLngLat(coordinates).setHTML(description).addTo(mmap);
        }
      });

      mmap.on('mouseleave', layers, () => {
        currentFeatureCoordinates = "";
        mmap.getCanvas().style.cursor = '';
        popup.remove();
      });
    }
  })
}

const popupRef = useTemplateRef('popup')
const closePopup = () => {
  popupRef.value!.remove();
};

const debugOSMAt = async (point: Point) => {
  const m = await $api.oSMNetworkAroundPoint(point.latlon!);
  const gj: GeoJSON<Geometry, GeoJsonProperties> = JSON.parse(m.value);
  debugGeojsonSource.value = gj;
}
</script>

<template>
  <div style="height: 100%;">
    <p>hi</p>
    <div style="display: flex; flex-direction: row;">
      <div v-for="(window, i) in routeMatchTrace.windowTraces" :key="i"
        style="width: 1rem; height: 2rem; padding: 1rem; cursor: pointer; border: solid 1px black;"
        @mouseup="selectWindow(i)">{{ i }}</div>
    </div>
    <ClientOnly>
      <MglMap map-style="https://api.maptiler.com/maps/streets/style.json?key=Ic6Mr5qetb5kn90hyEzO" :zoom="6" :center="[-43.4795272,
      -22.738402100000002]">
        <MglFullscreenControl />
        <MglNavigationControl />
        <MglGeolocateControl />
        <MglGeoJsonSource v-if="debugGeojsonSource" source-id="debug" :data="debugGeojsonSource">
          <MglLineLayer layer-id="debug" />
        </MglGeoJsonSource>

        <MglMarker v-for="(point, index) in routeMatchTrace.trip?.points" :key="Number(index)"
          :coordinates="[point.latlon!.lon, point.latlon!.lat]">
          <template #marker>
            <div
              style="width: 20px; height: 20px; text-align: center; border-radius: 5px; background-color: darkblue; color: white;">
              {{
                index }}</div>
          </template>
          <mgl-popup ref="popup">
            <button
              @click="debugGeojsonSource = JSON.parse(routeMatchTrace.pointCandidates[index].value) as GeoJSON<Geometry, GeoJsonProperties>">show
              candidates</button>
            <button @click="debugOSMAt(point)">osm around point</button>
            <br />
            <a href="#" @click.prevent="closePopup">Close popup</a>
          </mgl-popup>
        </MglMarker>
      </MglMap>
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
  background-color: #222;
  color: #AAA;
}

div#__nuxt {
  height: 100%;
  width: 100%;
}
</style>
