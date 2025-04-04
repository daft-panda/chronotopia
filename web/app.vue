<script lang="ts" setup>
import type { Point, WindowTrace, RoadSegment } from './model/chronotopia_pb';
import { type Map, Popup, type LineLayerSpecification, type LngLatLike } from 'maplibre-gl';
import { MglFullscreenControl, MglGeolocateControl, MglMap, MglMarker, MglNavigationControl } from '#components';
import type { GeoJSON, GeoJsonProperties, Geometry } from 'geojson';

const layout = {
  'line-join': 'round',
  'line-cap': 'round'
} as LineLayerSpecification['layout'];

const { $api } = useNuxtApp();
const routeMatchTrace = await $api.getRouteMatchTrace({});
const selectedWindowIndex: Ref<number | null> = ref(null);
const selectedWindow: Ref<WindowTrace | null> = ref(null);
const highlightedSegment: Ref<RoadSegment | null> = ref(null);

const selectWindow = (index: number) => {
  selectedWindowIndex.value = index;
  selectedWindow.value = routeMatchTrace.windowTraces[index];
}

const selectSegment = (segment: RoadSegment) => {
  highlightedSegment.value = segment;
}

const hoveredSegment: Ref<RoadSegment | null> = ref(null);

const hoverSegment = (segment: RoadSegment) => {
  hoveredSegment.value = segment;
}

const clearHoveredSegment = () => {
  hoveredSegment.value = null;
}

const debugGeojsonSource: Ref<undefined | GeoJSON<Geometry, GeoJsonProperties>> = ref(undefined);

// Calculate window statistics
const getWindowStats = (window: WindowTrace) => {
  if (!window) return null;

  // Calculate total distance (very rough approximation)
  let totalDistance = 0;
  const totalSegments = window.segments.length;
  const highwayTypes = new Set<string>();

  window.segments.forEach(segment => {
    highwayTypes.add(segment.highwayType);

    // Simple distance calculation between consecutive coordinates
    for (let i = 1; i < segment.coordinates.length; i++) {
      const prev = segment.coordinates[i - 1];
      const curr = segment.coordinates[i];
      // Haversine formula would be more accurate, but this is a simple approximation
      totalDistance += Math.sqrt(
        Math.pow((curr.lat - prev.lat) * 111.32, 2) +
        Math.pow((curr.lon - prev.lon) * 111.32 * Math.cos(prev.lat * Math.PI / 180), 2)
      );
    }
  });

  return {
    segmentCount: totalSegments,
    distance: totalDistance.toFixed(2), // in km
    highwayTypes: Array.from(highwayTypes),
    isBridge: window.bridge ? 'Yes' : 'No',
    pointRange: `${window.start} - ${window.end}`
  };
};

// Get colors for the window selector
const getWindowColor = (index: number) => {
  const hueStep = 255 / routeMatchTrace.windowTraces.length;
  return `hsl(${Math.round(hueStep * index)}, 50%, 50%)`;
};

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
        features: window.segments.map((segment, segIdx) => {
          // Use the actual segment coordinates, which are already filtered based on interim indices
          const coordinates = segment.coordinates.map(v => [v.lon, v.lat]);

          // Create properties object with all relevant info
          const properties = {
            window: windowIdx,
            segmentId: segment.id.toString(),
            segmentIndex: segIdx,
            name: segment.name || 'Unnamed',
            highwayType: segment.highwayType,
            isOneway: segment.isOneway,
            uniqueId: `${windowIdx}-${segment.id.toString()}`,
            // Add interim indices if available
            interimStartIdx: segment.interimStartIdx !== undefined ? segment.interimStartIdx : 0,
            interimEndIdx: segment.interimEndIdx !== undefined ? segment.interimEndIdx :
              (segment.fullCoordinates ? segment.fullCoordinates.length - 1 : coordinates.length - 1),
            isPartialSegment: segment.interimStartIdx !== undefined || segment.interimEndIdx !== undefined,
          };

          return {
            type: 'Feature',
            properties: properties,
            geometry: {
              type: 'LineString',
              coordinates: coordinates
            }
          }
        })
      }

      const mmap = map.map as Map;

      mmap.addSource(sourceId, {
        type: 'geojson',
        data: geojson
      });

      // Add the main line layer
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

      // For segments with interim indices, also add full segment with low opacity
      // to provide context
      if (window.segments.some(s => s.interimStartIdx !== undefined || s.interimEndIdx !== undefined)) {
        const fullSegmentSourceId = `window-${windowIdx}-full`;
        const fullSegmentsGeoJSON: GeoJSON<Geometry, GeoJsonProperties> = {
          type: 'FeatureCollection',
          features: window.segments.filter(segment =>
            segment.fullCoordinates &&
            (segment.interimStartIdx !== undefined || segment.interimEndIdx !== undefined)
          ).map((segment, segIdx) => {
            // Use full coordinates for context
            const fullCoordinates = segment.fullCoordinates?.map(v => [v.lon, v.lat]) || [];

            return {
              type: 'Feature',
              properties: {
                window: windowIdx,
                segmentId: segment.id.toString(),
                segmentIndex: segIdx,
                uniqueId: `${windowIdx}-${segment.id.toString()}-full`,
                isFullSegment: true
              },
              geometry: {
                type: 'LineString',
                coordinates: fullCoordinates
              }
            }
          })
        };

        // Only add if there are features
        if (fullSegmentsGeoJSON.features.length > 0) {
          mmap.addSource(fullSegmentSourceId, {
            type: 'geojson',
            data: fullSegmentsGeoJSON
          });

          // Add full segments with low opacity and dashed
          mmap.addLayer({
            id: fullSegmentSourceId,
            type: 'line',
            source: fullSegmentSourceId,
            layout: {
              'line-join': 'round',
              'line-cap': 'round'
            },
            paint: {
              'line-color': `hsl(${Math.round(hueStep * Number(windowIdx))}, 30%, 70%)`,
              'line-width': 4,
              'line-opacity': 0.3,
              'line-dasharray': [2, 2]
            } as LineLayerSpecification['paint']
          });
        }
      }

      // Add a highlight outline layer that will be visible when a segment is hovered
      mmap.addLayer({
        id: `${sourceId}-highlight`,
        type: 'line',
        source: sourceId,
        layout: layout,
        paint: {
          'line-color': '#FFFFFF',
          'line-width': 12,
          'line-opacity': 0 // Start with 0 opacity, we'll update this with setFilter
        } as LineLayerSpecification['paint'],
        filter: ['==', ['get', 'uniqueId'], ''] // Start with an empty filter
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
          const feature = e.features![0];
          const description = `
        <strong>Window:</strong> ${feature.properties.window}<br>
        <strong>Road:</strong> ${feature.properties.name}<br>
        <strong>Type:</strong> ${feature.properties.highwayType}<br>
        <strong>One-way:</strong> ${feature.properties.isOneway ? 'Yes' : 'No'}<br>
        ${feature.properties.isPartialSegment ?
              `<strong>Partial segment:</strong> Using ${feature.properties.interimStartIdx}-${feature.properties.interimEndIdx}`
              : ''}
      `;

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

      // Add click handlers for segments
      mmap.on('click', sourceId, (e) => {
        if (e.features && e.features.length > 0) {
          const windowIndex = parseInt(e.features[0].properties.window);
          const segmentIndex = e.features[0].properties.segmentIndex;

          selectWindow(windowIndex);
          selectSegment(routeMatchTrace.windowTraces[windowIndex].segments[segmentIndex]);

          // Fly to the clicked segment for better visibility
          const geometry = e.features[0].geometry;
          if (geometry.type === 'LineString' || geometry.type === 'MultiLineString') {
            // Now TypeScript knows this geometry has coordinates
            const coords = geometry.type === 'LineString'
              ? geometry.coordinates
              : geometry.coordinates[0]; // Take first line of multiline

            if (coords && coords.length > 0) {
              const midpoint = coords[Math.floor(coords.length / 2)];
              mmap.flyTo({
                center: midpoint as LngLatLike,
                zoom: 14,
                speed: 0.8
              });
            }
          }
        }
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
  <div class="main-container">
    <div class="header">
      <h1>ChronoTopia Route Visualization</h1>
    </div>

    <div class="window-selector">
      <div v-for="(window, i) in routeMatchTrace.windowTraces" :key="i" class="window-button" :style="{
        backgroundColor: getWindowColor(i),
        border: selectedWindow && selectedWindow === window ? '3px solid white' : '1px solid #333'
      }" @mouseup="selectWindow(i)">
        {{ i }}
      </div>
    </div>

    <div class="content-container">
      <!-- Structured view panel -->
      <div v-if="selectedWindow" class="info-panel">
        <div class="panel-header">
          <h2>Window {{ selectedWindowIndex }} Details</h2>
        </div>

        <div class="stats-container">
          <div class="stat-item">
            <div class="stat-label">Points Range:</div>
            <div class="stat-value">{{ selectedWindow.start }} - {{ selectedWindow.end }}</div>
          </div>

          <div class="stat-item">
            <div class="stat-label">Segments:</div>
            <div class="stat-value">{{ selectedWindow.segments.length }}</div>
          </div>

          <div class="stat-item">
            <div class="stat-label">Is Bridge:</div>
            <div class="stat-value">{{ selectedWindow.bridge ? 'Yes' : 'No' }}</div>
          </div>

          <div class="stat-item">
            <div class="stat-label">Distance (approx):</div>
            <div class="stat-value">{{ getWindowStats(selectedWindow)?.distance }} km</div>
          </div>

          <div class="stat-item">
            <div class="stat-label">Highway Types:</div>
            <div class="stat-value">
              <span v-for="(type, index) in getWindowStats(selectedWindow)?.highwayTypes" :key="index"
                class="highway-type">
                {{ type }}
              </span>
            </div>
          </div>
        </div>

        <div class="segment-list">
          <h3>Segments</h3>
          <div v-for="segment in selectedWindow.segments" :key="segment.id.toString()" class="segment-item" :class="{
            'highlighted': highlightedSegment === segment,
            'hovered': hoveredSegment === segment,
            'partial-segment': segment.interimStartIdx !== undefined || segment.interimEndIdx !== undefined
          }" @click="selectSegment(segment)" @mouseenter="hoverSegment(segment)" @mouseleave="clearHoveredSegment()">
            <div class="segment-name">{{ segment.name || 'Unnamed Road' }}</div>
            <div class="segment-details">
              <span class="segment-type">{{ segment.highwayType }}</span>
              <span v-if="segment.isOneway" class="segment-oneway">One-way</span>
              <span class="segment-id">OSM ID: {{ segment.osmWayId.toString() }}</span>
              <span v-if="segment.interimStartIdx !== undefined || segment.interimEndIdx !== undefined"
                class="segment-partial">
                Partial: {{ segment.interimStartIdx || 0 }}-{{ segment.interimEndIdx !== undefined ?
                  segment.interimEndIdx : (segment.fullCoordinates ? segment.fullCoordinates.length - 1 : '?') }}
              </span>
            </div>
          </div>
        </div>
      </div>

      <!-- Map view -->
      <div class="map-container">
        <ClientOnly>
          <MglMap map-style="https://api.maptiler.com/maps/streets/style.json?key=Ic6Mr5qetb5kn90hyEzO" :zoom="6"
            :center="[-43.4795272, -22.738402100000002]">
            <MglFullscreenControl position="top-right" />
            <MglNavigationControl position="top-right" />
            <MglGeolocateControl position="top-right" />
            <MglGeoJsonSource v-if="debugGeojsonSource" source-id="debug" :data="debugGeojsonSource">
              <MglLineLayer layer-id="debug" />
            </MglGeoJsonSource>

            <!-- Trip points -->
            <MglMarker v-for="(point, index) in routeMatchTrace.trip?.points" :key="Number(index)"
              :coordinates="[point.latlon!.lon, point.latlon!.lat]">
              <template #marker>
                <div class="point-marker" :class="{
                  'start-point': selectedWindow && index === selectedWindow.start,
                  'end-point': selectedWindow && index === selectedWindow.end,
                  'window-point': selectedWindow && index >= selectedWindow.start && index <= selectedWindow.end
                }">
                  {{ index }}
                </div>
              </template>
              <MglPopup ref="popup">
                <div class="popup-content">
                  <h3>Point {{ index }}</h3>
                  <p>Coordinates: {{ point.latlon?.lat.toFixed(6) }}, {{ point.latlon?.lon.toFixed(6) }}</p>
                  <div class="popup-actions">
                    <button
                      @click="debugGeojsonSource = JSON.parse(routeMatchTrace.pointCandidates[index].value) as GeoJSON<Geometry, GeoJsonProperties>">
                      Show Candidates
                    </button>
                    <button @click="debugOSMAt(point)">OSM Around Point</button>
                  </div>
                  <a href="#" class="close-link" @click.prevent="closePopup">Close</a>
                </div>
              </MglPopup>
            </MglMarker>
          </MglMap>
        </ClientOnly>
      </div>
    </div>
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
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

div#__nuxt {
  height: 100%;
  width: 100%;
}

.main-container {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.header {
  padding: 10px 20px;
  background-color: #333;
  border-bottom: 1px solid #444;
}

.header h1 {
  margin: 0;
  font-size: 1.5rem;
  color: #fff;
}

.window-selector {
  display: flex;
  flex-wrap: wrap;
  padding: 10px;
  background-color: #2a2a2a;
  border-bottom: 1px solid #444;
}

.window-button {
  width: 40px;
  height: 40px;
  margin: 5px;
  border-radius: 5px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-weight: bold;
  transition: all 0.2s ease;
  color: white;
  text-shadow: 0 0 3px rgba(0, 0, 0, 0.8);
}

.window-button:hover {
  transform: scale(1.1);
  box-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
}

.content-container {
  flex: 1;
  display: flex;
  overflow: hidden;
}

.info-panel {
  width: 350px;
  background-color: #2a2a2a;
  border-right: 1px solid #444;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

.panel-header {
  padding: 15px;
  background-color: #333;
  border-bottom: 1px solid #444;
}

.panel-header h2 {
  margin: 0;
  font-size: 1.2rem;
  color: #fff;
}

.stats-container {
  padding: 15px;
  border-bottom: 1px solid #444;
}

.stat-item {
  margin-bottom: 10px;
  display: flex;
}

.stat-label {
  font-weight: bold;
  width: 120px;
  color: #ddd;
}

.stat-value {
  flex: 1;
  color: #fff;
}

.highway-type {
  display: inline-block;
  background-color: #444;
  padding: 2px 6px;
  margin: 2px;
  border-radius: 3px;
  font-size: 0.8rem;
}

.segment-list {
  padding: 15px;
  flex: 1;
  overflow-y: auto;
}

.segment-list h3 {
  margin-top: 0;
  color: #ddd;
  border-bottom: 1px solid #444;
  padding-bottom: 5px;
}

.segment-item {
  background-color: #333;
  margin-bottom: 10px;
  border-radius: 5px;
  padding: 10px;
  cursor: pointer;
  transition: all 0.2s;
}

.segment-item:hover {
  background-color: #3a3a3a;
}

.segment-item.highlighted {
  background-color: #444;
  border-left: 5px solid #7af;
}

.segment-item.hovered {
  background-color: #3f3f3f;
  transform: translateX(5px);
  box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
}

.segment-name {
  font-weight: bold;
  margin-bottom: 5px;
  color: #fff;
}

.segment-details {
  display: flex;
  flex-wrap: wrap;
  font-size: 0.8rem;
  gap: 5px;
}

.segment-type,
.segment-oneway,
.segment-id {
  background-color: #222;
  padding: 2px 6px;
  border-radius: 3px;
}

.segment-oneway {
  background-color: #553;
}

.map-container {
  flex: 1;
  position: relative;
  height: 100%;
  width: 100%;
}

.point-marker {
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background-color: #444;
  color: white;
  font-weight: bold;
  border: 2px solid #222;
  transition: all 0.2s;
}

.start-point {
  background-color: #4caf50;
  transform: scale(1.3);
  z-index: 10;
}

.end-point {
  background-color: #f44336;
  transform: scale(1.3);
  z-index: 10;
}

.window-point {
  background-color: #2196f3;
  border-color: white;
}

.popup-content {
  color: #333;
  padding: 5px;
}

.popup-content h3 {
  margin-top: 0;
  margin-bottom: 5px;
}

.popup-actions {
  display: flex;
  gap: 5px;
  margin-top: 10px;
}

.popup-actions button {
  background-color: #555;
  color: white;
  border: none;
  padding: 5px 10px;
  border-radius: 3px;
  cursor: pointer;
  font-size: 0.8rem;
  transition: background-color 0.2s;
}

.popup-actions button:hover {
  background-color: #777;
}

.close-link {
  display: block;
  margin-top: 10px;
  color: #555;
  text-align: right;
  font-size: 0.8rem;
}
</style>