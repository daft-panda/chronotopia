<template>
  <div class="main-container h-full">
    <div class="header bg-gray-800 text-white p-4 flex justify-between items-center">
      <h1 class="text-xl font-bold">ChronoTopia Map Matching Viewer - Trip {{ $route.params.trip }}</h1>

      <div class="controls flex gap-2">
        <button class="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded"
          :class="{ 'bg-blue-600 hover:bg-blue-500': showConstraints }" @click="toggleConstraints">
          {{ showConstraints ? 'Hide Constraints' : 'Show Constraints' }}
        </button>
        <button class="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded"
          :class="{ 'bg-blue-600 hover:bg-blue-500': showDebugInfo }" @click="toggleDebugInfo">
          {{ showDebugInfo ? 'Hide Debug Info' : 'Show Debug Info' }}
        </button>
        <button class="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded" @click="goBack">
          Back to Trips
        </button>
      </div>
    </div>

    <div v-if="routeMatchTrace" class="window-selector bg-gray-700 p-2 flex flex-wrap">
      <div v-for="(window, i) in routeMatchTrace.windowTraces" :key="i"
        class="window-button m-1 w-10 h-10 flex items-center justify-center rounded cursor-pointer font-bold text-white text-shadow relative"
        :class="getWindowClass(window)" :style="{
          backgroundColor: getWindowColor(i),
          border: selectedWindow && selectedWindow === window ? '3px solid white' : '1px solid #333'
        }" @mouseup="selectWindow(i)">
        {{ i }}
      </div>
    </div>

    <div class="content-container flex flex-col md:flex-row w-full h-full">
      <!-- Structured view panel -->
      <div v-if="selectedWindow" class="info-panel w-full md:w-80 bg-gray-800 text-white overflow-y-auto">
        <div class="panel-header p-4 bg-gray-700 flex justify-between items-center">
          <h2 class="text-lg font-bold">Window {{ selectedWindowIndex }} Details</h2>
          <div class="window-type-indicator px-2 py-1 rounded text-xs" :class="getWindowClass(selectedWindow)">
            {{ selectedWindow.usedConstraints ? 'Constrained' : (selectedWindow.bridge ? 'Constraints failed (bridge)' :
              'Unconstrained') }}
          </div>
        </div>

        <div class="stats-container p-4 border-b border-gray-700">
          <div class="stat-item mb-2 flex">
            <div class="stat-label font-bold w-32 text-gray-300">Points Range:</div>
            <div class="stat-value">{{ selectedWindow.start }} - {{ selectedWindow.end }}</div>
          </div>

          <div class="stat-item mb-2 flex">
            <div class="stat-label font-bold w-32 text-gray-300">Segments:</div>
            <div class="stat-value">{{ selectedWindow.segments.length }}</div>
          </div>

          <div class="stat-item mb-2 flex">
            <div class="stat-label font-bold w-32 text-gray-300">Is Bridge:</div>
            <div class="stat-value">{{ selectedWindow.bridge ? 'Yes' : 'No' }}</div>
          </div>

          <div class="stat-item mb-2 flex">
            <div class="stat-label font-bold w-32 text-gray-300">Distance (approx):</div>
            <div class="stat-value">{{ getWindowStats(selectedWindow)?.distance }} km</div>
          </div>

          <div class="stat-item mb-2 flex">
            <div class="stat-label font-bold w-32 text-gray-300">Highway Types:</div>
            <div class="stat-value">
              <span v-for="(type, index) in getWindowStats(selectedWindow)?.highwayTypes" :key="index"
                class="highway-type bg-gray-600 px-2 py-1 mr-1 text-xs rounded inline-block">
                {{ type }}
              </span>
            </div>
          </div>
        </div>

        <!-- Debug information section - shows when debug mode is enabled -->
        <div v-if="showDebugInfo" class="debug-container p-4 bg-gray-700">
          <h3 class="text-blue-400 text-lg font-bold border-b border-gray-600 pb-2 mb-4">Debug Information</h3>

          <div class="stat-item mb-2 flex">
            <div class="stat-label font-bold w-32 text-gray-300">Constraints:</div>
            <div class="stat-value">{{ getWindowStats(selectedWindow)?.constraintsCount }}</div>
          </div>

          <div class="stat-item mb-2 flex">
            <div class="stat-label font-bold w-32 text-gray-300">Used Constraints:</div>
            <div class="stat-value">{{ getWindowStats(selectedWindow)?.usedConstraints }}</div>
          </div>

          <div class="stat-item mb-2 flex">
            <div class="stat-label font-bold w-32 text-gray-300">Constraint Score:</div>
            <div class="stat-value">{{ getWindowStats(selectedWindow)?.constraintScore }}</div>
          </div>

          <div class="stat-item mb-2 flex">
            <div class="stat-label font-bold w-32 text-gray-300">Unconstrained Score:</div>
            <div class="stat-value">{{ getWindowStats(selectedWindow)?.unconstrainedScore }}</div>
          </div>

          <div class="stat-item mb-2 flex">
            <div class="stat-label font-bold w-32 text-gray-300">Attempted Way IDs:</div>
            <div class="stat-value">{{ getWindowStats(selectedWindow)?.attemptedWayIds }}</div>
          </div>

          <!-- Additional debug notes -->
          <div v-if="selectedWindow.debugNotes && selectedWindow.debugNotes.length > 0"
            class="debug-notes bg-gray-800 p-3 rounded mb-4">
            <h4 class="text-gray-300 font-bold mb-2">Debug Notes:</h4>
            <ul class="list-disc pl-5">
              <li v-for="(note, index) in selectedWindow.debugNotes" :key="index" class="mb-1 text-gray-300">{{ note }}
              </li>
            </ul>
          </div>

          <!-- Constraint list -->
          <div v-if="selectedWindow.constraints && selectedWindow.constraints.length > 0" class="constraints-list mb-4">
            <h4 class="text-gray-300 font-bold mb-2">Constraints:</h4>
            <div v-for="(constraint, index) in selectedWindow.constraints" :key="index"
              class="constraint-item bg-gray-800 p-2 mb-2 rounded border-l-4 border-pink-500">
              <div class="constraint-header font-bold mb-1">
                Point {{ constraint.pointIdx }} → Segment {{ constraint.segmentId }} (Way {{ constraint.wayId }})
              </div>
              <div class="constraint-details text-sm text-gray-400">
                Distance: {{ constraint.distance.toFixed(2) }}m
              </div>
            </div>
          </div>

          <!-- Attempted Way IDs visualization -->
          <div v-if="selectedWindow.attemptedWayIds && selectedWindow.attemptedWayIds.length > 0"
            class="attempted-ways mb-4">
            <h4 class="text-gray-300 font-bold mb-2">Attempted Way IDs:</h4>
            <div class="way-id-tags flex flex-wrap gap-1">
              <span v-for="(wayId, index) in selectedWindow.attemptedWayIds" :key="index"
                class="bg-gray-800 px-2 py-1 rounded text-xs">
                {{ wayId }}
              </span>
            </div>
          </div>

          <div v-if="selectedWindow && showDebugInfo && selectedWindow.bridge" class="path-debugging mt-4">
            <h3 class="text-red-400 text-lg font-bold mb-2">Path Finding Debug</h3>
            <div class="debug-message bg-red-900 bg-opacity-25 p-3 rounded mb-4">
              <p class="mb-2">This window failed to find a valid route with constraints</p>
              <button class="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded text-sm"
                @click="debugFailedWindow">
                Debug Path Finding
              </button>
            </div>

            <div v-if="pathDebugInfo" class="debug-results bg-gray-800 p-3 rounded">
              <h4 class="font-bold mb-2">Path Finding Analysis</h4>
              <div class="text-sm text-red-400 mb-2">{{ pathDebugInfo.reason }}</div>

              <div class="mb-4">
                <div class="font-medium mb-1">Attempted Pairs:</div>
                <div class="grid grid-cols-4 gap-2">
                  <div class="bg-green-900 bg-opacity-40 p-2 rounded text-center">
                    <div class="font-bold text-green-400">
                      {{pathDebugInfo.attemptedPairs.filter(p => p.result!.type ===
                        PathfindingResult_ResultType.SUCCESS).length}}
                    </div>
                    <div class="text-xs">Successful</div>
                  </div>
                  <div class="bg-yellow-900 bg-opacity-40 p-2 rounded text-center">
                    <div class="font-bold text-yellow-400">
                      {{pathDebugInfo.attemptedPairs.filter(p => p.result!.type ===
                        PathfindingResult_ResultType.TOO_FAR).length}}
                    </div>
                    <div class="text-xs">Too Far</div>
                  </div>
                  <div class="bg-red-900 bg-opacity-40 p-2 rounded text-center">
                    <div class="font-bold text-red-400">
                      {{pathDebugInfo.attemptedPairs.filter(p => p.result!.type ===
                        PathfindingResult_ResultType.NO_CONNECTION).length}}
                    </div>
                    <div class="text-xs">No Connection</div>
                  </div>
                  <div class="bg-gray-700 p-2 rounded text-center">
                    <div class="font-bold text-gray-400">
                      {{pathDebugInfo.attemptedPairs.filter(p =>
                        p.result!.type !== PathfindingResult_ResultType.SUCCESS &&
                        p.result!.type !== PathfindingResult_ResultType.TOO_FAR &&
                        p.result!.type !== PathfindingResult_ResultType.NO_CONNECTION
                      ).length}}
                    </div>
                    <div class="text-xs">Other</div>
                  </div>
                </div>
              </div>

              <div v-if="pathDebugInfo.constraints && pathDebugInfo.constraints.length > 0" class="mb-4">
                <div class="font-medium mb-1">Constraints:</div>
                <div class="text-sm">
                  <div v-for="(constraint, i) in pathDebugInfo.constraints" :key="i" class="mb-1">
                    Point {{ constraint.pointIdx }} → Segment {{ constraint.segmentId }}
                  </div>
                </div>
              </div>

              <button class="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded text-sm mb-2"
                @click="visualizeConnectivity">
                Visualize Connectivity
              </button>

              <div class="path-attempts overflow-y-auto max-h-60">
                <div class="font-medium mb-1">Path Attempts:</div>
                <div v-for="(attempt, i) in pathDebugInfo.attemptedPairs.slice(0, 10)" :key="i"
                  :class="`attempt p-2 mb-2 rounded ${getAttemptClass(attempt)}`">
                  <div class="flex justify-between">
                    <span>{{ getSegmentLabel(Number(attempt.fromSegment)) }} → {{
                      getSegmentLabel(Number(attempt.toSegment))
                      }}</span>
                    <span class="font-medium">{{ getResultLabel(attempt.result!) }}</span>
                  </div>
                  <div class="text-xs">Distance: {{ attempt.distance.toFixed(2) }}m</div>

                  <div v-if="attempt.result!.type === PathfindingResult_ResultType.TOO_FAR">
                    <div class="text-xs">
                      Max allowed: {{ attempt.result?.maxDistance.toFixed(2) }}m
                    </div>
                  </div>

                  <div v-if="attempt.result!.type === PathfindingResult_ResultType.NO_PATH_FOUND">
                    <div class="text-xs text-red-400">
                      Error: {{ attempt.result?.reason }}
                    </div>
                  </div>
                </div>

                <div v-if="pathDebugInfo.attemptedPairs.length > 10" class="text-sm text-gray-400">
                  ...and {{ pathDebugInfo.attemptedPairs.length - 10 }} more attempts
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="segment-list p-4">
          <h3 class="text-lg font-bold border-b border-gray-700 pb-2 mb-4">Segments</h3>
          <div v-for="segment in selectedWindow.segments" :key="segment.id.toString()"
            class="segment-item bg-gray-700 mb-3 rounded p-3 cursor-pointer transition-all" :class="{
              'border-l-4 border-blue-500': highlightedSegment === segment,
              'transform translate-x-1': hoveredSegment === segment,
              'border-l-4 border-indigo-300': segment.interimStartIdx !== undefined || segment.interimEndIdx !== undefined
            }" @click="selectSegment(segment)" @mouseenter="hoverSegment(segment)" @mouseleave="clearHoveredSegment()">
            <div class="segment-name font-bold mb-1">{{ segment.name || 'Unnamed Road' }}</div>
            <div class="segment-details flex flex-wrap gap-1 text-xs">
              <span class="bg-gray-800 px-2 py-1 rounded">{{ segment.highwayType }}</span>
              <span v-if="segment.isOneway" class="bg-yellow-800 px-2 py-1 rounded">One-way</span>
              <span class="bg-gray-800 px-2 py-1 rounded">OSM ID: {{ segment.osmWayId.toString() }}</span>
              <span v-if="segment.interimStartIdx !== undefined || segment.interimEndIdx !== undefined"
                class="bg-indigo-800 px-2 py-1 rounded">
                Partial: {{ segment.interimStartIdx || 0 }}-{{ segment.interimEndIdx !== undefined ?
                  segment.interimEndIdx : (segment.fullCoordinates ? segment.fullCoordinates.length - 1 : '?') }}
              </span>
            </div>
          </div>
        </div>
      </div>

      <!-- Map view -->
      <div class="map-container flex-1 w-full h-full">
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
            <MglMarker v-for="(point, index) in routeMatchTrace?.trip?.points" :key="Number(index)"
              :coordinates="[point.latlon!.lon, point.latlon!.lat]">
              <template #marker>
                <div
                  class="point-marker w-5 h-5 flex items-center justify-center rounded-full bg-gray-500 text-white text-xs font-bold border-2 border-gray-700 transition-all"
                  :class="{
                    'bg-green-500 transform scale-125 z-10': selectedWindow && index === selectedWindow.start,
                    'bg-red-500 transform scale-125 z-10': selectedWindow && index === selectedWindow.end,
                    'bg-blue-500 border-white': selectedWindow && index >= selectedWindow.start && index <= selectedWindow.end,
                    'bg-pink-500 border-white transform scale-110 z-9': selectedWindow && selectedWindow.constraints &&
                      selectedWindow.constraints.some(c => c.pointIdx === index)
                  }">
                  {{ index }}
                </div>
              </template>
              <MglPopup ref="popup">
                <div class="popup-content p-2 text-gray-800">
                  <h3 class="font-bold mb-1">Point {{ index }}</h3>
                  <p class="mb-2">Coordinates: {{ point.latlon?.lat.toFixed(6) }}, {{ point.latlon?.lon.toFixed(6) }}
                  </p>

                  <!-- Show constraint info in popup if this point is constrained -->
                  <div v-if="selectedWindow && selectedWindow.constraints &&
                    selectedWindow.constraints.some(c => c.pointIdx === index)"
                    class="constraint-info bg-pink-100 p-2 rounded border-l-3 border-pink-500 mb-2">
                    <h4 class="font-bold text-pink-800 mb-1">Constraint Info:</h4>
                    <div v-for="(constraint, cIdx) in selectedWindow.constraints.filter(c => c.pointIdx === index)"
                      :key="cIdx" class="constraint-detail mb-1">
                      <p class="text-sm">Segment: {{ constraint.segmentId }} (Way: {{ constraint.wayId }})</p>
                      <p class="text-sm">Distance: {{ constraint.distance.toFixed(2) }}m</p>
                    </div>
                  </div>

                  <div class="popup-actions flex gap-2">
                    <button class="bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded text-xs"
                      @click="debugGeojsonSource = JSON.parse(routeMatchTrace!.pointCandidates[index].value) as GeoJSON<Geometry, GeoJsonProperties>">
                      Show Candidates
                    </button>
                    <button class="bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded text-xs"
                      @click="debugOSMAt(point)">
                      OSM Around Point
                    </button>
                  </div>
                  <a href="#" class="block text-right text-blue-500 text-sm mt-2" @click.prevent="closePopup">Close</a>
                </div>
              </MglPopup>
            </MglMarker>
          </MglMap>
        </ClientOnly>
      </div>
    </div>

    <div class="legend absolute bottom-5 right-5 bg-gray-800 bg-opacity-80 p-3 rounded-lg z-10 max-w-xs">
      <h3 class="text-white font-bold mb-2">Legend</h3>
      <div class="legend-item flex items-center mb-1">
        <div class="legend-color constrained w-7 h-1 bg-green-500 mr-2" />
        <div class="legend-label text-white text-sm">Constrained Route</div>
      </div>
      <div class="legend-item flex items-center mb-1">
        <div class="legend-color unconstrained w-7 h-1 bg-blue-500 mr-2" />
        <div class="legend-label text-white text-sm">Unconstrained Route</div>
      </div>
      <div class="legend-item flex items-center mb-1">
        <div class="legend-color unconstrained-bridge w-7 h-1 border-t border-dashed border-white bg-red-500 mr-2" />
        <div class="legend-label text-white text-sm">Unconstrained Bridge Route</div>
      </div>
      <div class="legend-item flex items-center mb-1">
        <div
          class="legend-point constrained-point-legend w-4 h-4 rounded-full bg-pink-500 border-2 border-white mr-2" />
        <div class="legend-label text-white text-sm">Constrained Point</div>
      </div>
      <div class="legend-item flex items-center mb-1">
        <div class="legend-point start-point-legend w-4 h-4 rounded-full bg-green-500 border-2 border-gray-700 mr-2" />
        <div class="legend-label text-white text-sm">Start Point</div>
      </div>
      <div class="legend-item flex items-center mb-1">
        <div class="legend-point end-point-legend w-4 h-4 rounded-full bg-red-500 border-2 border-gray-700 mr-2" />
        <div class="legend-label text-white text-sm">End Point</div>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { type PathfindingResult, type PathfindingAttempt, type Point, type WindowTrace, type PathfindingDebugInfo, PathfindingResult_ResultType } from '~/model/chronotopia_pb';
import type { Map } from 'maplibre-gl';
import { MglFullscreenControl, MglGeolocateControl, MglMap, MglMarker, MglNavigationControl, MglGeoJsonSource, MglLineLayer } from '#components';
import type { GeoJSON, GeoJsonProperties, Geometry } from 'geojson';
import { useRoute, useRouter } from 'vue-router';
import type { RouteMatchTrace } from '~/model/chronotopia_pb';

const { $api } = useNuxtApp();
const route = useRoute();
const router = useRouter();
const tripIndex = parseInt(route.params.trip as string);

const routeMatchTrace: Ref<RouteMatchTrace | null> = ref(null);
const selectedWindowIndex: Ref<number | null> = ref(null);
const selectedWindow: Ref<WindowTrace | null> = ref(null);
const highlightedSegment = ref(null);
const showConstraints = ref(false);
const showDebugInfo = ref(false);
const pathDebugInfo: Ref<PathfindingDebugInfo | null> = ref(null);
const map = useMglMap();
const mmap = map.map as Map;

// Fetch the route match trace for the specified trip
onMounted(async () => {
  try {
    const response = await $api.getRouteMatchTrace({ tripIndex });
    routeMatchTrace.value = response;
  } catch (error) {
    console.error('Error fetching route match trace:', error);
  }
});

const goBack = () => {
  router.push('/');
};

const selectWindow = (index: number) => {
  selectedWindowIndex.value = index;
  selectedWindow.value = routeMatchTrace.value?.windowTraces[index];
}

const selectSegment = (segment) => {
  highlightedSegment.value = segment;
}

const hoveredSegment = ref(null);

const hoverSegment = (segment) => {
  hoveredSegment.value = segment;
}

const clearHoveredSegment = () => {
  hoveredSegment.value = null;
}

const toggleConstraints = () => {
  showConstraints.value = !showConstraints.value;

  // Toggle visibility of constraint layers
  routeMatchTrace.value?.windowTraces.forEach((window, idx) => {
    const constraintLayerId = `window-${idx}-constraints`;
    if (mmap.getLayer(constraintLayerId)) {
      mmap.setLayoutProperty(
        constraintLayerId,
        'visibility',
        showConstraints.value ? 'visible' : 'none'
      );
    }

    // Also toggle constraint point layers
    const constraintPointsId = `window-${idx}-constraint-points`;
    if (mmap.getLayer(constraintPointsId)) {
      mmap.setLayoutProperty(
        constraintPointsId,
        'visibility',
        showConstraints.value ? 'visible' : 'none'
      );
    }
  });
}

const toggleDebugInfo = () => {
  showDebugInfo.value = !showDebugInfo.value;
}

const debugGeojsonSource = ref(undefined);

// Utility function to calculate segment centroid
const calculateSegmentCentroid = (segment) => {
  if (!segment.coordinates || segment.coordinates.length === 0) {
    return [0, 0];
  }

  const coords = segment.coordinates;
  let sumLon = 0;
  let sumLat = 0;

  for (const coord of coords) {
    sumLon += coord.lon;
    sumLat += coord.lat;
  }

  return [sumLon / coords.length, sumLat / coords.length];
};

// Calculate window statistics
const getWindowStats = (window: WindowTrace) => {
  if (!window) return null;

  // Calculate total distance (very rough approximation)
  let totalDistance = 0;
  const totalSegments = window.segments.length;
  const highwayTypes = new Set();

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

  // Count unique attempted way IDs
  const attemptedWayIds = window.attemptedWayIds ? window.attemptedWayIds.length : 0;

  return {
    segmentCount: totalSegments,
    distance: totalDistance.toFixed(2), // in km
    highwayTypes: Array.from(highwayTypes),
    isBridge: window.bridge ? 'Yes' : 'No',
    pointRange: `${window.start} - ${window.end}`,
    constraintsCount: window.constraints ? window.constraints.length : 0,
    usedConstraints: window.usedConstraints ? 'Yes' : 'No',
    constraintScore: window.constraintScore !== undefined ? window.constraintScore.toFixed(2) : 'N/A',
    unconstrainedScore: window.unconstrainedScore !== undefined ? window.unconstrainedScore.toFixed(2) : 'N/A',
    attemptedWayIds: attemptedWayIds
  };
};

// Get colors for the window selector
const getWindowColor = (index: number) => {
  const window = routeMatchTrace.value!.windowTraces[index]!;

  // Base color calculated from index
  const hueStep = 255 / routeMatchTrace.value!.windowTraces.length;
  const baseHue = Math.round(hueStep * index);

  // Modify saturation based on constraint usage
  const saturation = window.usedConstraints ? 80 : 50;

  // Modify lightness based on bridge status
  const lightness = window.bridge ? 65 : 50;

  return `hsl(${baseHue}, ${saturation}%, ${lightness}%)`;
};

// Get the window background class based on its properties
const getWindowClass = (window: WindowTrace) => {
  if (window.usedConstraints) return 'bg-green-500';
  if (window.bridge) return 'bg-red-500';
  return 'bg-blue-500';
};

const popupRef = useTemplateRef('popup')
const closePopup = () => {
  popupRef.value.remove();
};

const debugOSMAt = async (point: Point) => {
  const m = await $api.oSMNetworkAroundPoint(point.latlon);
  const gj = JSON.parse(m.value);
  debugGeojsonSource.value = gj;
}

const debugFailedWindow = async () => {
  if (!selectedWindow.value || !selectedWindow.value.bridge) return;

  // Call the backend to get debugging information for this window
  const failedWindowData = await $api.debugWindowPathFinding({
    tripIndex: tripIndex,
    windowIndex: selectedWindowIndex.value!,
    startPoint: selectedWindow.value.start,
    endPoint: selectedWindow.value.end
  });

  // Update the state with the debugging info
  pathDebugInfo.value = failedWindowData;

  // Log for debugging
  console.log("Path debug info:", pathDebugInfo.value);
};

// Add this method to visualize connectivity between segments
const visualizeConnectivity = async () => {
  if (!selectedWindow.value || !pathDebugInfo.value) return;

  // Get connectivity visualization
  const connectivityData = await $api.analyzeSegmentConnectivity({
    tripIndex: tripIndex,
    startPointIndex: selectedWindow.value.start,
    endPointIndex: selectedWindow.value.end
  });

  // Update the debug GeoJSON source to show the connectivity
  debugGeojsonSource.value = JSON.parse(connectivityData.value);
};

// Helper methods
const getAttemptClass = (attempt: PathfindingAttempt) => {
  switch (attempt.result!.type) {
    case PathfindingResult_ResultType.SUCCESS: return 'bg-green-100';
    case PathfindingResult_ResultType.TOO_FAR: return 'bg-yellow-100';
    case PathfindingResult_ResultType.NO_CONNECTION: return 'bg-red-100';
    default: return 'bg-gray-100';
  }
};

const getResultLabel = (result: PathfindingResult) => {
  switch (result.type) {
    case PathfindingResult_ResultType.SUCCESS: return 'Success';
    case PathfindingResult_ResultType.TOO_FAR: return 'Too Far';
    case PathfindingResult_ResultType.NO_CONNECTION: return 'No Connection';
    case PathfindingResult_ResultType.NO_PATH_FOUND: return 'Path Error';
    default: return 'Unknown';
  }
};

const getSegmentLabel = (segmentId: number) => {
  // Find the segment in the current window
  const segment = selectedWindow.value?.segments.find(s => Number(s.id) == segmentId);
  if (segment) {
    return `Seg ${segmentId} (Way ${segment.osmWayId})`;
  }
  return `Seg ${segmentId}`;
};
</script>

<style lang="css">
html,
body,
div#__nuxt {
  height: 100%;
  width: 100%;
}
</style>