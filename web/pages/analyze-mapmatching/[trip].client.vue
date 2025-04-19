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
        <button v-if="tripData" class="bg-green-600 hover:bg-green-500 px-4 py-2 rounded" @click="reprocessTrip">
          Reprocess Trip
        </button>
        <button class="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded" @click="loadInRouteplanner">
          Load in routeplanner
        </button>
        <button class="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded" @click="goBack">
          Back to Trip
        </button>
      </div>
    </div>

    <!-- Processing status notification -->
    <div v-if="processingStatus.isProcessing" class="bg-blue-600 text-white p-3 flex justify-between items-center">
      <div class="flex items-center">
        <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none"
          viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
        </svg>
        <span>{{ processingStatus.message }}</span>
      </div>
      <button v-if="processingStatus.canCancel"
        class="bg-white bg-opacity-20 hover:bg-opacity-30 px-3 py-1 rounded text-sm" @click="cancelProcessing">
        Cancel
      </button>
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
      <div v-if="selectedWindow" class="info-panel w-full md:w-96 bg-gray-800 text-white overflow-y-auto">
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
                      {{pathDebugInfo.attemptedPairs.filter(p => p.result?.type ===
                        PathfindingResult_ResultType.SUCCESS).length}}
                    </div>
                    <div class="text-xs">Successful</div>
                  </div>
                  <div class="bg-yellow-900 bg-opacity-40 p-2 rounded text-center">
                    <div class="font-bold text-yellow-400">
                      {{pathDebugInfo.attemptedPairs.filter(p => p.result?.type ===
                        PathfindingResult_ResultType.TOO_FAR).length}}
                    </div>
                    <div class="text-xs">Too Far</div>
                  </div>
                  <div class="bg-red-900 bg-opacity-40 p-2 rounded text-center">
                    <div class="font-bold text-red-400">
                      {{pathDebugInfo.attemptedPairs.filter(p => p.result?.type ===
                        PathfindingResult_ResultType.NO_CONNECTION).length}}
                    </div>
                    <div class="text-xs">No Connection</div>
                  </div>
                  <div class="bg-gray-700 p-2 rounded text-center">
                    <div class="font-bold text-gray-400">
                      {{pathDebugInfo.attemptedPairs.filter(p =>
                        p.result?.type !== PathfindingResult_ResultType.SUCCESS &&
                        p.result?.type !== PathfindingResult_ResultType.TOO_FAR &&
                        p.result?.type !== PathfindingResult_ResultType.NO_CONNECTION
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

              <div class="flex gap-2 mb-2">
                <button class="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded text-sm"
                  @click="visualizeConnectivity">
                  Visualize Connectivity
                </button>
                <button class="bg-indigo-500 hover:bg-indigo-600 text-white px-3 py-1 rounded text-sm"
                  @click="analyzeSegmentConnectivity">
                  Analyze Connectivity (Direct)
                </button>
              </div>

              <div class="path-attempts overflow-y-auto max-h-60">
                <div class="font-medium mb-1">Path Attempts:</div>
                <div v-for="(attempt, i) in pathDebugInfo.attemptedPairs.slice(0, 10)" :key="i"
                  :class="`attempt p-2 mb-2 rounded ${getAttemptClass(attempt as PathfindingAttempt)}`">
                  <div class="flex justify-between">
                    <span>{{ getSegmentLabel(Number(attempt.fromSegment)) }} → {{
                      getSegmentLabel(Number(attempt.toSegment))
                    }}</span>
                    <span class="font-medium">{{ getResultLabel(attempt.result! as PathfindingResult) }}</span>
                  </div>
                  <div class="text-xs">Distance: {{ attempt.distance.toFixed(2) }}m</div>

                  <div v-if="attempt.result?.type === PathfindingResult_ResultType.TOO_FAR">
                    <div class="text-xs">
                      Max allowed: {{ attempt.result?.maxDistance.toFixed(2) }}m
                    </div>
                  </div>

                  <div v-if="attempt.result?.type === PathfindingResult_ResultType.NO_PATH_FOUND">
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
              <span class="bg-gray-800 px-2 py-1 rounded">Segment ID: {{ segment.id.toString() }}</span>
              <span v-if="segment.interimStartIdx !== undefined || segment.interimEndIdx !== undefined"
                class="bg-indigo-800 px-2 py-1 rounded">
                Partial: {{ segment.interimStartIdx || 0 }}-{{ segment.interimEndIdx !== undefined ?
                  segment.interimEndIdx : (segment.fullCoordinates ? segment.fullCoordinates.length - 1 : '?') }}
              </span>
            </div>
            <div v-if="showDebugInfo" class="segment-actions mt-2 flex gap-1">
              <button class="bg-blue-600 hover:bg-blue-700 text-white px-2 py-1 rounded text-xs"
                @click.stop="debugSegmentConnections(segment)">
                Debug Connections
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Map view -->
      <div class="map-container flex-1 w-full h-full">
        <ClientOnly>
          <MglMap map-style="https://api.maptiler.com/maps/streets/style.json?key=Ic6Mr5qetb5kn90hyEzO" :zoom="6">
            <MglFullscreenControl position="top-right" />
            <MglNavigationControl position="top-right" />
            <MglGeolocateControl position="top-right" />
            <MglGeoJsonSource v-if="debugGeojsonSource" source-id="debug" :data="debugGeojsonSource">
              <MglLineLayer layer-id="debug-lines" source="debug" :layer="debugLayerConfig.lines" />
              <MglCircleLayer layer-id="debug-points" source="debug" :layer="debugLayerConfig.points" />
            </MglGeoJsonSource>

            <!-- Trip points -->
            <MglMarker v-for="(point, index) in tripData?.points" :key="Number(index)"
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
                  <p v-if="point.dateTime" class="mb-2">Time: {{ formatDateTime(point.dateTime) }}</p>

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
                      @click="showPointCandidates(index)">
                      Show Candidates
                    </button>
                    <button class="bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded text-xs"
                      @click="debugOSMAt(point)">
                      OSM Around Point
                    </button>
                  </div>
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

    <!-- Status Toast -->
    <div v-if="statusNotification.show"
      class="fixed bottom-5 left-5 p-4 rounded-lg shadow-lg z-50 text-white flex items-center"
      :class="statusNotification.type === 'success' ? 'bg-green-600' : 'bg-red-600'">
      <span v-if="statusNotification.type === 'success'" class="mr-2">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
        </svg>
      </span>
      <span v-else class="mr-2">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </span>
      {{ statusNotification.message }}
    </div>
  </div>
</template>

<script lang="ts" setup>
import { LngLatBounds, Popup, type LineLayerSpecification, type LngLatLike, type Map, type CircleLayerSpecification } from 'maplibre-gl';
import { MglFullscreenControl, MglGeolocateControl, MglMap, MglMarker, MglNavigationControl, MglGeoJsonSource, MglLineLayer, MglCircleLayer, MglPopup } from '#components';
import type { Feature, GeoJSON } from 'geojson';
import { useRoute, useRouter } from 'vue-router';
import type { RouteMatchTrace, Trip, WindowTrace } from '~/model/trips_pb';
import type { DateTime, Point, RoadSegment } from '~/model/common_pb';
import {
  ConnectivityRequest,
  PathfindingResult_ResultType,
  type PathfindingDebugInfo,
  type PathfindingAttempt,
  type PathfindingResult,
  WindowDebugRequest
} from '~/model/chronotopia_pb';

const { api, tripsApi } = useApi();
const route = useRoute();
const router = useRouter();
const tripId = route.params.trip as string;
const routeMatchTrace: Ref<RouteMatchTrace | null> = ref(null);
const tripData: Ref<Trip | undefined> = ref(undefined);
const selectedWindowIndex = ref<number | null>(null);
const selectedWindow: Ref<WindowTrace | undefined> = ref(undefined);
const debugGeojsonSource = ref<GeoJSON | undefined>(undefined);
const highlightedSegment = ref<RoadSegment | null>(null);
const hoveredSegment = ref<RoadSegment | null>(null);
const showConstraints = ref(false);
const showDebugInfo = ref(false);
const pathDebugInfo = ref<PathfindingDebugInfo | null>(null);

// Processing status
const processingStatus = ref({
  isProcessing: false,
  message: '',
  canCancel: false,
});

// Status notification
const statusNotification = ref({
  show: false,
  message: '',
  type: 'success',
});

// Config for debug layer visualization
const debugLayerConfig = {
  lines: {
    type: 'line',
    paint: {
      'line-color': ['case',
        ['==', ['get', 'type'], 'connection'], '#00FF00',
        ['==', ['get', 'type'], 'no-connection'], '#FF0000',
        ['==', ['get', 'type'], 'too-far'], '#FFAA00',
        '#3388FF'
      ],
      'line-width': 3,
      'line-opacity': 0.8,
      'line-dasharray': ['case',
        ['==', ['get', 'type'], 'candidate'], [1, 1],
        [1, 0]
      ]
    },
    layout: {
      'line-join': 'round',
      'line-cap': 'round'
    },
  } as unknown as LineLayerSpecification,
  points: {
    type: 'circle',
    paint: {
      'circle-radius': ['case',
        ['==', ['get', 'type'], 'segment-node'], 5,
        ['==', ['get', 'type'], 'candidate-point'], 4,
        3
      ],
      'circle-color': ['case',
        ['==', ['get', 'type'], 'segment-node'], '#0088FF',
        ['==', ['get', 'type'], 'candidate-point'], '#FF00FF',
        '#888888'
      ],
      'circle-opacity': 0.7,
      'circle-stroke-width': 1,
      'circle-stroke-color': '#FFFFFF'
    },
    filter: ['==', ['geometry-type'], 'Point']
  } as unknown as CircleLayerSpecification
};

const map = useMglMap();
const layers: string[] = [];
const layout = {
  'line-join': 'round',
  'line-cap': 'round'
} as LineLayerSpecification['layout'];

// Use the tripsApi pattern to fetch the trip details first
useAsyncData(
  'tripDetail',
  async () => {
    try {
      processingStatus.value.isProcessing = true;
      processingStatus.value.message = 'Loading trip data...';

      const response = await tripsApi.getTripDetails({
        tripId: {
          value: tripId
        }
      });

      tripData.value = response.trip;

      // Set the routeMatchTrace from the trip details
      if (response.trip?.routeMatchTrace) {
        routeMatchTrace.value = response.trip.routeMatchTrace;
      } else {
        showNotification('No route match trace available for this trip', 'error');
      }

      processingStatus.value.isProcessing = false;
      return response.trip;
    } catch (err) {
      processingStatus.value.isProcessing = false;
      console.error(`Error fetching trip ${tripId}:`, err);
      showNotification('Failed to load trip details', 'error');
      throw new Error('Failed to load trip details');
    }
  }
);

// Watch for when the map and data are loaded
watch(() => [map.isLoaded, routeMatchTrace.value], ([isLoaded, trace]) => {
  if (isLoaded && trace) {
    setupMapLayers();
  }
}, { immediate: true });

// Setup map layers once both map and data are loaded
const setupMapLayers = () => {
  if (!map.map || !routeMatchTrace.value) return;

  const mmap = map.map as Map;

  // Clear any existing layers
  layers.forEach(layerId => {
    if (mmap.getLayer(layerId)) {
      mmap.removeLayer(layerId);
    }
    if (mmap.getSource(layerId)) {
      mmap.removeSource(layerId);
    }
  });
  layers.length = 0;

  // Add layers for each window
  for (const windowIdx in routeMatchTrace.value.windowTraces) {
    const window = routeMatchTrace.value.windowTraces[windowIdx];
    const sourceId = `window-${windowIdx}`;
    layers.push(sourceId);

    // Create the main segment geojson
    const geojson: GeoJSON = {
      type: 'FeatureCollection',
      features: window.segments.map((segment, segIdx) => {
        // Extract coordinates for the feature
        let coordinates = [];

        // Handle cases where interimStartIdx > interimEndIdx (reversed segment)
        const isReversedSegment = segment.interimStartIdx !== undefined &&
          segment.interimEndIdx !== undefined &&
          segment.interimStartIdx > segment.interimEndIdx;

        if (isReversedSegment && segment.fullCoordinates) {
          // For reversed segments, we need to extract the coordinates in reverse
          const startIdx = segment.interimStartIdx!;
          const endIdx = segment.interimEndIdx!;

          // Get coordinates from full coordinates in reverse order
          const extractedCoords = [];
          for (let i = startIdx; i >= endIdx; i--) {
            if (i < segment.fullCoordinates.length) {
              extractedCoords.push([segment.fullCoordinates[i].lon,
              segment.fullCoordinates[i].lat]);
            }
          }

          coordinates = extractedCoords;
        } else { // Normal case: use coordinates as provided
          coordinates = segment.coordinates.map(v => [v.lon, v.lat]);
        }

        // Create properties object with all relevant info
        const properties = {
          window: windowIdx,
          segmentId: segment.id.toString(),
          osmWayId: segment.osmWayId.toString(),
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
          isReversedSegment: isReversedSegment,
          // Add constraint status
          isConstrained: window.usedConstraints,
          isBridge: window.bridge
        };

        return {
          type: 'Feature',
          properties: properties,
          geometry: {
            type: 'LineString',
            coordinates: coordinates
          }
        } as Feature;
      })
    };

    // Add source
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
        'line-color': getWindowColor(Number(windowIdx)),
        'line-width': 8,
        // Add dashed line for bridges (unconstrained paths)
        'line-dasharray': window.bridge ? [2, 1] : [1, 0]
      } as LineLayerSpecification['paint']
    });

    // For segments with interim indices, also add full segment with low opacity
    // to provide context
    if (window.segments.some(s => s.interimStartIdx !== undefined || s.interimEndIdx !== undefined)) {
      const fullSegmentSourceId = `window-${windowIdx}-full`;
      layers.push(fullSegmentSourceId);

      const fullSegmentsGeoJSON: GeoJSON = {
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
          } as Feature;
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
            'line-color': getWindowColor(Number(windowIdx)),
            'line-width': 4,
            'line-opacity': 0.3,
            'line-dasharray': [2, 2]
          } as LineLayerSpecification['paint']
        });
      }
    }

    // Add constraint visualization
    if (window.constraints && window.constraints.length > 0) {
      // Create a source for constraint connections
      const constraintSourceId = `window-${windowIdx}-constraints`;
      const constraintPointsId = `window-${windowIdx}-constraint-points`;
      layers.push(constraintSourceId, constraintPointsId);

      // Create constraint connections
      const constraintFeatures = window.constraints.map(constraint => {
        // Get the point coordinates
        const point = tripData.value?.points[constraint.pointIdx];
        if (!point || !point.latlon) return null;

        // Find the segment
        const segment = window.segments.find(s => s.id === constraint.segmentId);
        if (!segment) return null;

        // Create a line from point to the segment centroid
        const segmentCentroid = calculateSegmentCentroid(segment);

        return {
          type: 'Feature',
          properties: {
            pointIdx: constraint.pointIdx,
            segmentId: constraint.segmentId.toString(),
            wayId: constraint.wayId.toString(),
            distance: constraint.distance,
            description: `Constraint: Point ${constraint.pointIdx} to Segment ${constraint.segmentId} (Way ${constraint.wayId}),
  Distance: ${constraint.distance.toFixed(2)}m`
          },
          geometry: {
            type: 'LineString',
            coordinates: [
              [point.latlon.lon, point.latlon.lat],
              segmentCentroid
            ]
          }
        } as Feature;
      }).filter(f => f !== null) as Feature[];

      const constraintPointsFeatures = window.constraints.map(constraint => {
        const point = tripData.value?.points[constraint.pointIdx];
        if (!point || !point.latlon) return null;

        return {
          type: 'Feature',
          properties: {
            pointIdx: constraint.pointIdx,
            segmentId: constraint.segmentId.toString(),
            wayId: constraint.wayId.toString(),
            description: `Constrained Point ${constraint.pointIdx}`
          },
          geometry: {
            type: 'Point',
            coordinates: [point.latlon.lon, point.latlon.lat]
          }
        } as Feature;
      }).filter(f => f !== null) as Feature[];

      // Add constraint connections source and layer
      mmap.addSource(constraintSourceId, {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: constraintFeatures
        }
      });

      mmap.addLayer({
        id: constraintSourceId,
        type: 'line',
        source: constraintSourceId,
        layout: {
          'line-join': 'round',
          'line-cap': 'round',
          'visibility': 'none' // Hidden by default
        },
        paint: {
          'line-color': '#FF00FF',
          'line-width': 3,
          'line-dasharray': [2, 1]
        } as LineLayerSpecification['paint']
      });

      // Add constraint points source and layer
      mmap.addSource(constraintPointsId, {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: constraintPointsFeatures
        }
      });

      mmap.addLayer({
        id: constraintPointsId,
        type: 'circle',
        source: constraintPointsId,
        layout: {
          'visibility': 'none' // Hidden by default
        },
        paint: {
          'circle-radius': 8,
          'circle-color': '#FF00FF',
          'circle-stroke-width': 2,
          'circle-stroke-color': '#FFFFFF'
        }
      });

      // Popup for constraints
      mmap.on('click', constraintSourceId, (e) => {
        if (e.features && e.features.length > 0) {
          const coordinates = e.lngLat;
          const feature = e.features[0];

          new Popup()
            .setLngLat(coordinates)
            .setHTML(`
  <div class="constraint-popup">
                <h4>Constraint Details</h4>
                <p>${feature.properties.description}</p>
              </div>
  `)
            .addTo(mmap);
        }
      });
    }

    // Add a highlight outline layer that will be visible when a segment is hovered
    const highlightLayerId = `${sourceId}-highlight`;
    layers.push(highlightLayerId);

    mmap.addLayer({
      id: highlightLayerId,
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

    // Setup event handlers for segments
    setupSegmentEvents(mmap, popup);
  }

  // Fit bounds to show all segments
  fitMapBoundsToTrip();

  // Select the first window by default
  if (routeMatchTrace.value.windowTraces.length > 0) {
    selectWindow(0);
  }
};

// Setup mouse events for segments
const setupSegmentEvents = (mmap: Map, popup: Popup) => {
  // Make sure to detect marker change for overlapping markers
  // and use mousemove instead of mouseenter event
  let currentFeatureCoordinates = "";

  mmap.on('mousemove', layers.filter(id => !id.includes('constraints') && !id.includes('-full') &&
    !id.includes('-highlight')), (e) => {
      if (!e.features || e.features.length === 0) return;

      const featureCoordinates = e.lngLat.toString();
      if (currentFeatureCoordinates !== featureCoordinates) {
        currentFeatureCoordinates = featureCoordinates;

        // Change the cursor style as a UI indicator
        mmap.getCanvas().style.cursor = 'pointer';

        const coordinates = e.lngLat;
        const feature = e.features[0];
        const description = `
  <strong>Window:</strong> ${feature.properties.window}<br>
  <strong>Road:</strong> ${feature.properties.name}<br>
  <strong>Type:</strong> ${feature.properties.highwayType}<br>
  <strong>OSM Way ID:</strong> ${feature.properties.osmWayId}<br>
  <strong>Segment ID:</strong> ${feature.properties.segmentId}<br>
  <strong>One-way:</strong> ${feature.properties.isOneway ? 'Yes' : 'No'}<br>
  <strong>Constraints:</strong> ${feature.properties.isConstrained ? 'Used' : 'Not used'}<br>
  <strong>Bridge:</strong> ${feature.properties.isBridge ? 'Yes (unconstrained)' : 'No'}<br>
  ${feature.properties.isPartialSegment ?
            `<strong>Partial segment:</strong> Using ${feature.properties.interimStartIdx}-${feature.properties.interimEndIdx}`
            : ''}
  `;

        // Populate the popup and set its coordinates
        popup.setLngLat(coordinates).setHTML(description).addTo(mmap);
      }
    });

  mmap.on('mouseleave', layers.filter(id => !id.includes('constraints') && !id.includes('-full') &&
    !id.includes('-highlight')), () => {
      currentFeatureCoordinates = "";
      mmap.getCanvas().style.cursor = '';
      popup.remove();
    });

  // Add click handlers for segments
  mmap.on('click', layers.filter(id => !id.includes('constraints') && !id.includes('-full') &&
    !id.includes('-highlight')), (e) => {
      if (e.features && e.features.length > 0) {
        const windowIndex = parseInt(e.features[0].properties.window);
        const segmentIndex = e.features[0].properties.segmentIndex;

        selectWindow(windowIndex);
        if (routeMatchTrace.value) {
          selectSegment(routeMatchTrace.value.windowTraces[windowIndex].segments[segmentIndex]);
        }

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
};

const fitMapBoundsToTrip = () => {
  if (!map.map || !tripData.value?.points?.length) return;

  const bounds = new LngLatBounds();

  // Add all points to bounds
  for (const point of tripData.value.points) {
    if (point.latlon) {
      bounds.extend([point.latlon.lon, point.latlon.lat]);
    }
  }

  // Apply padding and fly to bounds
  if (!bounds.isEmpty()) {
    map.map.fitBounds(bounds, {
      padding: 50,
      maxZoom: 14
    });
  }
};

const goBack = () => {
  router.push(`/trips/${tripId}`);
};

const loadInRouteplanner = () => {
  router.push(`routeplanner/${tripId}`);
};

// Select a window for detailed inspection
const selectWindow = (index: number) => {
  selectedWindowIndex.value = index;
  if (routeMatchTrace.value) {
    selectedWindow.value = routeMatchTrace.value.windowTraces[index];
  }
}

const selectSegment = (segment: RoadSegment) => {
  highlightedSegment.value = segment;
}

const hoverSegment = (segment: RoadSegment) => {
  hoveredSegment.value = segment;
}

const clearHoveredSegment = () => {
  hoveredSegment.value = null;
}

// Toggle constraint visibility
const toggleConstraints = () => {
  showConstraints.value = !showConstraints.value;

  // Toggle visibility of constraint layers
  if (!map.map || !routeMatchTrace.value) return;

  routeMatchTrace.value.windowTraces.forEach((window, idx) => {
    const constraintLayerId = `window-${idx}-constraints`;
    if (map.map.getLayer(constraintLayerId)) {
      map.map.setLayoutProperty(
        constraintLayerId,
        'visibility',
        showConstraints.value ? 'visible' : 'none'
      );
    }

    // Also toggle constraint point layers
    const constraintPointsId = `window-${idx}-constraint-points`;
    if (map.map.getLayer(constraintPointsId)) {
      map.map.setLayoutProperty(
        constraintPointsId,
        'visibility',
        showConstraints.value ? 'visible' : 'none'
      );
    }
  });
}

// Toggle debug information visibility
const toggleDebugInfo = () => {
  showDebugInfo.value = !showDebugInfo.value;
}

// Show notification toast
const showNotification = (message: string, type: 'success' | 'error') => {
  statusNotification.value = {
    show: true,
    message,
    type
  };

  // Hide after 5 seconds
  setTimeout(() => {
    statusNotification.value.show = false;
  }, 5000);
};

// Utility function to calculate segment centroid
const calculateSegmentCentroid = (segment: RoadSegment): [number, number] => {
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
}

// Get colors for the window selector
const getWindowColor = (index: number) => {
  if (!routeMatchTrace.value) return 'hsl(200, 50%, 50%)';

  const window = routeMatchTrace.value.windowTraces[index];

  // Base color calculated from index
  const hueStep = 255 / routeMatchTrace.value.windowTraces.length;
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

// Debug OSM around a specific point
const debugOSMAt = async (point: Point) => {
  try {
    processingStatus.value.isProcessing = true;
    processingStatus.value.message = 'Loading OSM network data...';

    const response = await api.oSMNetworkAroundPoint(point.latlon!);
    debugGeojsonSource.value = JSON.parse(response.value);

    processingStatus.value.isProcessing = false;
    showNotification('OSM network data loaded', 'success');
  } catch (error) {
    processingStatus.value.isProcessing = false;
    console.error('Error fetching OSM data around point:', error);
    showNotification('Error loading OSM network data', 'error');
  }
}

// Show point candidate segments
const showPointCandidates = (pointIndex: number) => {
  if (!routeMatchTrace.value || !routeMatchTrace.value.pointCandidates) {
    showNotification('No candidate data available for this point', 'error');
    return;
  }

  try {
    // Get candidates for the specific point
    const candidateJson = routeMatchTrace.value.pointCandidates[pointIndex];
    if (!candidateJson) {
      showNotification('No candidate data available for this point', 'error');
      return;
    }

    // Parse and display the GeoJSON
    debugGeojsonSource.value = JSON.parse(candidateJson) as GeoJSON;
    showNotification(`Showing candidates for point ${pointIndex}`, 'success');
  } catch (error) {
    console.error('Error showing point candidates:', error);
    showNotification('Error processing candidate data', 'error');
  }
}

// Debug a failed window pathfinding
const debugFailedWindow = async () => {
  if (!selectedWindow.value || !selectedWindow.value.bridge) return;

  try {
    processingStatus.value.isProcessing = true;
    processingStatus.value.message = 'Analyzing pathfinding data...';

    // Create the request to debug window pathfinding
    const debugRequest = new WindowDebugRequest({
      tripId: tripId,
      windowIndex: selectedWindowIndex.value!,
      startPoint: selectedWindow.value.start,
      endPoint: selectedWindow.value.end
    });

    // Get the first segment ID from the window if it exists
    const firstSegment = selectedWindow.value.segments[0];
    if (firstSegment) {
      debugRequest.fromSegmentId = firstSegment.id;
    }

    // Get the last segment ID from the window if it exists
    const lastSegment = selectedWindow.value.segments[selectedWindow.value.segments.length - 1];
    if (lastSegment) {
      debugRequest.toSegmentId = lastSegment.id;
    }

    // Call the debug API directly
    const response = await api.debugWindowPathFinding(debugRequest);

    // Update the state with the debugging info
    pathDebugInfo.value = response;
    processingStatus.value.isProcessing = false;
    showNotification('Pathfinding debug info loaded', 'success');
  } catch (error) {
    processingStatus.value.isProcessing = false;
    console.error('Error debugging failed window:', error);
    showNotification('Error analyzing pathfinding data', 'error');
  }
};

// Visualize connectivity between segments
const visualizeConnectivity = async () => {
  if (!selectedWindow.value || !pathDebugInfo.value) return;

  try {
    processingStatus.value.isProcessing = true;
    processingStatus.value.message = 'Visualizing segment connectivity...';

    // Prepare the connectivity request
    const connectivityRequest: ConnectivityRequest = new ConnectivityRequest({
      tripId: tripId,
      startPointIndex: selectedWindow.value.start,
      endPointIndex: selectedWindow.value.end
    });

    // Add from/to segment IDs if we have them from the pathfinding debug info
    if (pathDebugInfo.value.attemptedPairs && pathDebugInfo.value.attemptedPairs.length > 0) {
      const firstAttempt = pathDebugInfo.value.attemptedPairs[0];
      connectivityRequest.fromSegmentId = firstAttempt.fromSegment;
      connectivityRequest.toSegmentId = firstAttempt.toSegment;
    }

    // Call the connectivity analysis API
    const response = await api.analyzeSegmentConnectivity(connectivityRequest);

    // Update the debug GeoJSON source to show the connectivity
    debugGeojsonSource.value = JSON.parse(response.value);
    processingStatus.value.isProcessing = false;
    showNotification('Connectivity visualization loaded', 'success');
  } catch (error) {
    processingStatus.value.isProcessing = false;
    console.error('Error visualizing connectivity:', error);
    showNotification('Error visualizing connectivity', 'error');
  }
};

// Analyze segment connectivity directly
const analyzeSegmentConnectivity = async () => {
  if (!selectedWindow.value) return;

  try {
    processingStatus.value.isProcessing = true;
    processingStatus.value.message = 'Analyzing segment connectivity...';

    // Create connectivity request using selected window
    const connectivityRequest: ConnectivityRequest = new ConnectivityRequest({
      tripId: tripId,
      startPointIndex: selectedWindow.value.start,
      endPointIndex: selectedWindow.value.end
    });

    // Use first and last segment of window as from/to segments
    if (selectedWindow.value.segments.length > 0) {
      connectivityRequest.fromSegmentId = selectedWindow.value.segments[0].id;
      connectivityRequest.toSegmentId = selectedWindow.value.segments[selectedWindow.value.segments.length - 1].id;
    }

    // Call the API directly to analyze segment connectivity
    const response = await api.analyzeSegmentConnectivity(connectivityRequest);

    // Update visualization with the response
    debugGeojsonSource.value = JSON.parse(response.value);
    processingStatus.value.isProcessing = false;
    showNotification('Segment connectivity analysis complete', 'success');
  } catch (error) {
    processingStatus.value.isProcessing = false;
    console.error('Error analyzing segment connectivity:', error);
    showNotification('Error analyzing segment connectivity', 'error');
  }
};

// Debug segment connections
const debugSegmentConnections = async (segment: RoadSegment) => {
  if (!map.map) return;

  try {
    processingStatus.value.isProcessing = true;
    processingStatus.value.message = 'Analyzing segment connections...';

    // Create a GeoJSON to visualize the segment and its connections
    const features: Feature[] = [];

    // Add the segment itself
    features.push({
      type: 'Feature',
      properties: {
        id: segment.id.toString(),
        name: segment.name || 'Unnamed Road',
        type: 'segment',
        description: `Segment ${segment.id}: ${segment.name || 'Unnamed'} (${segment.highwayType})`
      },
      geometry: {
        type: 'LineString',
        coordinates: segment.coordinates.map(coord => [coord.lon, coord.lat])
      }
    });

    // Add points for each node in the segment
    segment.coordinates.forEach((coord, idx) => {
      features.push({
        type: 'Feature',
        properties: {
          id: `${segment.id}-node-${idx}`,
          type: 'segment-node',
          description: `Node ${idx} of segment ${segment.id}`
        },
        geometry: {
          type: 'Point',
          coordinates: [coord.lon, coord.lat]
        }
      });
    });

    // Add connection lines if available
    if (segment.connections && segment.connections.length > 0) {
      // First, we need to find all the connected segments in all windows
      const connectedSegments: RoadSegment[] = [];

      if (routeMatchTrace.value) {
        routeMatchTrace.value.windowTraces.forEach(window => {
          window.segments.forEach(s => {
            if (segment.connections.includes(s.id) && !connectedSegments.some(cs => cs.id === s.id)) {
              connectedSegments.push(s);
            }
          });
        });
      }

      // Add connected segments to the visualization
      connectedSegments.forEach(connectedSeg => {
        // Add the connected segment line
        features.push({
          type: 'Feature',
          properties: {
            id: connectedSeg.id.toString(),
            name: connectedSeg.name || 'Unnamed Road',
            type: 'connection',
            description: `Connected to segment ${segment.id}: ${connectedSeg.name || 'Unnamed'} (${connectedSeg.highwayType})`
          },
          geometry: {
            type: 'LineString',
            coordinates: connectedSeg.coordinates.map(coord => [coord.lon, coord.lat])
          }
        });

        // Add a line connecting the nearest points between the segments
        const [fromPoint, toPoint] = findClosestPoints(segment, connectedSeg);

        features.push({
          type: 'Feature',
          properties: {
            id: `connection-${segment.id}-${connectedSeg.id}`,
            type: 'connection-line',
            description: `Connection between segment ${segment.id} and ${connectedSeg.id}`
          },
          geometry: {
            type: 'LineString',
            coordinates: [
              [fromPoint.lon, fromPoint.lat],
              [toPoint.lon, toPoint.lat]
            ]
          }
        });
      });
    }

    // Update the debug geojson source
    debugGeojsonSource.value = {
      type: 'FeatureCollection',
      features: features
    };

    // Fly to the segment
    const segmentCenter = calculateSegmentCentroid(segment);
    map.map.flyTo({
      center: segmentCenter as LngLatLike,
      zoom: 15,
      speed: 0.8
    });

    processingStatus.value.isProcessing = false;
    showNotification(`Showing connections for segment ${segment.id}`, 'success');
  } catch (error) {
    processingStatus.value.isProcessing = false;
    console.error('Error debugging segment connections:', error);
    showNotification('Error analyzing segment connections', 'error');
  }
};

// Find the closest points between two segments
const findClosestPoints = (segmentA: RoadSegment, segmentB: RoadSegment) => {
  let minDistance = Infinity;
  let closestFromPoint = segmentA.coordinates[0];
  let closestToPoint = segmentB.coordinates[0];

  // Find the closest pair of points between the segments
  for (const fromPoint of segmentA.coordinates) {
    for (const toPoint of segmentB.coordinates) {
      const distance = calculateDistance(fromPoint, toPoint);
      if (distance < minDistance) { minDistance = distance; closestFromPoint = fromPoint; closestToPoint = toPoint; }
    }
  }
  return [closestFromPoint, closestToPoint];
}; // Calculate distance between two points const

const calculateDistance = (pointA: { lat: number, lon: number }, pointB: { lat: number, lon: number }) => {
  // Simple Euclidean distance (good enough for visualization purposes)
  return Math.sqrt(
    Math.pow((pointA.lat - pointB.lat) * 111.32, 2) +
    Math.pow((pointA.lon - pointB.lon) * 111.32 * Math.cos(pointA.lat * Math.PI / 180), 2)
  );
};

// Reprocess the trip
const reprocessTrip = async () => {
  if (!tripData.value) return;

  try {
    processingStatus.value.isProcessing = true;
    processingStatus.value.message = 'Reprocessing trip...';
    processingStatus.value.canCancel = true;

    // Create a UUID reference to the trip
    const tripReference = {
      tripId: {
        value: tripId
      }
    };

    // Call the reprocessing API
    const response = await tripsApi.reprocessTrip(tripReference);

    if (response.success) {
      processingStatus.value.isProcessing = false;
      showNotification('Trip reprocessing started successfully. Refresh the page in a few moments to see results.',
        'success');

      // Schedule a refresh after 5 seconds
      setTimeout(() => {
        refreshTripData();
      }, 5000);
    } else {
      processingStatus.value.isProcessing = false;
      showNotification(`Trip reprocessing failed: ${response.message}`, 'error');
    }
  } catch (error) {
    processingStatus.value.isProcessing = false;
    console.error('Error reprocessing trip:', error);
    showNotification('Error while reprocessing trip', 'error');
  }
};

// Refresh trip data
const refreshTripData = async () => {
  try {
    processingStatus.value.isProcessing = true;
    processingStatus.value.message = 'Refreshing trip data...';

    // Fetch updated trip details
    const response = await tripsApi.getTripDetails({
      tripId: {
        value: tripId
      }
    });

    tripData.value = response.trip;

    // Set the routeMatchTrace from the updated trip details
    if (response.trip?.routeMatchTrace) {
      routeMatchTrace.value = response.trip.routeMatchTrace;

      // Reset the map layers with new data
      setupMapLayers();

      showNotification('Trip data refreshed successfully', 'success');
    } else {
      showNotification('Trip data refreshed, but no route match trace available', 'error');
    }

    processingStatus.value.isProcessing = false;
  } catch (error) {
    processingStatus.value.isProcessing = false;
    console.error('Error refreshing trip data:', error);
    showNotification('Failed to refresh trip data', 'error');
  }
};

// Cancel processing operation
const cancelProcessing = () => {
  processingStatus.value.isProcessing = false;
  showNotification('Operation cancelled', 'success');
};

// Helper methods for path finding debug
const getAttemptClass = (attempt: PathfindingAttempt) => {
  switch (attempt.result?.type) {
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

// Format DateTime to readable string
const formatDateTime = (dateTime: DateTime) => {
  if (!dateTime) return '';

  try {
    // Extract datetime components
    const { year, month, day, hours, minutes, seconds } = dateTime;

    // JavaScript months are 0-based (0-11), while DateTime months are 1-based (1-12)
    const jsDate = new Date(year, month - 1, day, hours, minutes, seconds);
    return jsDate.toLocaleString();
  } catch (e) {
    console.error('Error formatting date time:', e);
    return '';
  }
};

</script>

<style lang="css">
html,
body,
div#__nuxt {
  height: 100%;
  width: 100%;
}

.window-button {
  transition: all 0.2s ease;
}

.window-button:hover {
  transform: scale(1.1);
  box-shadow: 0 0 8px rgba(255, 255, 255, 0.5);
}

.segment-item {
  transition: all 0.2s ease;
}

.point-marker {
  transition: transform 0.2s ease;
}
</style>