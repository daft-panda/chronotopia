<template>
  <div class="main-container">
    <div class="header">
      <h1>ChronoTopia Route Visualization</h1>

      <div class="controls">
        <button class="control-button" :class="{ active: showConstraints }" @click="toggleConstraints">
          {{ showConstraints ? 'Hide Constraints' : 'Show Constraints' }}
        </button>
        <button class="control-button" :class="{ active: showDebugInfo }" @click="toggleDebugInfo">
          {{ showDebugInfo ? 'Hide Debug Info' : 'Show Debug Info' }}
        </button>
      </div>
    </div>

    <div class="window-selector">
      <div v-for="(window, i) in routeMatchTrace.windowTraces" :key="i" class="window-button"
        :class="getWindowClass(window)" :style="{
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
          <div class="window-type-indicator" :class="getWindowClass(selectedWindow)">
            {{ selectedWindow.usedConstraints ? 'Constrained' : (selectedWindow.bridge ? 'Constraints failed (bridge)' :
              'Unconstrained') }}
          </div>
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

        <!-- Debug information section - shows when debug mode is enabled -->
        <div v-if="showDebugInfo" class="debug-container">
          <h3>Debug Information</h3>

          <div class="stat-item">
            <div class="stat-label">Constraints:</div>
            <div class="stat-value">{{ getWindowStats(selectedWindow)?.constraintsCount }}</div>
          </div>

          <div class="stat-item">
            <div class="stat-label">Used Constraints:</div>
            <div class="stat-value">{{ getWindowStats(selectedWindow)?.usedConstraints }}</div>
          </div>

          <div class="stat-item">
            <div class="stat-label">Constraint Score:</div>
            <div class="stat-value">{{ getWindowStats(selectedWindow)?.constraintScore }}</div>
          </div>

          <div class="stat-item">
            <div class="stat-label">Unconstrained Score:</div>
            <div class="stat-value">{{ getWindowStats(selectedWindow)?.unconstrainedScore }}</div>
          </div>

          <div class="stat-item">
            <div class="stat-label">Attempted Way IDs:</div>
            <div class="stat-value">{{ getWindowStats(selectedWindow)?.attemptedWayIds }}</div>
          </div>

          <!-- Additional debug notes -->
          <div v-if="selectedWindow.debugNotes && selectedWindow.debugNotes.length > 0" class="debug-notes">
            <h4>Debug Notes:</h4>
            <ul>
              <li v-for="(note, index) in selectedWindow.debugNotes" :key="index">{{ note }}</li>
            </ul>
          </div>

          <!-- Constraint list -->
          <div v-if="selectedWindow.constraints && selectedWindow.constraints.length > 0" class="constraints-list">
            <h4>Constraints:</h4>
            <div v-for="(constraint, index) in selectedWindow.constraints" :key="index" class="constraint-item">
              <div class="constraint-header">
                Point {{ constraint.pointIdx }} → Segment {{ constraint.segmentId }} (Way {{ constraint.wayId }})
              </div>
              <div class="constraint-details">
                Distance: {{ constraint.distance.toFixed(2) }}m
              </div>
            </div>
          </div>

          <!-- Attempted Way IDs visualization -->
          <div v-if="selectedWindow.attemptedWayIds && selectedWindow.attemptedWayIds.length > 0"
            class="attempted-ways">
            <h4>Attempted Way IDs:</h4>
            <div class="way-id-tags">
              <span v-for="(wayId, index) in selectedWindow.attemptedWayIds" :key="index" class="way-id-tag">
                {{ wayId }}
              </span>
            </div>
          </div>

          <div v-if="selectedWindow && showDebugInfo && selectedWindow.bridge" class="path-debugging">
            <h3>Path Finding Debug</h3>
            <div class="debug-message">
              <p>This window failed to find a valid route with constraints</p>
              <button class="debug-button" @click="debugFailedWindow">
                Debug Path Finding
              </button>
            </div>

            <div v-if="pathDebugInfo" class="debug-results mt-4">
              <h4>Path Finding Analysis</h4>
              <div class="text-sm text-red-500 mb-2">{{ pathDebugInfo.reason }}</div>

              <div class="mb-4">
                <div class="font-medium mb-1">Attempted Pairs:</div>
                <div class="grid grid-cols-4 gap-2">
                  <div class="bg-green-100 p-2 rounded text-center">
                    <div class="font-bold text-green-800">
                      {{pathDebugInfo.attemptedPairs.filter(p => p.result.type === 'Success').length}}
                    </div>
                    <div class="text-xs">Successful</div>
                  </div>
                  <div class="bg-yellow-100 p-2 rounded text-center">
                    <div class="font-bold text-yellow-800">
                      {{pathDebugInfo.attemptedPairs.filter(p => p.result.type === 'TooFar').length}}
                    </div>
                    <div class="text-xs">Too Far</div>
                  </div>
                  <div class="bg-red-100 p-2 rounded text-center">
                    <div class="font-bold text-red-800">
                      {{pathDebugInfo.attemptedPairs.filter(p => p.result.type === 'NoConnection').length}}
                    </div>
                    <div class="text-xs">No Connection</div>
                  </div>
                  <div class="bg-gray-200 p-2 rounded text-center">
                    <div class="font-bold text-gray-800">
                      {{pathDebugInfo.attemptedPairs.filter(p =>
                        p.result.type !== 'Success' &&
                        p.result.type !== 'TooFar' &&
                        p.result.type !== 'NoConnection'
                      ).length}}
                    </div>
                    <div class="text-xs">Other Errors</div>
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

              <button class="debug-button mb-2" @click="visualizeConnectivity">
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
                    <span class="font-medium">{{ getResultLabel(attempt.result) }}</span>
                  </div>
                  <div class="text-xs">Distance: {{ attempt.distance.toFixed(2) }}m</div>

                  <div v-if="attempt.result.type === 'TooFar'">
                    <div class="text-xs">
                      Max allowed: {{ attempt.result?.maxDistance.toFixed(2) }}m
                    </div>
                  </div>

                  <div v-if="attempt.result.type === 'NoPathFound'">
                    <div class="text-xs text-red-500">
                      Error: {{ attempt.result?.reason }}
                    </div>
                  </div>
                </div>

                <div v-if="pathDebugInfo.attemptedPairs.length > 10" class="text-sm text-gray-500">
                  ...and {{ pathDebugInfo.attemptedPairs.length - 10 }} more attempts
                </div>
              </div>
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
                  'window-point': selectedWindow && index >= selectedWindow.start && index <= selectedWindow.end,
                  'constrained-point': selectedWindow && selectedWindow.constraints &&
                    selectedWindow.constraints.some(c => c.pointIdx === index)
                }">
                  {{ index }}
                </div>
              </template>
              <MglPopup ref="popup">
                <div class="popup-content">
                  <h3>Point {{ index }}</h3>
                  <p>Coordinates: {{ point.latlon?.lat.toFixed(6) }}, {{ point.latlon?.lon.toFixed(6) }}</p>

                  <!-- Show constraint info in popup if this point is constrained -->
                  <div v-if="selectedWindow && selectedWindow.constraints &&
                    selectedWindow.constraints.some(c => c.pointIdx === index)" class="constraint-info">
                    <h4>Constraint Info:</h4>
                    <div v-for="(constraint, cIdx) in selectedWindow.constraints.filter(c => c.pointIdx === index)"
                      :key="cIdx" class="constraint-detail">
                      <p>Segment: {{ constraint.segmentId }} (Way: {{ constraint.wayId }})</p>
                      <p>Distance: {{ constraint.distance.toFixed(2) }}m</p>
                    </div>
                  </div>

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

    <div class="legend">
      <h3>Legend</h3>
      <div class="legend-item">
        <div class="legend-color constrained" />
        <div class="legend-label">Constrained Route</div>
      </div>
      <div class="legend-item">
        <div class="legend-color unconstrained" />
        <div class="legend-label">Unconstrained Route</div>
      </div>
      <div class="legend-item">
        <div class="legend-color unconstrained-bridge" />
        <div class="legend-label">Unconstrained Bridge Route</div>
      </div>
      <div class="legend-item">
        <div class="legend-point constrained-point-legend" />
        <div class="legend-label">Constrained Point</div>
      </div>
      <div class="legend-item">
        <div class="legend-point start-point-legend" />
        <div class="legend-label">Start Point</div>
      </div>
      <div class="legend-item">
        <div class="legend-point end-point-legend" />
        <div class="legend-label">End Point</div>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import type { Point, WindowTrace, RoadSegment, PathfindingDebugInfo } from './model/chronotopia_pb';
import { type Map, Popup, type LineLayerSpecification, type LngLatLike } from 'maplibre-gl';
import { MglFullscreenControl, MglGeolocateControl, MglMap, MglMarker, MglNavigationControl } from '#components';
import type { Feature, GeoJSON, GeoJsonProperties, Geometry } from 'geojson';

const layout = {
  'line-join': 'round',
  'line-cap': 'round'
} as LineLayerSpecification['layout'];

const { $api } = useNuxtApp();
const routeMatchTrace = await $api.getRouteMatchTrace({});
const selectedWindowIndex: Ref<number | null> = ref(null);
const selectedWindow: Ref<WindowTrace | null> = ref(null);
const highlightedSegment: Ref<RoadSegment | null> = ref(null);
const showConstraints: Ref<boolean> = ref(false);
const showDebugInfo: Ref<boolean> = ref(false);
const pathDebugInfo: Ref<PathfindingDebugInfo | null> = ref(null);

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

const toggleConstraints = () => {
  showConstraints.value = !showConstraints.value;

  if (import.meta.client) {
    const map = useMglMap();
    const mmap = map.map as Map;

    // Toggle visibility of constraint layers
    routeMatchTrace.windowTraces.forEach((window, idx) => {
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
}

const toggleDebugInfo = () => {
  showDebugInfo.value = !showDebugInfo.value;
}

const debugGeojsonSource: Ref<undefined | GeoJSON<Geometry, GeoJsonProperties>> = ref(undefined);

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
};

// Get colors for the window selector
const getWindowColor = (index: number) => {
  const window = routeMatchTrace.windowTraces[index];

  // Base color calculated from index
  const hueStep = 255 / routeMatchTrace.windowTraces.length;
  const baseHue = Math.round(hueStep * index);

  // Modify saturation based on constraint usage
  const saturation = window.usedConstraints ? 80 : 50;

  // Modify lightness based on bridge status
  const lightness = window.bridge ? 65 : 50;

  return `hsl(${baseHue}, ${saturation}%, ${lightness}%)`;
};

// Get the window background class based on its properties
const getWindowClass = (window: WindowTrace) => {
  if (window.usedConstraints) return 'constrained';
  if (window.bridge) return 'unconstrained-bridge';
  return 'unconstrained';
};

if (import.meta.client) {
  const map = useMglMap();
  const layers: string[] = [];

  watch(() => map.isLoaded, () => {
    for (const windowIdx in routeMatchTrace.windowTraces) {
      const window = routeMatchTrace.windowTraces[windowIdx];
      const sourceId = `window-${windowIdx}`;
      layers.push(sourceId);

      // Create the main segment geojson
      const geojson: GeoJSON<Geometry, GeoJsonProperties> = {
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
                extractedCoords.push([
                  segment.fullCoordinates[i].lon,
                  segment.fullCoordinates[i].lat
                ]);
              }
            }
            coordinates = extractedCoords;
          } else {
            // Normal case: use coordinates as provided
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

        // Create constraint connections
        const constraintFeatures = window.constraints.map(constraint => {
          // Get the point coordinates
          const point = routeMatchTrace.trip?.points[constraint.pointIdx];
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
              description: `Constraint: Point ${constraint.pointIdx} to Segment ${constraint.segmentId} (Way ${constraint.wayId}), Distance: ${constraint.distance.toFixed(2)}m`
            },
            geometry: {
              type: 'LineString',
              coordinates: [
                [point.latlon.lon, point.latlon.lat],
                segmentCentroid
              ]
            }
          };
        }).filter(f => f !== null) as Feature[];

        const constraintPointsFeatures = window.constraints.map(constraint => {
          const point = routeMatchTrace.trip?.points[constraint.pointIdx];
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
          };
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
  <strong>OSM Way ID:</strong> ${feature.properties.osmWayId}<br>
  <strong>One-way:</strong> ${feature.properties.isOneway ? 'Yes' : 'No'}<br>
  <strong>Constraints:</strong> ${feature.properties.isConstrained ? 'Used' : 'Not used'}<br>
  <strong>Bridge:</strong> ${feature.properties.isBridge ? 'Yes (unconstrained)' : 'No'}<br>
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
  });
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

const debugFailedWindow = async () => {
  if (!selectedWindow.value || !selectedWindow.value.bridge) return;

  // Call the backend to get debugging information for this window
  const failedWindowData = await $api.debugWindowPathFinding({
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
    startPoint: selectedWindow.value.start,
    endPoint: selectedWindow.value.end
  });

  // Update the debug GeoJSON source to show the connectivity
  debugGeojsonSource.value = JSON.parse(connectivityData.value) as GeoJSON<Geometry, GeoJsonProperties>;
};

// Helper methods
const getAttemptClass = (attempt) => {
  switch (attempt.result.type) {
    case 'Success': return 'bg-green-100';
    case 'TooFar': return 'bg-yellow-100';
    case 'NoConnection': return 'bg-red-100';
    default: return 'bg-gray-100';
  }
};

const getResultLabel = (result) => {
  switch (result.type) {
    case 'Success': return 'Success';
    case 'TooFar': return 'Too Far';
    case 'NoConnection': return 'No Connection';
    case 'NoPathFound': return 'Path Error';
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
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header h1 {
  margin: 0;
  font-size: 1.5rem;
  color: #fff;
}

.controls {
  display: flex;
  gap: 10px;
}

.control-button {
  background-color: #555;
  color: white;
  border: none;
  padding: 8px 15px;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
}

.control-button:hover {
  background-color: #666;
}

.control-button.active {
  background-color: #2196f3;
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
  position: relative;
}

.window-button:hover {
  transform: scale(1.1);
  box-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
}

.window-button.constrained::after {
  content: "C";
  position: absolute;
  top: -5px;
  right: -5px;
  background-color: #4caf50;
  color: white;
  font-size: 10px;
  width: 15px;
  height: 15px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.window-button.unconstrained-bridge::after {
  content: "B";
  position: absolute;
  top: -5px;
  right: -5px;
  background-color: #f44336;
  color: white;
  font-size: 10px;
  width: 15px;
  height: 15px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}

border .window-button.unconstrained-bridge::after {
  content: "B";
  position: absolute;
  top: -5px;
  right: -5px;
  background-color: #f44336;
  color: white;
  font-size: 10px;
  width: 15px;
  height: 15px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
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
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.panel-header h2 {
  margin: 0;
  font-size: 1.2rem;
  color: #fff;
}

.window-type-indicator {
  padding: 3px 8px;
  border-radius: 3px;
  font-size: 0.8rem;
  color: white;
}

.window-type-indicator.constrained {
  background-color: #4caf50;
}

.window-type-indicator.unconstrained {
  background-color: #2196f3;
}

.window-type-indicator.unconstrained-bridge {
  background-color: #f44336;
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

.debug-container {
  padding: 15px;
  border-bottom: 1px solid #444;
  background-color: #333;
}

.debug-container h3 {
  margin-top: 0;
  color: #2196f3;
  border-bottom: 1px solid #444;
  padding-bottom: 5px;
}

.debug-container h4 {
  margin: 10px 0 5px;
  color: #ddd;
}

.debug-notes {
  margin-top: 10px;
  padding: 10px;
  background-color: #2a2a2a;
  border-radius: 4px;
}

.debug-notes ul {
  margin: 0;
  padding-left: 20px;
}

.debug-notes li {
  margin-bottom: 5px;
  color: #ddd;
}

.constraints-list {
  margin-top: 10px;
}

.constraint-item {
  padding: 8px;
  margin-bottom: 5px;
  background-color: #2a2a2a;
  border-radius: 4px;
  border-left: 3px solid #ff00ff;
}

.constraint-header {
  font-weight: bold;
  margin-bottom: 3px;
}

.constraint-details {
  font-size: 0.9rem;
  color: #aaa;
}

.attempted-ways {
  margin-top: 10px;
}

.way-id-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}

.way-id-tag {
  background-color: #2a2a2a;
  padding: 3px 6px;
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
.segment-id,
.segment-partial {
  background-color: #222;
  padding: 2px 6px;
  border-radius: 3px;
}

.segment-oneway {
  background-color: #553;
}

.segment-partial {
  background-color: #335;
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

.constrained-point {
  background-color: #ff00ff;
  border-color: white;
  transform: scale(1.2);
  z-index: 9;
}

.popup-content {
  color: #333;
  padding: 5px;
}

.popup-content h3 {
  margin-top: 0;
  margin-bottom: 5px;
}

.constraint-info {
  margin-top: 10px;
  padding: 5px;
  background-color: #f9e9f9;
  border-radius: 3px;
  border-left: 3px solid #ff00ff;
}

.constraint-info h4 {
  margin: 0 0 5px;
  color: #800080;
}

.constraint-detail {
  margin-bottom: 5px;
}

.constraint-detail p {
  margin: 3px 0;
  font-size: 0.85rem;
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

.constraint-popup {
  color: #333;
}

.constraint-popup h4 {
  margin-top: 0;
  color: #800080;
}

.legend {
  position: absolute;
  bottom: 20px;
  right: 20px;
  background-color: rgba(42, 42, 42, 0.8);
  padding: 10px;
  border-radius: 5px;
  z-index: 10;
  max-width: 250px;
}

.legend h3 {
  margin-top: 0;
  margin-bottom: 10px;
  font-size: 1rem;
  color: #fff;
}

.legend-item {
  display: flex;
  align-items: center;
  margin-bottom: 5px;
}

.legend-color {
  width: 30px;
  height: 5px;
  margin-right: 10px;
}

.legend-color.constrained {
  background-color: #4caf50;
}

.legend-color.unconstrained {
  background-color: #2196f3;
}

.legend-color.unconstrained-bridge {
  background-color: #f44336;
  border-top: 1px dashed white;
}

.legend-point {
  width: 15px;
  height: 15px;
  border-radius: 50%;
  margin-right: 10px;
}

.legend-point.constrained-point-legend {
  background-color: #ff00ff;
  border: 2px solid white;
}

.legend-point.start-point-legend {
  background-color: #4caf50;
  border: 2px solid #222;
}

.legend-point.end-point-legend {
  background-color: #f44336;
  border: 2px solid #222;
}

.legend-label {
  font-size: 0.8rem;
  color: #fff;
}

.debug-button {
  background-color: #2196f3;
  color: white;
  font-size: 0.85rem;
  padding: 0.5rem 1rem;
  border-radius: 0.25rem;
  cursor: pointer;
  border: none;
  transition: background-color 0.2s;
}

.debug-button:hover {
  background-color: #0d8aea;
}

.debug-message {
  padding: 0.75rem;
  background-color: #fff3f3;
  border-radius: 0.25rem;
  border-left: 3px solid #f44336;
  margin-bottom: 1rem;
}

.debug-results {
  background-color: #f8f9fa;
  padding: 0.75rem;
  border-radius: 0.25rem;
  border: 1px solid #dee2e6;
}

.path-debugging h3 {
  margin-top: 1rem;
  margin-bottom: 0.5rem;
  color: #dc3545;
  font-weight: 600;
}

.path-attempts {
  margin-top: 0.5rem;
}

.attempt {
  border-left: 3px solid transparent;
}

.attempt.bg-green-100 {
  border-left-color: #28a745;
}

.attempt.bg-yellow-100 {
  border-left-color: #ffc107;
}

.attempt.bg-red-100 {
  border-left-color: #dc3545;
}

.attempt.bg-gray-100 {
  border-left-color: #6c757d;
}
</style>