// pages/route-planner/[id].vue

<template>
    <div class="container h-full mx-auto px-4 py-8">
        <!-- Back button -->
        <div class="mb-6">
            <button class="flex items-center text-blue-500 hover:text-blue-700 transition" @click="router.back()">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24"
                    stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                </svg>
                Back to trips
            </button>
        </div>

        <!-- Loading state -->
        <div v-if="pending" class="w-full flex justify-center items-center py-12">
            <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500" />
        </div>

        <!-- Error state -->
        <div v-else-if="error" class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6" role="alert">
            <p class="font-bold">Error</p>
            <p>{{ error.message }}</p>
        </div>

        <!-- Trip content -->
        <div v-else-if="trip" class="bg-white rounded-lg shadow-lg h-3/4 overflow-hidden">
            <!-- Header -->
            <div class="bg-gradient-to-r from-blue-500 to-indigo-600 p-6 text-white">
                <h1 class="text-2xl font-bold">Route Planner - {{ trip.label || formatTripTitle(trip) }}</h1>
                <div class="flex items-center mt-2 text-blue-100">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24"
                        stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <span>{{ formatDate(trip.startTime!) }} at {{ formatTime(trip.startTime!) }}</span>
                </div>
            </div>

            <!-- Map and controls container -->
            <div class="h-full flex flex-col lg:flex-row">
                <!-- Map -->
                <div class="h-full lg:flex-grow relative">
                    <ClientOnly>
                        <MglMap ref="mapRef"
                            map-style="https://api.maptiler.com/maps/streets/style.json?key=Ic6Mr5qetb5kn90hyEzO"
                            :zoom="13" :center="initialCenter" @map:click="onMapClick">
                            <MglFullscreenControl position="top-right" />
                            <MglNavigationControl position="top-right" />
                            <MglGeolocateControl position="top-right" />

                            <!-- Trip points for reference -->
                            <MglMarker v-for="(point, index) in trip.points" :key="`trip-point-${index}`"
                                :coordinates="[point.latlon!.lon, point.latlon!.lat]">
                                <template #marker>
                                    <div
                                        class="w-3 h-3 rounded-full bg-gray-500 border-2 border-white shadow-md opacity-50" />
                                </template>
                                <MglPopup>
                                    <div class="p-2">
                                        <h3 class="font-bold">Trip Point {{ index }}</h3>
                                        <p class="text-sm">
                                            {{ point.latlon?.lat.toFixed(6) }}, {{ point.latlon?.lon.toFixed(6) }}
                                        </p>
                                        <p v-if="point.dateTime" class="text-sm text-gray-600">
                                            {{ formatTime(point.dateTime) }}
                                        </p>
                                    </div>
                                </MglPopup>
                            </MglMarker>

                            <!-- Route planning markers -->
                            <MglMarker v-if="startPoint" :coordinates="[startPoint.lon, startPoint.lat]"
                                :draggable="true" @dragend="updateStartPoint">
                                <template #marker>
                                    <div
                                        class="w-6 h-6 bg-green-500 rounded-full border-2 border-white flex items-center justify-center text-white font-bold">
                                        S
                                    </div>
                                </template>
                                <MglPopup>
                                    <div class="p-2">
                                        <h3 class="font-bold">Start Point</h3>
                                        <p class="text-sm">
                                            {{ startPoint.lat.toFixed(6) }}, {{ startPoint.lon.toFixed(6) }}
                                        </p>
                                    </div>
                                </MglPopup>
                            </MglMarker>

                            <MglMarker v-if="endPoint" :coordinates="[endPoint.lon, endPoint.lat]" :draggable="true"
                                @dragend="updateEndPoint">
                                <template #marker>
                                    <div
                                        class="w-6 h-6 bg-red-500 rounded-full border-2 border-white flex items-center justify-center text-white font-bold">
                                        E
                                    </div>
                                </template>
                                <MglPopup>
                                    <div class="p-2">
                                        <h3 class="font-bold">End Point</h3>
                                        <p class="text-sm">
                                            {{ endPoint.lat.toFixed(6) }}, {{ endPoint.lon.toFixed(6) }}
                                        </p>
                                    </div>
                                </MglPopup>
                            </MglMarker>

                            <MglMarker v-for="(point, index) in viaPoints" :key="`via-${index}`"
                                :coordinates="[point.lon, point.lat]" :draggable="true"
                                @dragend="(e) => updateViaPoint(e, index)">
                                <template #marker>
                                    <div
                                        class="w-6 h-6 bg-blue-500 rounded-full border-2 border-white flex items-center justify-center text-white font-bold">
                                        {{ index + 1 }}
                                    </div>
                                </template>
                                <MglPopup>
                                    <div class="p-2">
                                        <h3 class="font-bold">Via Point {{ index + 1 }}</h3>
                                        <p class="text-sm">
                                            {{ point.lat.toFixed(6) }}, {{ point.lon.toFixed(6) }}
                                        </p>
                                        <button class="mt-2 text-xs text-red-500 hover:text-red-700"
                                            @click="removeViaPoint(index)">
                                            Remove
                                        </button>
                                    </div>
                                </MglPopup>
                            </MglMarker>

                            <!-- Planned route -->
                            <MglGeoJsonSource v-if="routeGeoJson" source-id="planned-route" :data="routeGeoJson">
                                <MglLineLayer layer-id="planned-route-line" source="planned-route"
                                    :layout="{ 'line-join': 'round', 'line-cap': 'round' }" :paint="{
                                        'line-color': [
                                            'match',
                                            ['get', 'type'],
                                            'route',
                                            '#3388ff',
                                            '#888888'
                                        ],
                                        'line-width': [
                                            'match',
                                            ['get', 'type'],
                                            'route',
                                            6,
                                            2
                                        ],
                                        'line-opacity': 0.8
                                    }" />
                            </MglGeoJsonSource>
                        </MglMap>
                    </ClientOnly>
                </div>

                <!-- Route controls -->
                <div class="lg:w-80 p-4 bg-gray-50 border-l">
                    <h2 class="text-lg font-medium mb-4">Route Planning</h2>

                    <div class="space-y-4">
                        <!-- Mode selection -->
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">Mode</label>
                            <div class="flex gap-2">
                                <button :class="[
                                    'px-3 py-1 rounded text-sm',
                                    mode === 'start'
                                        ? 'bg-green-500 text-white'
                                        : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
                                ]" @click="mode = 'start'">
                                    Start
                                </button>
                                <button :class="[
                                    'px-3 py-1 rounded text-sm',
                                    mode === 'via'
                                        ? 'bg-blue-500 text-white'
                                        : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
                                ]" @click="mode = 'via'">
                                    Via
                                </button>
                                <button :class="[
                                    'px-3 py-1 rounded text-sm',
                                    mode === 'end'
                                        ? 'bg-red-500 text-white'
                                        : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
                                ]" @click="mode = 'end'">
                                    End
                                </button>
                            </div>
                            <p class="text-xs text-gray-500 mt-1">
                                Select a mode and click on the map to place points
                            </p>
                        </div>

                        <!-- Via points list -->
                        <div v-if="viaPoints.length > 0">
                            <label class="block text-sm font-medium text-gray-700 mb-1">Via Points</label>
                            <div class="max-h-40 overflow-y-auto">
                                <VueDraggable v-model="viaPoints" :item-key="'id'" ghost-class="bg-blue-100"
                                    handle=".drag-handle" @end="calculateRoute">
                                    <template #item="{ index }">
                                        <div class="flex items-center p-2 mb-1 bg-white rounded border">
                                            <div class="drag-handle cursor-move mr-2">
                                                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-gray-400"
                                                    fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                    <path stroke-linecap="round" stroke-linejoin="round"
                                                        stroke-width="2" d="M4 8h16M4 16h16" />
                                                </svg>
                                            </div>
                                            <div class="flex-grow text-sm truncate">
                                                Via Point {{ index + 1 }}
                                            </div>
                                            <button class="text-red-500 hover:text-red-700"
                                                @click="removeViaPoint(index)">
                                                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none"
                                                    viewBox="0 0 24 24" stroke="currentColor">
                                                    <path stroke-linecap="round" stroke-linejoin="round"
                                                        stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                                                </svg>
                                            </button>
                                        </div>
                                    </template>
                                </VueDraggable>
                            </div>
                        </div>

                        <!-- Route actions -->
                        <div class="space-y-2">
                            <button :disabled="!canCalculateRoute" :class="[
                                'w-full py-2 px-4 rounded font-medium',
                                canCalculateRoute
                                    ? 'bg-blue-600 text-white hover:bg-blue-700'
                                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                            ]" @click="calculateRoute">
                                Calculate Route
                            </button>
                            <button
                                class="w-full py-2 px-4 rounded font-medium bg-gray-200 text-gray-800 hover:bg-gray-300"
                                @click="clearRoute">
                                Clear All
                            </button>
                        </div>

                        <!-- Route info -->
                        <div v-if="routeSegments.length > 0" class="border-t pt-3">
                            <h3 class="text-sm font-medium text-gray-700 mb-1">Route Info</h3>
                            <div class="text-sm">
                                <div class="flex justify-between">
                                    <span>Segments:</span>
                                    <span>{{ routeSegments.length }}</span>
                                </div>
                                <div class="flex justify-between">
                                    <span>Distance:</span>
                                    <span>{{ formatDistance(estimateRouteDistance()) }}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script lang="ts" setup>
import { LngLatBounds, type LngLatLike, type Map as MaplibreMap } from 'maplibre-gl';
import { VueDraggable } from 'vue-draggable-plus';
import { v4 as uuidv4 } from 'uuid';
import type { Trip } from '~/model/trips_pb';
import { LatLon, type RoadSegment } from '~/model/common_pb';
import {
    formatDate,
    formatDistance,
    formatTime,
} from '~/utils/formatting';

// Get route params
const { tripsApi, commonApi } = useApi();
const router = useRouter();
const route = useRoute();
const tripId = route.params.id;

// Map elements
const map = useMglMap();
const mapRef = ref<any>(null);
const mapInstance = ref<MaplibreMap | null>(null);

// Route planning state
const mode = ref<'start' | 'via' | 'end'>('start');
const startPoint: Ref<LatLon | null> = ref(null);
const endPoint: Ref<LatLon | null> = ref(null);
const viaPoints: Ref<({ lat: number, lon: number, id: string })[]> = ref([]);
const routeGeoJson = ref<any>(null);
const routeSegments: Ref<RoadSegment[]> = ref([]);
const calculating = ref(false);

// Computed
const canCalculateRoute = computed(() => {
    return !!startPoint.value && !!endPoint.value && !calculating.value;
});

// Fetch trip details
const { data: trip, pending, error } = useAsyncData(
    'tripDetail',
    async () => {
        try {
            const response = await tripsApi.getTripDetails({
                tripId: {
                    value: tripId as string
                }
            });
            return response.trip;
        } catch (err) {
            console.error(`Error fetching trip ${tripId}:`, err);
            throw new Error('Failed to load trip details');
        }
    }
);

// Calculate initial center point for the map
const initialCenter = computed((): LngLatLike => {
    if (trip.value?.points && trip.value.points.length > 0) {
        // Find a central point from the trip data
        let sumLat = 0;
        let sumLon = 0;
        let count = 0;

        for (const point of trip.value.points) {
            if (point.latlon) {
                sumLat += point.latlon.lat;
                sumLon += point.latlon.lon;
                count++;
            }
        }

        if (count > 0) {
            return [sumLon / count, sumLat / count];
        }
    }

    // Default center if no points available
    return [0, 0];
});

watch(() => map.isLoaded, (isLoaded) => {
    if (isLoaded) {
        mapInstance.value = map.map;

        // Fit map to trip bounds once data is loaded
        fitMapToBounds();
    }
}, { immediate: true });

// Fit map to bounds of the trip points
function fitMapToBounds() {
    if (!mapInstance.value || !trip.value?.points?.length) return;

    // Create a bounds object
    const bounds = new LngLatBounds();

    // Add all points to the bounds
    for (const point of trip.value.points) {
        if (point.latlon) {
            bounds.extend([point.latlon.lon, point.latlon.lat]);
        }
    }

    // Only fit bounds if we have valid coordinates
    if (!bounds.isEmpty()) {
        mapInstance.value.fitBounds(bounds, {
            padding: 50,
            maxZoom: 14
        });
    }
}

// Map interaction
function onMapClick(e: any) {
    const { lngLat } = e.event;
    const point = {
        lat: lngLat.lat,
        lon: lngLat.lng
    };

    if (mode.value === 'start') {
        startPoint.value = new LatLon(point);
    } else if (mode.value === 'end') {
        endPoint.value = new LatLon(point);
    } else if (mode.value === 'via') {
        viaPoints.value.push({
            ...point,
            id: uuidv4() // Add unique ID for draggable
        });
    }

    // Auto calculate route if we have start and end points
    if (startPoint.value && endPoint.value) {
        calculateRoute();
    }
}

// Update marker positions on drag
function updateStartPoint(e: any) {
    const { lngLat } = e.event;
    startPoint.value = new LatLon(lngLat);
    calculateRoute();
}

function updateEndPoint(e: any) {
    const { lngLat } = e.event;
    new LatLon(lngLat);
    calculateRoute();
}

function updateViaPoint(e: any, index: number) {
    const { lngLat } = e.event;
    viaPoints.value[index] = {
        ...viaPoints.value[index],
        lat: lngLat.lat,
        lon: lngLat.lng
    };
    calculateRoute();
}

function removeViaPoint(index: number) {
    viaPoints.value.splice(index, 1);
    calculateRoute();
}

// Route calculation
async function calculateRoute() {
    if (!startPoint.value || !endPoint.value) {
        return;
    }

    try {
        calculating.value = true;

        const response = await commonApi.planRoute({
            startPoint: startPoint.value,
            endPoint: endPoint.value,
            viaPoints: viaPoints.value
        });

        // Parse GeoJSON
        routeGeoJson.value = JSON.parse(response.geojson);
        routeSegments.value = response.segments;

    } catch (error) {
        console.error('Error calculating route:', error);
        // Display error to user
        alert('Failed to calculate route. Please try different points.');
    } finally {
        calculating.value = false;
    }
}

// Clear all route points
function clearRoute() {
    startPoint.value = null;
    endPoint.value = null;
    viaPoints.value = [];
    routeGeoJson.value = null;
    routeSegments.value = [];
}

// Format trip title if no label exists
function formatTripTitle(trip: Trip) {
    if (!trip.startTime) return 'Untitled Trip';

    try {
        const date = new Date(
            trip.startTime.year,
            trip.startTime.month - 1,
            trip.startTime.day,
            trip.startTime.hours,
            trip.startTime.minutes,
            trip.startTime.seconds
        );

        return `Trip on ${date.toLocaleDateString(undefined, {
            month: 'long',
            day: 'numeric',
            year: 'numeric'
        })}`;
    } catch (_e: any) {
        return 'Untitled Trip';
    }
}

// Estimate route distance
function estimateRouteDistance(): number {
    if (!routeSegments.value.length) return 0;

    // Sum up lengths of individual segments
    let totalDistance = 0;

    for (const segment of routeSegments.value) {
        // Use coordinates to calculate segment length
        if (segment.coordinates && segment.coordinates.length > 1) {
            for (let i = 0; i < segment.coordinates.length - 1; i++) {
                const p1 = segment.coordinates[i];
                const p2 = segment.coordinates[i + 1];

                // Calculate haversine distance
                totalDistance += calculateHaversineDistance(
                    p1.lat, p1.lon,
                    p2.lat, p2.lon
                );
            }
        }
    }

    return totalDistance;
}

// Calculate haversine distance between two points
function calculateHaversineDistance(
    lat1: number, lon1: number,
    lat2: number, lon2: number
): number {
    const R = 6371000; // Earth radius in meters
    const φ1 = lat1 * Math.PI / 180;
    const φ2 = lat2 * Math.PI / 180;
    const Δφ = (lat2 - lat1) * Math.PI / 180;
    const Δλ = (lon2 - lon1) * Math.PI / 180;

    const a = Math.sin(Δφ / 2) * Math.sin(Δφ / 2) +
        Math.cos(φ1) * Math.cos(φ2) *
        Math.sin(Δλ / 2) * Math.sin(Δλ / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

    return R * c; // Distance in meters
}
</script>

<style lang="css">
html,
body,
#__nuxt {
    height: 100%;
}

.drag-handle {
    cursor: move;
}

.point-marker {
    transition: all 0.2s;
}

.point-marker:hover {
    transform: scale(1.5);
}

.ghost-point {
    opacity: 0.5;
}

/* Route colors */
.start-point-marker {
    background-color: #22c55e;
    color: white;
}

.end-point-marker {
    background-color: #ef4444;
    color: white;
}

.via-point-marker {
    background-color: #3b82f6;
    color: white;
}

/* Maplibre overrides */
.maplibregl-popup-content {
    padding: 10px;
    border-radius: 4px;
}

.maplibregl-ctrl-group {
    border-radius: 4px;
    overflow: hidden;
}
</style>