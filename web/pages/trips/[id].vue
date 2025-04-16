<template>
    <div class="container mx-auto px-4 py-8">
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

        <!-- Trip detail content -->
        <div v-else-if="trip" class="bg-white rounded-lg shadow-lg overflow-hidden">
            <!-- Trip header -->
            <div class="bg-gradient-to-r from-blue-500 to-indigo-600 p-6 text-white">
                <h1 class="text-2xl font-bold">{{ trip.label || formatTripTitle(trip) }}</h1>
                <div class="flex items-center mt-2 text-blue-100">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24"
                        stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <span>{{ formatDate(trip.startTime!) }} at {{ formatTime(trip.startTime!) }}</span>
                </div>
            </div>

            <!-- Trip map with visualization -->
            <div class="h-96 bg-gray-200 relative">
                <div v-if="mapData || parsedGeoJson" class="absolute inset-0">
                    <ClientOnly>
                        <MglMap map-style="https://api.maptiler.com/maps/streets/style.json?key=Ic6Mr5qetb5kn90hyEzO"
                            :zoom="10" @load="onMapLoaded">
                            <MglFullscreenControl position="top-right" />
                            <MglNavigationControl position="top-right" />
                            <MglGeolocateControl position="top-right" />

                            <!-- Regular GeoJSON visualization -->
                            <MglGeoJsonSource v-if="parsedGeoJson" source-id="trip-route" :data="parsedGeoJson">
                                <MglLineLayer layer-id="trip-route-line" source="trip-route" :paint="{
                                    'line-color': '#4F46E5',
                                    'line-width': 4
                                }" />
                            </MglGeoJsonSource>

                            <!-- Display trip points as markers -->
                            <MglMarker v-for="(point, index) in trip.points" :key="`point-${index}`"
                                :coordinates="[point.latlon!.lon, point.latlon!.lat]">
                                <template #marker>
                                    <div
                                        class="point-marker w-3 h-3 rounded-full bg-blue-500 border-2 border-white shadow-md" />
                                </template>
                                <MglPopup>
                                    <div class="p-2">
                                        <h3 class="font-bold">Point {{ index }}</h3>
                                        <p class="text-sm">{{ point.latlon?.lat.toFixed(6) }}, {{
                                            point.latlon?.lon.toFixed(6) }}</p>
                                        <p v-if="point.dateTime" class="text-sm text-gray-600">
                                            {{ formatTime(point.dateTime) }}
                                        </p>
                                    </div>
                                </MglPopup>
                            </MglMarker>

                            <!-- Window Traces Visualization (if RouteMatchTrace is available) -->
                            <template v-if="trip.routeMatchTrace">
                                <template v-for="(window, windowIdx) in trip.routeMatchTrace.windowTraces"
                                    :key="`window-${windowIdx}`">
                                    <MglGeoJsonSource :source-id="`window-${windowIdx}`" :data="parsedGeoJson">
                                        <MglLineLayer :layer-id="`window-${windowIdx}`"
                                            :source-id="`window-${windowIdx}`" :paint="{
                                                'line-color': '#00FF00',
                                                'line-width': 5,
                                                'line-dasharray': window.bridge ? [2, 1] : [1, 0],
                                                'line-opacity': 0.8
                                            }" />
                                    </MglGeoJsonSource>
                                </template>
                            </template>
                        </MglMap>
                    </ClientOnly>
                </div>
                <div v-else class="absolute inset-0 flex items-center justify-center text-gray-500">
                    <span v-if="trip.processed">No Map Data</span>
                    <span v-else>Processing</span>
                </div>
            </div>

            <!-- Map controls (if RouteMatchTrace is available) -->
            <div v-if="trip.routeMatchTrace" class="bg-gray-100 p-3 flex justify-between items-center">
                <button class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition"
                    @click="viewDetailedMapMatching">
                    View Detailed Map Matching Analysis
                </button>

                <!-- Simple legend -->
                <div class="flex items-center gap-4">
                    <div class="flex items-center">
                        <div class="w-6 h-2 bg-green-500 mr-2" />
                        <span class="text-xs text-gray-700">Constrained</span>
                    </div>
                    <div class="flex items-center">
                        <div class="w-6 h-2 bg-blue-500 mr-2" />
                        <span class="text-xs text-gray-700">Unconstrained</span>
                    </div>
                    <div class="flex items-center">
                        <div class="w-6 h-2 bg-red-500 mr-2" />
                        <span class="text-xs text-gray-700">Bridge</span>
                    </div>
                </div>
            </div>

            <!-- Trip details -->
            <div class="p-6">
                <!-- Stats summary -->
                <div class="grid grid-cols-3 gap-4 mb-6">
                    <div class="border rounded-lg p-4 text-center">
                        <div class="text-lg font-semibold">{{ formatDuration(trip) }}</div>
                        <div class="text-xs text-gray-500">Duration</div>
                    </div>
                    <div class="border rounded-lg p-4 text-center">
                        <div class="text-lg font-semibold">{{ formatDistance(trip.distanceMeters) }}</div>
                        <div class="text-xs text-gray-500">Distance</div>
                    </div>
                    <div class="border rounded-lg p-4 text-center">
                        <div class="text-lg font-semibold">{{ trip.points?.length || 0 }}</div>
                        <div class="text-xs text-gray-500">Data Points</div>
                    </div>
                </div>

                <!-- Trip notes -->
                <div v-if="trip.notes" class="mb-6">
                    <h3 class="text-lg font-medium mb-2">Notes</h3>
                    <p class="text-gray-700">{{ trip.notes }}</p>
                </div>

                <!-- Visits during trip -->
                <div v-if="trip.visits && trip.visits.length > 0" class="mb-6">
                    <h3 class="text-lg font-medium mb-2">Visits</h3>
                    <div class="space-y-3">
                        <div v-for="(visit, index) in trip.visits" :key="index" class="border rounded-lg p-3">
                            <div class="flex items-start">
                                <div class="bg-blue-100 p-2 rounded-full mr-3">
                                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-blue-500" fill="none"
                                        viewBox="0 0 24 24" stroke="currentColor">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                            d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                            d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                                    </svg>
                                </div>
                                <div>
                                    <div class="font-medium">{{ visit.canonicalLabel || 'Unknown location' }}</div>
                                    <div class="text-sm text-gray-500">
                                        {{ formatTime(visit.arrival!) }} - {{ formatTime(visit.departure!) }}
                                    </div>
                                    <div class="text-xs text-gray-400">
                                        {{ formatDurationFromVisit(visit) }}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Activities during trip -->
                <div v-if="trip.activities && trip.activities.length > 0">
                    <h3 class="text-lg font-medium mb-2">Activities</h3>
                    <div class="space-y-3">
                        <div v-for="(activity, index) in trip.activities" :key="index" class="border rounded-lg p-3">
                            <div class="flex items-start">
                                <div class="bg-green-100 p-2 rounded-full mr-3">
                                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-green-500" fill="none"
                                        viewBox="0 0 24 24" stroke="currentColor">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                            d="M13 10V3L4 14h7v7l9-11h-7z" />
                                    </svg>
                                </div>
                                <div>
                                    <div class="font-medium">{{ ActivityEvent_ActivityType[activity.type] }}</div>
                                    <div class="text-sm text-gray-500">
                                        {{ formatTime(activity.start) }}
                                        <span v-if="activity.end">- {{ formatTime(activity.end) }}</span>
                                    </div>
                                    <div class="text-xs text-gray-400 mt-1">
                                        <span v-if="activity.distance">Distance: {{ formatDistance(activity.distance)
                                            }}</span>
                                        <span v-if="activity.stepCount" class="ml-3">Steps: {{ activity.stepCount
                                            }}</span>
                                    </div>
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
import { ActivityEvent_ActivityType } from '~/model/ingest_pb';
import type { Trip } from '~/model/trips_pb';
import { formatDate, formatDistance, formatDuration, formatDurationFromVisit, formatTime } from '~/utils/formatting';
import { MglMap, MglNavigationControl, MglFullscreenControl, MglGeolocateControl, MglGeoJsonSource, MglLineLayer, MglMarker, MglPopup } from '#components';
import { LngLatBounds, type Map as MaplibreMap } from 'maplibre-gl';
import type { GeoJSON } from 'geojson';

const { tripsApi } = useApi();
const router = useRouter();
const route = useRoute();
const tripId = route.params.id;
const map = useMglMap();
const mapLoaded = ref(false);
const parsedGeoJson = ref<GeoJSON | null>(null);
const mapData = ref<any>(null);
const mapInstance = ref<MaplibreMap | null>(null);

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

            // Try to parse the GeoJSON if available
            if (response.trip?.geojson) {
                try {
                    parsedGeoJson.value = JSON.parse(response.trip.geojson);
                } catch (err) {
                    console.error('Error parsing GeoJSON:', err);
                }
            }

            return response.trip;
        } catch (err) {
            console.error(`Error fetching trip ${tripId}:`, err);
            throw new Error('Failed to load trip details');
        }
    }
);

// Fit map to bounds of the trip points
const fitMapToBounds = () => {
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
            maxZoom: 15
        });
    }
};

watch(() => map.isLoaded, (_isLoaded) => {
    mapInstance.value = map.map;
    mapLoaded.value = true;

    // Fit bounds to the trip points if available
    fitMapToBounds();
}, { immediate: true });

// Navigate to the detailed map matching view
const viewDetailedMapMatching = () => {
    router.push(`/analyze-mapmatching/${tripId}`);
};

// Generate a title for the trip if no label exists
const formatTripTitle = (trip: Trip) => {
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
};

</script>

<style>
.point-marker {
    transition: all 0.2s;
}

.point-marker:hover {
    transform: scale(1.5);
}
</style>