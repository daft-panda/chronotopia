<template>
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-3xl font-bold mb-6">ChronoTopia Trips</h1>

        <div class="flex flex-col md:flex-row gap-4 h-full">
            <!-- Trip list and timeline container - takes full width on mobile, left 50% on desktop -->
            <div class="w-full md:w-1/2 flex flex-col">
                <!-- Trip cards grid -->
                <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
                    <div v-for="trip in trips.filter(t => t.status !== MapMatchingStatus.FAILED)" :key="trip.index"
                        class="trip-card bg-white rounded-lg shadow-md overflow-hidden cursor-pointer"
                        :class="{ 'border-2 border-blue-500': selectedTrip?.index === trip.index }"
                        @click="selectTrip(trip)">
                        <div class="relative h-32 bg-gray-200">
                            <!-- Placeholder image for trip visualization -->
                            <div class="absolute inset-0 flex items-center justify-center">
                                <div v-if="trip.status === MapMatchingStatus.COMPLETED" alt="Trip Map"
                                    class="w-full h-full object-cover">DONE</div>
                                <div v-else class="w-full h-full flex items-center justify-center bg-gray-700">
                                    <span class="text-white font-medium">{{ getStatusDisplay(trip.status) }}</span>
                                </div>
                            </div>

                            <!-- Status badge -->
                            <div class="absolute top-2 right-2 px-2 py-1 rounded-full text-xs font-medium"
                                :class="getStatusBadgeClass(trip.status)">
                                {{ getStatusDisplay(trip.status) }}
                            </div>
                        </div>

                        <div class="p-4">
                            <div class="flex justify-between mb-2">
                                <h3 class="font-bold text-lg">Trip {{ trip.index }}</h3>
                                <span v-if="trip.durationSeconds" class="text-sm text-gray-500">{{
                                    formatDuration(trip.durationSeconds) }}</span>
                            </div>

                            <div class="text-sm text-gray-600 mb-2">
                                {{ formatDateTime(trip.start) }}
                            </div>

                            <div class="flex justify-between text-xs text-gray-500">
                                <span>{{ trip.pointCount }} points</span>
                                <span v-if="trip.status === MapMatchingStatus.COMPLETED">{{ trip.matchedSegmentCount }}
                                    segments</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Timeline section - always at bottom 20% -->
                <div class="mt-auto h-48 bg-gray-900 rounded-lg shadow-md p-4">
                    <h2 class="text-xl font-bold mb-4">Timeline</h2>
                    <ClientOnly>
                        <div class="h-32 w-full">
                            <Timeline :items="timelineItems" :start-time="timelineConfig.startTime"
                                :end-time="timelineConfig.endTime" :show-dates="true" :zoom="false"
                                @click-item="handleTimelineItemClick" />
                        </div>
                    </ClientOnly>
                </div>
            </div>

            <!-- Map container - right 50% on desktop, hidden on mobile unless a trip is selected -->
            <div v-if="selectedTrip"
                class="w-full md:w-1/2 h-[600px] md:h-auto bg-gray-100 rounded-lg shadow-md overflow-hidden">
                <div class="relative h-full">
                    <ClientOnly>
                        <MglMap map-style="https://api.maptiler.com/maps/streets/style.json?key=Ic6Mr5qetb5kn90hyEzO"
                            :zoom="12" :center="getMapCenter()">
                            <MglNavigationControl position="top-right" />
                            <MglGeolocateControl position="top-right" />

                            <MglGeoJsonSource v-if="selectedTripGeoJson" source-id="trip-route"
                                :data="selectedTripGeoJson">
                                <MglLineLayer layer-id="trip-route-line" :paint="{
                                    'line-color': '#FF0000',
                                    'line-width': 4
                                }" />
                            </MglGeoJsonSource>

                            <!-- Start point marker -->
                            <MglMarker v-if="selectedTrip.startPoint"
                                :coordinates="[selectedTrip.startPoint.lon, selectedTrip.startPoint.lat]">
                                <template #marker>
                                    <div class="w-4 h-4 rounded-full bg-green-500 border-2 border-white" />
                                </template>
                            </MglMarker>

                            <!-- End point marker -->
                            <MglMarker v-if="selectedTrip.endPoint"
                                :coordinates="[selectedTrip.endPoint.lon, selectedTrip.endPoint.lat]">
                                <template #marker>
                                    <div class="w-4 h-4 rounded-full bg-red-500 border-2 border-white" />
                                </template>
                            </MglMarker>
                        </MglMap>
                    </ClientOnly>

                    <div class="absolute top-4 left-4 z-10">
                        <button v-if="selectedTrip.status === MapMatchingStatus.COMPLETED"
                            class="bg-blue-500 text-white px-4 py-2 rounded-lg shadow hover:bg-blue-600 transition"
                            @click="viewMapMatching">
                            View Map Matching
                        </button>
                    </div>
                </div>
            </div>

            <!-- Placeholder when no trip is selected on mobile -->
            <div v-else
                class="hidden md:block md:w-1/2 h-[600px] md:h-auto bg-gray-100 rounded-lg shadow-md flex items-center justify-center">
                <p class="text-gray-500 text-lg">Select a trip to view the map</p>
            </div>
        </div>
    </div>
</template>

<script lang="ts" setup>
import { MglMap, MglNavigationControl, MglGeolocateControl, MglGeoJsonSource, MglLineLayer, MglMarker } from '#components';
import { ref, onMounted, computed } from 'vue';
import { MapMatchingStatus, type TripSummary } from '~/model/chronotopia_pb';
import { Timeline, type TimelineItem, type TimelineItemRange } from 'vue-timeline-chart'
import 'vue-timeline-chart/style.css'
import type { LngLatLike } from 'maplibre-gl';
import type { DateTime } from '~/model/datetime_pb';

const { $api } = useNuxtApp();
const router = useRouter();

const trips: Ref<TripSummary[]> = ref([]);
const selectedTrip: Ref<TripSummary | null> = ref(null);
const selectedTripGeoJson = ref(null);

onMounted(async () => {
    try {
        const response = await $api.getTripSummaries({});
        trips.value = response.summaries;
    } catch (error) {
        console.error('Error fetching trips:', error);
    }
});

const selectTrip = async (trip: TripSummary) => {
    selectedTrip.value = trip;

    // Only fetch GeoJSON for completed trips
    if (trip.status === MapMatchingStatus.COMPLETED) {
        try {
            const geoJsonResponse = await $api.getTripGeoJSON({ tripIndex: trip.index });
            selectedTripGeoJson.value = JSON.parse(geoJsonResponse.value);
        } catch (error) {
            console.error('Error fetching trip GeoJSON:', error);
            selectedTripGeoJson.value = null;
        }
    } else {
        selectedTripGeoJson.value = null;
    }
};

const viewMapMatching = () => {
    if (selectedTrip.value) {
        router.push(`/analyze-mapmatching/${selectedTrip.value.index}`);
    }
};

const getMapCenter = (): LngLatLike => {
    if (selectedTrip.value && selectedTrip.value.startPoint) {
        return [selectedTrip.value.startPoint.lon, selectedTrip.value.startPoint.lat];
    }
    return [-43.4795272, -22.7384021]; // Default center
};

// Convert DateTime object to JavaScript Date
const dateTimeToJsDate = (dateTime: DateTime) => {
    if (!dateTime) return null;

    // Extract datetime components
    const { year, month, day, hours, minutes, seconds, nanos } = dateTime;

    // JavaScript months are 0-based (0-11), while DateTime months are 1-based (1-12)
    return new Date(year, month - 1, day, hours, minutes, seconds, nanos / 1000000);
};

// Format DateTime to readable string
const formatDateTime = (dateTime: DateTime) => {
    if (!dateTime) return '';

    const jsDate = dateTimeToJsDate(dateTime);
    if (!jsDate) return '';

    return jsDate.toLocaleString();
};

// Format duration from seconds to readable string
const formatDuration = (seconds: number) => {
    if (!seconds) return '';

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);

    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
};

// Get status display text
const getStatusDisplay = (status: MapMatchingStatus) => {
    const statusMap: Record<string, string> = {
        'PENDING': 'Pending',
        'PROCESSING': 'Processing',
        'COMPLETED': 'Completed',
        'FAILED': 'Failed',
        'UNKNOWN': 'Unknown'
    };

    return statusMap[MapMatchingStatus[status]] || 'Unknown';
};

// Get status badge class
const getStatusBadgeClass = (status: MapMatchingStatus) => {
    const classMap: Record<string, string> = {
        'PENDING': 'bg-yellow-500 text-white',
        'PROCESSING': 'bg-blue-500 text-white',
        'COMPLETED': 'bg-green-500 text-white',
        'FAILED': 'bg-red-500 text-white',
        'UNKNOWN': 'bg-gray-500 text-white'
    };

    return classMap[MapMatchingStatus[status]] || 'bg-gray-500 text-white';
};

// Get DateTime timestamp in milliseconds for timeline
const getDateTimeTimestamp = (dateTime: DateTime) => {
    if (!dateTime) return Date.now();

    const jsDate = dateTimeToJsDate(dateTime);
    return jsDate ? jsDate.getTime() : Date.now();
};

// Timeline configuration
const timelineConfig = {
    startTime: computed(() => {
        if (trips.value.length === 0) return new Date().getTime() - 86400000; // 1 day ago
        const firstTrip = trips.value.reduce((earliest: TripSummary, trip) => {
            if (!trip.start) return earliest;
            return (!earliest.start || getDateTimeTimestamp(trip.start) < getDateTimeTimestamp(earliest.start)) ? trip : earliest;
        }, {});
        return firstTrip.start ? dateTimeToJsDate(firstTrip.start) : new Date(Date.now() - 86400000);
    }),
    endTime: computed(() => {
        if (trips.value.length === 0) return new Date();
        const lastTrip = trips.value.reduce((latest: TripSummary, trip) => {
            if (!trip.end) return latest;
            return (!latest.end || getDateTimeTimestamp(trip.end) > getDateTimeTimestamp(latest.end)) ? trip : latest;
        }, {});
        return lastTrip.end ? dateTimeToJsDate(lastTrip.end) : new Date();
    }),
};

// Create timeline items for vue-timeline-chart
const timelineItems = computed((): TimelineItem[] => {
    return trips.value.map(trip => {
        if (!trip.start || !trip.end) return null;

        // Map status to color
        const colorMap: Record<string, string> = {
            'PENDING': '#EAB308', // yellow
            'PROCESSING': '#3B82F6', // blue
            'COMPLETED': '#10B981', // green
            'FAILED': '#EF4444', // red
            'UNKNOWN': '#6B7280' // gray
        };

        return {
            id: `${trip.index}`,
            label: `Trip ${trip.index}`,
            start: Number(trip.start.seconds),
            end: Number(trip.end.seconds),
            color: colorMap[MapMatchingStatus[trip.status]] || '#6B7280',
            type: 'range',
            group: 'main'
        } as TimelineItemRange;
    }).filter(v => v !== null);
});

// Handle timeline item click
const handleTimelineItemClick = (item) => {
    const trip = trips.value.find(t => t.index === item.id);
    if (trip) {
        selectTrip(trip);
    }
};
</script>