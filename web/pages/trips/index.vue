<template>
    <div class="container mx-auto px-4 py-8">
        <div class="flex justify-between items-center mb-6">
            <h1 class="text-3xl font-bold">My Trips</h1>

            <div class="flex space-x-3">
                <NuxtLink to="/processing"
                    class="bg-gray-100 hover:bg-gray-200 text-gray-700 px-4 py-2 rounded-lg transition-colors">
                    View Processing Status
                </NuxtLink>
                <button class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors"
                    @click="async () => { await refresh() }">
                    Refresh
                </button>
            </div>
        </div>

        <!-- Google Timeline Uploader Component -->
        <GoogleTimelineUploader @upload-success="handleUploadSuccess" />

        <!-- Loading state -->
        <div v-if="pending" class="w-full flex justify-center items-center py-12">
            <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500" />
        </div>

        <!-- Error state -->
        <div v-else-if="error" class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6" role="alert">
            <p class="font-bold">Error</p>
            <p>{{ error.message }}</p>
        </div>

        <!-- No trips state -->
        <div v-else-if="!trips?.length" class="text-center py-12">
            <div class="mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-16 w-16 mx-auto text-gray-400" fill="none"
                    viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                        d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                </svg>
            </div>
            <h2 class="text-xl font-semibold text-gray-700">No trips found</h2>
            <p class="text-gray-500 mt-2">Start recording your travels with ChronoTopia or upload a Google Maps Timeline
                export</p>
        </div>

        <!-- Trips list -->
        <div v-else class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div v-for="trip in trips" :key="trip?.id?.value"
                class="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow cursor-pointer"
                @click="viewTripDetails(trip)">
                <!-- Trip map preview - placeholder -->
                <div class="h-48 bg-gray-200 relative">
                    <div v-if="trip.geojson" class="absolute inset-0">
                        <!-- In a real app, render the GeoJSON with a map library -->
                        <div class="w-full h-full flex items-center justify-center bg-blue-50">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-blue-500" fill="none"
                                viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                    d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                            </svg>
                        </div>
                    </div>
                    <div v-else class="absolute inset-0 flex items-center justify-center text-gray-500">
                        <span v-if="trip.processed">No Map Data</span>
                        <span v-else>Processing</span>
                    </div>

                    <!-- Trip stats overlay -->
                    <div
                        class="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-3 text-white">
                        <div class="flex justify-between items-end">
                            <div>
                                <div class="font-semibold">{{ formatDate(trip.startTime!) }}</div>
                                <div class="text-sm opacity-80">{{ formatDuration(trip) }}</div>
                            </div>
                            <div class="text-right">
                                <div class="font-semibold">{{ formatDistance(trip.distanceMeters) }}</div>
                                <div class="text-sm opacity-80">{{ trip.points.length }} points</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Trip details -->
                <div class="p-4">
                    <div class="flex justify-between items-start mb-2">
                        <h3 class="font-semibold text-lg">
                            {{ trip.label || formatTripTitle(trip) }}
                        </h3>
                        <span v-if="!trip.processed"
                            class="px-2 py-1 text-xs rounded-full bg-yellow-100 text-yellow-800">
                            Processing
                        </span>
                    </div>

                    <p v-if="trip.notes" class="text-gray-600 text-sm line-clamp-2">
                        {{ trip.notes }}
                    </p>
                    <p v-else class="text-gray-400 text-sm italic line-clamp-2">
                        No description
                    </p>

                    <!-- Visit indicators -->
                    <div v-if="trip.visits && trip.visits.length > 0"
                        class="mt-3 flex items-center text-xs text-gray-500">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24"
                            stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                        {{ trip.visits.length }} {{ trip.visits.length === 1 ? 'visit' : 'visits' }}
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script lang="ts" setup>
import { formatDate, formatDistance, formatDuration } from '~/utils/formatting';
import type { Trip } from '~/model/trips_pb';
import GoogleTimelineUploader from '~/components/GoogleTimelineUploader.vue';

const { tripsApi } = useApi();
const router = useRouter();

// Fetch trips
const { data, pending, error, refresh } = useAsyncData(
    'trips',
    async () => {
        try {
            const response = await tripsApi.getTripsForUser({
                limit: 100
            });

            return response.trips?.trips || [];
        } catch (err) {
            console.error('Error fetching trips:', err);
            throw new Error('Failed to load trips');
        }
    }
);

// Extract trips from the response
const trips = computed(() => data.value);

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
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (_e: any) {
        return 'Untitled Trip';
    }
};

// View trip details
const viewTripDetails = (trip: Trip) => {
    if (!trip.id?.value) return;
    router.push(`/trips/${trip.id.value}`);
};

// Handle successful upload
const handleUploadSuccess = async (uploadData: any) => {
    // Show a notification or perform any actions needed after upload
    console.log('Upload successful:', uploadData);

    // Refresh the trips list after a short delay to allow for processing
    setTimeout(async () => {
        await refresh();
    }, 2000);
};
</script>