<template>
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-3xl font-bold mb-6">Processing Status</h1>

        <div v-if="pending" class="w-full flex justify-center items-center py-12">
            <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500" />
        </div>

        <div v-else-if="error" class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6" role="alert">
            <p class="font-bold">Error</p>
            <p>{{ error.message }}</p>
        </div>

        <div v-else class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <!-- Processing overview card -->
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-xl font-semibold mb-4">Processing Overview</h2>

                <div class="flex items-center justify-between mb-3">
                    <span class="text-gray-600">Data processed up to:</span>
                    <span class="font-medium">{{ formatDateTime(processingState?.state?.lastProcessedTime) }}</span>
                </div>

                <div class="flex items-center justify-between mb-3">
                    <span class="text-gray-600">Total trips generated:</span>
                    <span class="font-medium">{{ processingState?.state?.totalTripsGenerated || 0 }}</span>
                </div>

                <div class="flex items-center justify-between mb-3">
                    <span class="text-gray-600">Visits detected:</span>
                    <span class="font-medium">{{ processingState?.state?.totalVisitsDetected || 0 }}</span>
                </div>

                <div class="flex items-center justify-between mb-6">
                    <span class="text-gray-600">Last updated:</span>
                    <span class="font-medium">{{ formatDateTime(processingState?.state?.lastUpdated) }}</span>
                </div>

                <button
                    class="w-full py-2 px-4 bg-blue-500 hover:bg-blue-600 text-white rounded-lg shadow transition-colors"
                    :disabled="isProcessingTriggered" @click="triggerProcessing">
                    {{ isProcessingTriggered ? 'Processing...' : 'Process New Data' }}
                </button>
            </div>

            <!-- Import summary card -->
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-xl font-semibold mb-4">Import Summary</h2>

                <div v-if="processingState?.imports && processingState.imports.length > 0">
                    <div class="overflow-hidden rounded-lg border border-gray-200 mb-3">
                        <table class="min-w-full divide-y divide-gray-200">
                            <thead class="bg-gray-50">
                                <tr>
                                    <th
                                        class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Source</th>
                                    <th
                                        class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Date</th>
                                    <th
                                        class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Items</th>
                                    <th
                                        class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Status</th>
                                </tr>
                            </thead>
                            <tbody class="bg-white divide-y divide-gray-200">
                                <tr v-for="import_ in processingState.imports" :key="import_?.id?.value"
                                    class="hover:bg-gray-50">
                                    <td class="px-3 py-2 whitespace-nowrap">
                                        <div class="text-sm font-medium text-gray-900">{{
                                            formatImportType(import_.importType) }}</div>
                                        <div class="text-xs text-gray-500">{{ import_.importName }}</div>
                                    </td>
                                    <td class="px-3 py-2 whitespace-nowrap text-sm text-gray-500">
                                        {{ formatDate(import_.importDateTime) }}
                                    </td>
                                    <td class="px-3 py-2 whitespace-nowrap">
                                        <div class="text-xs text-gray-600">
                                            {{ import_.locationCount }} locations<br>
                                            {{ import_.activityCount }} activities<br>
                                            {{ import_.visitCount }} visits
                                        </div>
                                    </td>
                                    <td class="px-3 py-2 whitespace-nowrap">
                                        <span :class="getStatusClass(import_.processingComplete)">
                                            {{ import_.processingComplete ? 'Completed' : 'Processing' }}
                                        </span>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                <div v-else class="text-center py-8 text-gray-500">
                    No imports available
                </div>
            </div>
        </div>
    </div>
</template>

<script lang="ts" setup>
import { formatDate, formatDateTime } from '~/utils/formatting';

const { tripsApi } = useApi();
const isProcessingTriggered = ref(false);

// Fetch processing state data
const { data: processingState, pending, error, refresh } = useAsyncData(
    'processingState',
    async () => {
        try {
            const response = await tripsApi.getUserProcessingState({});
            return response;
        } catch (err) {
            console.error('Error fetching processing state:', err);
            throw new Error('Failed to load processing data');
        }
    }
);


// Format import type to a readable string
const formatImportType = (type: string) => {
    const typeMap: Record<string, string> = {
        'google_maps_timeline': 'Google Maps',
        'apple_health': 'Apple Health',
        'fitness_app': 'Fitness App',
        'chronotopia_app': 'ChronoTopia App',
        'chronotopia_web': 'ChronoTopia Web',
        'chronotopia_api': 'ChronoTopia API'
    };

    return typeMap[type] || type;
};

// Get status badge class
const getStatusClass = (isComplete: boolean) => {
    return isComplete
        ? 'px-2 py-1 text-xs rounded-full bg-green-100 text-green-800'
        : 'px-2 py-1 text-xs rounded-full bg-yellow-100 text-yellow-800';
};

// Trigger processing
const triggerProcessing = async () => {
    if (isProcessingTriggered.value) return;

    try {
        isProcessingTriggered.value = true;
        const response = await tripsApi.triggerProcessing({});

        if (response.success) {
            // Show success message with toast or notification
            console.log('Processing triggered successfully');

            // Refresh data after a delay to give processing time to start
            setTimeout(() => {
                refresh();
                isProcessingTriggered.value = false;
            }, 2000);
        } else {
            console.error('Failed to trigger processing:', response.message);
            isProcessingTriggered.value = false;
        }
    } catch (err) {
        console.error('Error triggering processing:', err);
        isProcessingTriggered.value = false;
    }
};
</script>