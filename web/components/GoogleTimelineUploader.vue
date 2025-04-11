<template>
    <div class="bg-white rounded-lg shadow-md p-6 mb-6">
        <h2 class="text-xl font-semibold mb-4">Upload Google Maps Timeline</h2>

        <div class="space-y-4">
            <div class="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-500 transition-colors cursor-pointer"
                :class="{ 'border-blue-500 bg-blue-50': isDragging }" @click="triggerFileInput"
                @dragover.prevent="isDragging = true" @dragleave.prevent="isDragging = false"
                @drop.prevent="handleFileDrop">

                <div v-if="isUploading" class="flex flex-col items-center justify-center">
                    <div class="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-blue-500 mb-3" />
                    <p class="text-gray-600">Uploading your timeline...</p>
                </div>

                <div v-else class="flex flex-col items-center justify-center">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 text-gray-400 mb-3" fill="none"
                        viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    <p class="text-gray-600 mb-1">Drag and drop your Google Timeline JSON file here</p>
                    <p class="text-gray-500 text-sm">or click to select file</p>
                </div>
            </div>

            <input ref="fileInput" type="file" accept=".json,application/json" class="hidden"
                @change="handleFileChange">

            <div v-if="selectedFile" class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div class="flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-500 mr-2" fill="none"
                        viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <div>
                        <p class="text-sm font-medium">{{ selectedFile.name }}</p>
                        <p class="text-xs text-gray-500">{{ formatFileSize(selectedFile.size) }}</p>
                    </div>
                </div>
                <button v-if="!isUploading" class="text-red-500 hover:text-red-700 text-sm" @click="removeFile">
                    Remove
                </button>
            </div>

            <div class="flex justify-between items-center">
                <input v-model="exportName" type="text" placeholder="Export name (optional)"
                    class="flex-grow mr-3 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    :disabled="isUploading">

                <button
                    class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
                    :disabled="!canUpload || isUploading" @click="uploadFile">
                    Upload Timeline
                </button>
            </div>
        </div>

        <!-- Success message -->
        <div v-if="uploadMessage" :class="[
            'mt-4 p-3 rounded-lg',
            uploadSuccess ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
        ]">
            {{ uploadMessage }}
        </div>
    </div>
</template>

<script lang="ts" setup>
import { ref, computed } from 'vue';
import { GoogleMapsTimelineExport } from '~/model/ingest_pb';
import { DateTime } from '~/model/common_pb';

const { ingestApi } = useApi();
const emit = defineEmits(['upload-success']);

// State
const fileInput = ref<HTMLInputElement | null>(null);
const selectedFile = ref<File | null>(null);
const isDragging = ref(false);
const isUploading = ref(false);
const uploadMessage = ref('');
const uploadSuccess = ref(false);
const exportName = ref('');

// Computed
const canUpload = computed(() => selectedFile.value !== null);

// Methods
const triggerFileInput = () => {
    fileInput.value?.click();
};

const handleFileChange = (event: Event) => {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
        selectedFile.value = input.files[0];
        uploadMessage.value = '';
    }
};

const handleFileDrop = (event: DragEvent) => {
    isDragging.value = false;

    if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
        const file = event.dataTransfer.files[0];

        // Check if file is JSON
        if (file.type === 'application/json' || file.name.endsWith('.json')) {
            selectedFile.value = file;
            uploadMessage.value = '';
        } else {
            uploadMessage.value = 'Please upload a JSON file from Google Timeline.';
            uploadSuccess.value = false;
        }
    }
};

const removeFile = () => {
    selectedFile.value = null;
    if (fileInput.value) {
        fileInput.value.value = '';
    }
    uploadMessage.value = '';
};

const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) {
        return bytes + ' bytes';
    } else if (bytes < 1024 * 1024) {
        return (bytes / 1024).toFixed(1) + ' KB';
    } else {
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }
};

const uploadFile = async () => {
    if (!selectedFile.value) return;

    isUploading.value = true;
    uploadMessage.value = '';

    try {
        // Read the file content
        const fileContent = await readFileAsText(selectedFile.value);

        // Create the request payload
        const now = new Date();
        const currentDateTime = new DateTime({
            year: now.getFullYear(),
            month: now.getMonth() + 1,
            day: now.getDate(),
            hours: now.getHours(),
            minutes: now.getMinutes(),
            seconds: now.getSeconds()
        });

        const exportData = new GoogleMapsTimelineExport({
            jsonContent: fileContent,
            exportName: exportName.value || selectedFile.value.name,
            exportDate: currentDateTime
        });

        // Submit the export
        const response = await ingestApi.submitGoogleMapsTimelineExport(exportData);

        if (response.success) {
            uploadSuccess.value = true;
            uploadMessage.value = 'Timeline uploaded successfully! Processing your data...';

            // Emit an event to notify parent component
            emit('upload-success', {
                fileName: selectedFile.value.name,
                exportName: exportName.value,
                response
            });

            // Reset the form
            selectedFile.value = null;
            exportName.value = '';
            if (fileInput.value) {
                fileInput.value.value = '';
            }
        } else {
            uploadSuccess.value = false;
            uploadMessage.value = `Upload failed: ${response.alertMessage || 'Unknown error'}`;
        }
    } catch (error: any) {
        console.error('Error uploading timeline:', error);
        uploadSuccess.value = false;
        uploadMessage.value = `Error uploading timeline: ${error.message || 'Unknown error'}`;
    } finally {
        isUploading.value = false;
    }
};

const readFileAsText = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            if (typeof reader.result === 'string') {
                resolve(reader.result);
            } else {
                reject(new Error('Failed to read file as text'));
            }
        };
        reader.onerror = () => reject(reader.error);
        reader.readAsText(file);
    });
};
</script>