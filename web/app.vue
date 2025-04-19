<script setup>
import { useDevice } from '#imports';
const device = useDevice();

// Conditionally add the CSS file based on device detection
if (device.isMobile) {
    useHead({
        link: [
            {
                rel: 'stylesheet',
                href: '/css/native-mobile.css' // Path to your smartphone-specific CSS file
            }
        ]
    });
}
</script>


<template>
    <div class="h-full w-full">
        <NuxtPage />
    </div>
</template>