<template>
    <div class="w-full h-full">
        <canvas ref="canvas" class="w-full h-full" />
    </div>
</template>

<script setup lang="ts">
import { createHostInterfaceManager, type HostInterface } from 'streets-gl-lib'
import { UUID } from '../model/common_pb';

const { tripsApi } = useApi();
const goodTrip = await tripsApi.getTripDetails({ tripId: new UUID({ value: "8faa72ce-c87e-4aee-97be-73427facdcbc" }) });

const canvas: Ref<HTMLCanvasElement | null> = ref(null)

onMounted(async () => {
    const hostInterface: HostInterface = {
        canvas: canvas.value!,
        eventHandlers: {
            fileLoadingProgressUpdate: (_pct) => { },
            frameTimeUpdate(_frameTime) {

            },
            loadingFile(_fileName) {

            },
            activeFeatureChanged(_type, _id) {

            },
        },
        parameterProvider(_deltaTime) {
            return {
                highlightObjects: {
                    osmIds: goodTrip.trip!.matchedSegments.map(ms => Number(ms.osmWayId))
                },
                mapTime: 0
            }
        },
        systems: {
            picking: true,
            vehicle: false
        },
        baseUrl: '/streets-gl/resources'
    };
    const hostInterfaceManager = createHostInterfaceManager();
    await hostInterfaceManager.init(hostInterface);
})


</script>