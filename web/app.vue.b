<template>
    <div class="main-container">

        <!-- Add the route debugger component -->
        <RouteDebugger v-if="showRouteDebugger" @add-map-click-listener="addMapClickListener"
            @remove-map-click-listener="removeMapClickListener" @add-map-source="addMapSource"
            @add-map-layer="addMapLayer" @update-map-source-data="updateMapSourceData"
            @set-layer-visibility="setLayerVisibility" @fit-bounds="fitBounds" @fly-to="flyTo"
            @set-cursor="setCursor" />

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


                </MglMap>
            </ClientOnly>
        </div>

    </div>
</template>

<script>
import RouteDebugger from './RouteDebugger.vue';

export default defineComponent({
    components: {
        RouteDebugger
    },
    data() {
        return {
            // Add this to your existing data
            showRouteDebugger: false,
            map: import.meta.client ? useMglMap() : null,
        };
    },
    mounted() {
        if (!import.meta.client) {
            return;
        }

        watch(() => this.map.isLoaded, () => { this.showRouteDebugger = true });
    },
    methods: {
        // Add this to your methods
        toggleRouteDebugger() {
            this.showRouteDebugger = !this.showRouteDebugger;
        },

        // Add these map control methods if they don't already exist
        addMapClickListener(listener) {
            if (this.map.map) {
                this.map.map.on('click', listener);
            }
        },
        removeMapClickListener(listener) {
            if (this.map.map) {
                this.map.map.off('click', listener);
            }
        },
        addMapSource(id, source) {
            if (this.map.map && !this.map.map.getSource(id)) {
                this.map.map.addSource(id, source);
            }
        },
        addMapLayer(layer) {
            if (this.map.map && !this.map.map.getLayer(layer.id)) {
                this.map.map.addLayer(layer);
            }
        },
        updateMapSourceData(sourceId, data) {
            if (this.map.map && this.map.map.getSource(sourceId)) {
                this.map.map.getSource(sourceId).setData(data);
            }
        },
        setLayerVisibility(layerId, visibility) {
            if (this.map.map && this.map.map.getLayer(layerId)) {
                this.map.map.setLayoutProperty(layerId, 'visibility', visibility);
            }
        },
        fitBounds(bounds, options) {
            if (this.map.map) {
                this.map.map.fitBounds(bounds, options || {});
            }
        },
        flyTo(options) {
            if (this.map.map) {
                this.map.map.flyTo(options);
            }
        },
        setCursor(cursor) {
            if (this.map.map) {
                this.map.map.getCanvas().style.cursor = cursor;
            }
        }
    }
});
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
    width: 100%;
    display: flex;
    flex-direction: column;
}

.map-container {
    height: 100%;
    width: 100%;
}
</style>