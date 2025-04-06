<template>
    <div class="route-debugger">
        <div class="debugger-controls">
            <div class="debugger-header">
                <h3>Route Debugger</h3>
                <div class="mode-indicator">{{ modeDisplay }}</div>
            </div>
            <div class="button-group">
                <button class="debug-button" @click="resetDebugger">Reset</button>
                <button class="debug-button" :class="{ active: true }" :disabled="!readyToFindRoute" @click="findRoute">
                    {{ isLoading ? 'Working...' : 'Find Route' }}
                </button>
                <button class="debug-button" :class="{ active: showNetworkDetails }" :disabled="!routeResult"
                    @click="showNetworkDetails = !showNetworkDetails">
                    {{ showNetworkDetails ? 'Hide Network' : 'Show Network' }}
                </button>
            </div>
        </div>

        <div v-if="error" class="error-message">
            {{ error }}
        </div>

        <!-- Results Panel -->
        <div v-if="routeResult" class="results-panel">
            <div class="results-summary">
                <div class="summary-box">
                    <div class="summary-title">Route Status</div>
                    <div class="summary-content" :class="routeResult.status === 'success' ? 'success' : 'failure'">
                        {{ routeResult.status === 'success' ? 'SUCCESS' : 'FAILED' }}
                    </div>
                    <div v-if="routeResult.status === 'failed'" class="failure-reason">
                        {{ routeResult.failureReason }}
                    </div>
                </div>

                <div class="summary-box">
                    <div class="summary-title">Distance</div>
                    <div class="summary-content">
                        {{ (routeResult.directDistance || 0).toFixed(0) }}m (direct)<br>
                        {{ (routeResult.totalDistance || 0).toFixed(0) }}m (path)
                    </div>
                </div>

                <div class="summary-box">
                    <div class="summary-title">Segments</div>
                    <div class="summary-content">
                        <div>From: {{ markerASegment?.osm_way_id || '?' }}</div>
                        <div>To: {{ markerBSegment?.osm_way_id || '?' }}</div>
                        <div>Path: {{ routeResult.route ? routeResult.route.length : 0 }} segments</div>
                    </div>
                </div>
            </div>

            <!-- Pathfinding Details Section -->
            <div class="debug-section">
                <h4>A* Pathfinding Analysis</h4>

                <div class="path-stats">
                    <div class="stat-row">
                        <span class="stat-label">Algorithm:</span>
                        <span class="stat-value">{{ routeResult.algorithm || 'A* with distance and road type heuristics'
                        }}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Nodes Explored:</span>
                        <span class="stat-value">{{ routeResult.nodesExplored || '-' }}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Max Distance Limit:</span>
                        <span class="stat-value">{{ routeResult.maxDistanceLimit ?
                            routeResult.maxDistanceLimit.toFixed(0) + 'm' : '-' }}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Total Cost:</span>
                        <span class="stat-value">{{ routeResult.totalCost ? routeResult.totalCost.toFixed(2) : '-'
                        }}</span>
                    </div>
                </div>

                <div v-if="routeResult.status === 'failed'" class="pathfinding-failure">
                    <div class="failure-title">Failure Analysis</div>
                    <div class="failure-details">
                        <div class="failure-step" v-for="(step, index) in routeResult.failureSteps || []" :key="index">
                            <div class="step-number">{{ index + 1 }}</div>
                            <div class="step-description">{{ step }}</div>
                        </div>
                        <div v-if="!routeResult.failureSteps || routeResult.failureSteps.length === 0" class="no-steps">
                            No detailed failure steps available
                        </div>
                    </div>
                </div>
            </div>

            <!-- Network Analysis Section -->
            <div v-if="showNetworkDetails && networkAnalysis" class="debug-section">
                <h4>Network Analysis</h4>

                <div class="network-stats">
                    <div class="stat-row">
                        <span class="stat-label">Segments in Area:</span>
                        <span class="stat-value">{{ networkAnalysis.segmentCount || 0 }}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Connected from Source:</span>
                        <span class="stat-value">{{ networkAnalysis.connectedFromSource ?
                            networkAnalysis.connectedFromSource.length : 0 }}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Connected to Target:</span>
                        <span class="stat-value">{{ networkAnalysis.connectedToTarget ?
                            networkAnalysis.connectedToTarget.length : 0 }}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Missing Connections:</span>
                        <span class="stat-value">{{ networkAnalysis.missingConnections ?
                            networkAnalysis.missingConnections.length : 0 }}</span>
                    </div>
                </div>

                <div v-if="networkAnalysis.issues && networkAnalysis.issues.length > 0" class="network-issues">
                    <div class="issues-title">Detected Issues</div>
                    <div class="issues-list">
                        <div class="issue-item" v-for="(issue, index) in networkAnalysis.issues" :key="index">
                            <div class="issue-icon">⚠️</div>
                            <div class="issue-text">{{ issue }}</div>
                        </div>
                    </div>
                </div>

                <div v-if="networkAnalysis.suggestions && networkAnalysis.suggestions.length > 0"
                    class="network-suggestions">
                    <div class="suggestions-title">Suggestions</div>
                    <div class="suggestions-list">
                        <div class="suggestion-item" v-for="(suggestion, index) in networkAnalysis.suggestions"
                            :key="index">
                            <div class="suggestion-icon">💡</div>
                            <div class="suggestion-text">{{ suggestion }}</div>
                        </div>
                    </div>
                </div>

                <div class="connection-map">
                    <div class="connection-map-title">Connection Map</div>
                    <div class="connection-visualization">
                        <!-- This would be a visualization of the segments and connections -->
                        <svg width="100%" height="150" class="connection-svg">
                            <!-- Dynamic content would be generated here -->
                            <g v-if="networkAnalysis.visualData">
                                <!-- Source node -->
                                <circle cx="50" cy="75" r="10" class="source-node" />
                                <text x="50" y="95" text-anchor="middle" class="node-label">A</text>

                                <!-- Target node -->
                                <circle cx="350" cy="75" r="10" class="target-node" />
                                <text x="350" y="95" text-anchor="middle" class="node-label">B</text>

                                <!-- Connections -->
                                <line v-for="(conn, i) in networkAnalysis.visualData.connections || []"
                                    :key="'conn-' + i" :x1="conn.x1" :y1="conn.y1" :x2="conn.x2" :y2="conn.y2"
                                    :class="conn.type" />

                                <!-- Intermediate nodes -->
                                <g v-for="(node, i) in networkAnalysis.visualData.nodes || []" :key="'node-' + i">
                                    <circle :cx="node.x" :cy="node.y" r="6" :class="node.type" />
                                    <text :x="node.x" :y="node.y + 15" text-anchor="middle" class="node-label small">
                                        {{ node.label }}
                                    </text>
                                </g>
                            </g>
                            <text v-else x="200" y="75" text-anchor="middle" class="placeholder-text">
                                Connection visualization would be here
                            </text>
                        </svg>
                    </div>
                </div>
            </div>

            <!-- Segments List -->
            <div class="route-segments-section">
                <h4>Route Segments ({{ routeResult.route ? routeResult.route.length : 0 }})</h4>
                <div class="segments-list">
                    <div v-for="(segment, index) in routeResult.route || []" :key="index" class="segment-item"
                        @click="highlightSegmentOnMap(segment.id)">
                        <div class="segment-number">{{ index + 1 }}</div>
                        <div class="segment-details">
                            <div class="segment-id">
                                Segment {{ segment.id }} (Way {{ segment.osmWayId }})
                            </div>
                            <div class="segment-type">
                                {{ segment.highway_type }} {{ segment.name ? '- ' + segment.name : '' }}
                            </div>
                            <div class="segment-stats">
                                {{ segment.length ? segment.length.toFixed(0) + 'm' : '' }}
                                {{ segment.is_oneway ? '• One-way' : '' }}
                            </div>
                        </div>
                    </div>
                    <div v-if="!routeResult.route || routeResult.route.length === 0" class="no-segments">
                        No segment data available
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script lang="ts">
import type { LngLat, MapMouseEvent } from 'maplibre-gl';
import { PathfindingResult_ResultType, type RoadSegment, type PathfindingDebugInfo } from './model/chronotopia_pb';

export default defineComponent({
    data() {
        return {
            mode: 'placeA', // 'placeA', 'placeB', 'findingRoute', 'viewResults'
            markerA: null as LngLat | null,
            markerB: null as LngLat | null,
            markerASegment: null as RoadSegment | null,
            markerBSegment: null as RoadSegment | null,
            routeResult: null,
            networkAnalysis: null,
            isLoading: false,
            error: null as string | null,
            showNetworkDetails: false,
            mapClickListener: null,
            markersLayer: null,
            routeLayer: null,
            segmentLayer: null,
            networkLayer: null,
            bufferRadius: 500 // meter radius for network analysis
        };
    },
    computed: {
        modeDisplay() {
            switch (this.mode) {
                case 'placeA': return 'Click on map to place marker A';
                case 'placeB': return 'Click on map to place marker B';
                case 'findingRoute': return 'Finding route...';
                case 'viewResults': return 'Viewing results';
                default: return '';
            }
        },
        readyToFindRoute() {
            return this.markerA && this.markerB && this.markerASegment && this.markerBSegment && this.mode !== 'findingRoute';
        }
    },
    watch: {
        mode(newMode) {
            // Update cursor or other UI elements based on mode
            if (newMode === 'placeA' || newMode === 'placeB') {
                this.$emit('setCursor', 'crosshair');
            } else {
                this.$emit('setCursor', 'default');
            }
        }
    },
    mounted() {
        if (!import.meta.client) {
            return;
        }

        this.setupMap();
    },
    beforeUnmount() {
        this.cleanup();
    },
    methods: {
        setupMap() {
            // Set up map click listener
            this.mapClickListener = this.handleMapClick.bind(this);
            this.$emit('addMapClickListener', this.mapClickListener);

            // Create layers for markers, route, and network visualization
            this.initializeMapLayers();
        },

        initializeMapLayers() {
            // Create a source and layer for markers
            this.$emit('addMapSource', 'route-debug-markers', {
                type: 'geojson',
                data: {
                    type: 'FeatureCollection',
                    features: []
                }
            });

            this.$emit('addMapLayer', {
                id: 'route-debug-markers',
                type: 'circle',
                source: 'route-debug-markers',
                paint: {
                    'circle-radius': 8,
                    'circle-color': [
                        'match',
                        ['get', 'marker'],
                        'A', '#4caf50',
                        'B', '#f44336',
                        '#000000'
                    ],
                    'circle-stroke-width': 2,
                    'circle-stroke-color': '#ffffff'
                }
            });

            // Create a source and layer for the route
            this.$emit('addMapSource', 'route-debug-route', {
                type: 'geojson',
                data: {
                    type: 'FeatureCollection',
                    features: []
                }
            });

            this.$emit('addMapLayer', {
                id: 'route-debug-route',
                type: 'line',
                source: 'route-debug-route',
                paint: {
                    'line-color': '#2196f3',
                    'line-width': 4
                }
            });

            // Create a source and layer for segments
            this.$emit('addMapSource', 'route-debug-segments', {
                type: 'geojson',
                data: {
                    type: 'FeatureCollection',
                    features: []
                }
            });

            this.$emit('addMapLayer', {
                id: 'route-debug-segments',
                type: 'line',
                source: 'route-debug-segments',
                paint: {
                    'line-color': '#ff9800',
                    'line-width': 6,
                    'line-opacity': 0.7
                }
            });

            // Create a source and layer for network analysis
            this.$emit('addMapSource', 'route-debug-network', {
                type: 'geojson',
                data: {
                    type: 'FeatureCollection',
                    features: []
                }
            });

            this.$emit('addMapLayer', {
                id: 'route-debug-network',
                type: 'line',
                source: 'route-debug-network',
                paint: {
                    'line-color': [
                        'match',
                        ['get', 'status'],
                        'connected', '#4caf50',
                        'missing', '#f44336',
                        'a_reachable', '#2196f3',
                        'b_reachable', '#ff9800',
                        '#999999'
                    ],
                    'line-width': 3,
                    'line-opacity': 0.6,
                    // 'line-dasharray': [
                    //     'match',
                    //     ['get', 'status'],
                    //     'missing', ['literal', [2, 2]],
                    //     ['literal', [1, 0]]
                    // ]
                },
                layout: {
                    'visibility': 'none' // Hidden by default
                }
            });

        },

        handleMapClick(event: MapMouseEvent) {
            if (this.mode === 'placeA') {
                this.markerA = event.lngLat;
                this.updateMarkers();
                this.findNearestSegment(this.markerA, 'A');
                this.mode = 'placeB';
            } else if (this.mode === 'placeB') {
                this.markerB = event.lngLat;
                this.updateMarkers();
                this.findNearestSegment(this.markerB, 'B');
                this.mode = 'viewResults';
            }
        },

        updateMarkers() {
            // Update the markers GeoJSON data
            const features = [];

            if (this.markerA) {
                features.push({
                    type: 'Feature',
                    properties: {
                        marker: 'A',
                        description: 'Start point'
                    },
                    geometry: {
                        type: 'Point',
                        coordinates: [this.markerA.lng, this.markerA.lat]
                    }
                });
            }

            if (this.markerB) {
                features.push({
                    type: 'Feature',
                    properties: {
                        marker: 'B',
                        description: 'End point'
                    },
                    geometry: {
                        type: 'Point',
                        coordinates: [this.markerB.lng, this.markerB.lat]
                    }
                });
            }

            // Update source data
            this.$emit('updateMapSourceData', 'route-debug-markers', {
                type: 'FeatureCollection',
                features
            });

            // If both markers are placed, fit bounds to include both
            if (this.markerA && this.markerB) {
                this.$emit('fitBounds', [
                    [this.markerA.lng, this.markerA.lat],
                    [this.markerB.lng, this.markerB.lat]
                ], { padding: 100 });
            }
        },

        async findNearestSegment(lngLat: LngLat, markerType) {
            this.isLoading = true;
            this.error = null;

            try {
                // Make API call to find nearest segment
                const response = await this.$api.oSMNetworkAroundPoint({
                    lat: lngLat.lat,
                    lon: lngLat.lng
                });

                // Parse the GeoJSON response
                const data = JSON.parse(response.value);

                // Find the closest segment from the features
                const segments = data.features.filter(f =>
                    f.properties.type === 'network_segment'
                );

                if (segments.length === 0) {
                    throw new Error('No road segments found near this location');
                }

                // Calculate distances to point
                segments.forEach(segment => {
                    // Calculate the minimum distance from point to each segment
                    // This is a simplification - in real app you would use proper 
                    // point-to-linestring calculation
                    const coords = segment.geometry.coordinates;
                    let minDistance = Infinity;

                    for (let i = 0; i < coords.length - 1; i++) {
                        const p1 = coords[i];
                        const p2 = coords[i + 1];

                        // Simple point-to-line segment distance approximation
                        const distance = this.pointToLineSegmentDistance(
                            lngLat.lng, lngLat.lat,
                            p1[0], p1[1],
                            p2[0], p2[1]
                        );

                        minDistance = Math.min(minDistance, distance);
                    }

                    segment.properties.distance = minDistance;
                });

                // Sort by distance and get the closest
                segments.sort((a, b) => a.properties.distance - b.properties.distance);
                const closestSegment = segments[0];

                // Store the segment data
                if (markerType === 'A') {
                    this.markerASegment = {
                        id: closestSegment.properties.segment_id,
                        osm_way_id: closestSegment.properties.osm_way_id,
                        highway_type: closestSegment.properties.highway_type,
                        is_oneway: closestSegment.properties.is_oneway,
                        name: closestSegment.properties.name || 'Unnamed',
                        geometry: closestSegment.geometry,
                        connections: closestSegment.properties.connections || []
                    };
                    this.highlightSegment(this.markerASegment, 'A');
                } else {
                    this.markerBSegment = {
                        id: closestSegment.properties.segment_id,
                        osm_way_id: closestSegment.properties.osm_way_id,
                        highway_type: closestSegment.properties.highway_type,
                        is_oneway: closestSegment.properties.is_oneway,
                        name: closestSegment.properties.name || 'Unnamed',
                        geometry: closestSegment.geometry,
                        connections: closestSegment.properties.connections || []
                    };
                    this.highlightSegment(this.markerBSegment, 'B');
                }
            } catch (err) {
                this.error = `Failed to find nearest segment: ${err.message}`;
                console.error(err);
            } finally {
                this.isLoading = false;
            }
        },

        pointToLineSegmentDistance(px, py, x1, y1, x2, y2) {
            // Convert degrees to approx meters for distance calculation
            // This is a rough approximation with the Haversine formula
            const toRadians = (degrees) => degrees * Math.PI / 180;
            const EARTH_RADIUS = 6371000; // Earth radius in meters

            // Convert to radians
            const p_lat = toRadians(py);
            const p_lng = toRadians(px);
            const lat1 = toRadians(y1);
            const lng1 = toRadians(x1);
            const lat2 = toRadians(y2);
            const lng2 = toRadians(x2);

            // Function to calculate Haversine distance between two points
            const haversineDistance = (lat1, lng1, lat2, lng2) => {
                const dlat = lat2 - lat1;
                const dlng = lng2 - lng1;
                const a = Math.sin(dlat / 2) ** 2 +
                    Math.cos(lat1) * Math.cos(lat2) *
                    Math.sin(dlng / 2) ** 2;
                const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
                return EARTH_RADIUS * c;
            };

            // Distance from p to both endpoints
            const d1 = haversineDistance(p_lat, p_lng, lat1, lng1);
            const d2 = haversineDistance(p_lat, p_lng, lat2, lng2);

            // Length of the line segment
            const line_length = haversineDistance(lat1, lng1, lat2, lng2);

            // If the line segment is very short, just return distance to an endpoint
            if (line_length < 1) return Math.min(d1, d2);

            // Calculate the projection factor using dot product
            // Convert to flat coordinates for simplicity (for short distances)
            const dx = (lng2 - lng1) * Math.cos((lat1 + lat2) / 2);
            const dy = lat2 - lat1;
            const dpx = (p_lng - lng1) * Math.cos((lat1 + p_lat) / 2);
            const dpy = p_lat - lat1;

            const t = (dpx * dx + dpy * dy) / (dx * dx + dy * dy);

            // If the projection is outside the segment, return distance to endpoint
            if (t < 0) return d1;
            if (t > 1) return d2;

            // Calculate projection point
            const lng_proj = lng1 + t * (lng2 - lng1);
            const lat_proj = lat1 + t * (lat2 - lat1);

            // Return distance to projection point
            return haversineDistance(p_lat, p_lng, lat_proj, lng_proj);
        },

        highlightSegment(segment, markerType) {
            const features = [];

            features.push({
                type: 'Feature',
                properties: {
                    marker: markerType,
                    segmentId: segment.id,
                    wayId: segment.osm_way_id,
                    description: `Segment ${segment.id} (${segment.highway_type})`
                },
                geometry: segment.geometry
            });

            // Update source data
            this.$emit('updateMapSourceData', 'route-debug-segments', {
                type: 'FeatureCollection',
                features
            });
        },

        async findRoute() {
            if (!this.markerASegment || !this.markerBSegment) {
                this.error = 'Please place both markers first';
                return;
            }

            this.isLoading = true;
            this.error = null;
            this.mode = 'findingRoute';
            this.routeResult = null;
            this.networkAnalysis = null;

            try {
                // Create a debug window request for path finding
                const response = await this.$api.debugWindowPathFinding({
                    windowIndex: 0, // Dummy value, will be ignored
                    startPoint: 0,  // Dummy value, will be ignored
                    endPoint: 0,    // Dummy value, will be ignored
                    fromSegmentId: this.markerASegment.id,
                    toSegmentId: this.markerBSegment.id
                });

                // Process the response into a more friendly format
                this.processRouteResult(response);

                // Analyze network connectivity
                await this.analyzeNetworkConnectivity();

                this.mode = 'viewResults';
            } catch (err) {
                this.error = `Failed to find route: ${err.message}`;
                console.error(err);
                this.mode = 'viewResults';
            } finally {
                this.isLoading = false;
            }
        },

        processRouteResult(response: PathfindingDebugInfo) {
            // Check for successful path
            const successful = response.attemptedPairs.some(pair =>
                pair.result?.type === PathfindingResult_ResultType.SUCCESS
            );

            // Find the successful attempt
            let successfulPath: null | RoadSegment[] = null;
            let failureReason = '';
            const failureSteps = [];

            if (successful) {
                // Find the successful attempt
                const successAttempt = response.attemptedPairs.find(pair =>
                    pair.result?.type === PathfindingResult_ResultType.SUCCESS
                );

                if (successAttempt && successAttempt.result) {
                    successfulPath = successAttempt.result.path;
                }
            } else {
                // Analyze failure reasons
                const failureTypes = response.attemptedPairs.map(pair => pair.result?.type);

                if (failureTypes.every(type => type === PathfindingResult_ResultType.TOO_FAR)) {
                    failureReason = 'All paths exceed maximum allowed distance';
                } else if (failureTypes.every(type => type === PathfindingResult_ResultType.NO_CONNECTION)) {
                    failureReason = 'No connection exists between segments';
                } else if (failureTypes.includes(PathfindingResult_ResultType.NO_PATH_FOUND)) {
                    failureReason = 'Path finding algorithm failed to find a valid path';
                } else {
                    failureReason = 'Mixed failure reasons';
                }

                // Add detailed failure steps
                if (response.reason) {
                    failureSteps.push(response.reason);
                }

                // Add details from attempted pairs
                response.attemptedPairs.forEach((pair, index) => {
                    if (index < 5) { // Limit to 5 detailed steps
                        const failDetails = this.getFailureDetails(pair);
                        if (failDetails) failureSteps.push(failDetails);
                    }
                });
            }

            // Calculate direct distance between segments
            const directDistance = this.calculateDistanceBetweenSegments(
                this.markerASegment,
                this.markerBSegment
            );

            // Create a result object
            this.routeResult = {
                status: successful ? 'success' : 'failed',
                route: successfulPath || [],
                directDistance: directDistance,
                totalDistance: this.calculateRouteDistance(successfulPath),
                failureReason: failureReason,
                failureSteps: failureSteps,
                nodesExplored: response.attemptedPairs.length,
                maxDistanceLimit: response.attemptedPairs.find(p => p.result?.type === PathfindingResult_ResultType.TOO_FAR)?.result?.maxDistance,
                totalCost: response.attemptedPairs.find(p => p.result?.type === PathfindingResult_ResultType.SUCCESS)?.result?.cost
            };

            // Draw the route on the map if successful
            if (successful && successfulPath) {
                this.drawRoute(successfulPath);
            }
        },

        getFailureDetails(attempt) {
            if (!attempt || !attempt.result) return null;

            const fromSegId = attempt.fromSegment;
            const toSegId = attempt.toSegment;

            switch (attempt.result.type) {
                case PathfindingResult_ResultType.TOO_FAR:
                    return `Path from ${fromSegId} to ${toSegId} exceeds distance limit: ` +
                        `${attempt.result.actualDistance?.toFixed(0)}m > ${attempt.result.maxDistance?.toFixed(0)}m`;
                case PathfindingResult_ResultType.NO_CONNECTION:
                    return `No connection exists between segments ${fromSegId} and ${toSegId}`;
                case PathfindingResult_ResultType.NO_PATH_FOUND:
                    return `No path found: ${attempt.result.reason || 'Unknown reason'}`;
                default:
                    return null;
            }
        },

        calculateDistanceBetweenSegments(segment1, segment2) {
            // Simple distance calculation between segment centroids
            // In a real implementation, you would want a more sophisticated algorithm
            if (!segment1 || !segment2 ||
                !segment1.geometry || !segment2.geometry ||
                !segment1.geometry.coordinates || !segment2.geometry.coordinates) {
                return 0;
            }

            // Calculate centroids
            const coords1 = segment1.geometry.coordinates;
            const coords2 = segment2.geometry.coordinates;

            const centroid1 = [
                coords1.reduce((sum, p) => sum + p[0], 0) / coords1.length,
                coords1.reduce((sum, p) => sum + p[1], 0) / coords1.length
            ];

            const centroid2 = [
                coords2.reduce((sum, p) => sum + p[0], 0) / coords2.length,
                coords2.reduce((sum, p) => sum + p[1], 0) / coords2.length
            ];

            // Calculate distance with Haversine formula
            const toRadians = (degrees) => degrees * Math.PI / 180;
            const EARTH_RADIUS = 6371000; // Earth radius in meters

            const lat1 = toRadians(centroid1[1]);
            const lng1 = toRadians(centroid1[0]);
            const lat2 = toRadians(centroid2[1]);
            const lng2 = toRadians(centroid2[0]);

            const dlat = lat2 - lat1;
            const dlng = lng2 - lng1;
            const a = Math.sin(dlat / 2) ** 2 +
                Math.cos(lat1) * Math.cos(lat2) *
                Math.sin(dlng / 2) ** 2;
            const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
            return EARTH_RADIUS * c;
        },

        calculateRouteDistance(segments) {
            if (!segments || segments.length === 0) return 0;

            let totalDistance = 0;

            // Sum up segment lengths
            segments.forEach(segment => {
                if (segment.length) {
                    totalDistance += segment.length;
                } else if (segment.coordinates && segment.coordinates.length > 1) {
                    // Calculate length if not provided
                    let segmentDistance = 0;

                    for (let i = 0; i < segment.coordinates.length - 1; i++) {
                        const p1 = segment.coordinates[i];
                        const p2 = segment.coordinates[i + 1];

                        // Haversine formula
                        const toRadians = (degrees) => degrees * Math.PI / 180;
                        const EARTH_RADIUS = 6371000; // Earth radius in meters

                        const lat1 = toRadians(p1.lat);
                        const lng1 = toRadians(p1.lon);
                        const lat2 = toRadians(p2.lat);
                        const lng2 = toRadians(p2.lon);

                        const dlat = lat2 - lat1;
                        const dlng = lng2 - lng1;
                        const a = Math.sin(dlat / 2) ** 2 +
                            Math.cos(lat1) * Math.cos(lat2) *
                            Math.sin(dlng / 2) ** 2;
                        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

                        segmentDistance += EARTH_RADIUS * c;
                    }

                    totalDistance += segmentDistance;
                }
            });

            return totalDistance;
        },

        drawRoute(segments) {
            if (!segments || segments.length === 0) return;

            const features = [];

            // Create a feature for each segment
            segments.forEach((segment, index) => {
                if (!segment.coordinates || segment.coordinates.length < 2) return;

                const coords = segment.coordinates.map(coord => [coord.lon, coord.lat]);

                features.push({
                    type: 'Feature',
                    properties: {
                        index: index,
                        segmentId: segment.id,
                        wayId: segment.osmWayId,
                        name: segment.name || 'Unnamed',
                        highway_type: segment.highwayType,
                        is_oneway: segment.isOneway
                    },
                    geometry: {
                        type: 'LineString',
                        coordinates: coords
                    }
                });
            });

            // Update source data
            this.$emit('updateMapSourceData', 'route-debug-route', {
                type: 'FeatureCollection',
                features
            });

            // Fit map to route bounds
            if (features.length > 0) {
                // Calculate bounds from all coordinates
                const allCoords = features.flatMap(f => f.geometry.coordinates);

                if (allCoords.length > 0) {
                    const bounds = allCoords.reduce(
                        (bounds, coord) => [
                            [
                                Math.min(bounds[0][0], coord[0]),
                                Math.min(bounds[0][1], coord[1])
                            ],
                            [
                                Math.max(bounds[1][0], coord[0]),
                                Math.max(bounds[1][1], coord[1])
                            ]
                        ],
                        [
                            [Infinity, Infinity],
                            [-Infinity, -Infinity]
                        ]
                    );

                    this.$emit('fitBounds', bounds, { padding: 50 });
                }
            }
        },

        async analyzeNetworkConnectivity() {
            if (!this.markerASegment || !this.markerBSegment) return;

            try {
                // Make API call to analyze segment connectivity
                const response = await this.$api.analyzeSegmentConnectivity({
                    fromSegmentId: this.markerASegment.id,
                    toSegmentId: this.markerBSegment.id
                });

                // Parse the GeoJSON response
                const data = JSON.parse(response.value);

                // Analyze connectivity issues
                const issues = [];
                const suggestions = [];

                // Count segments by type
                const segments = data.features.filter(f =>
                    f.properties.type === 'network_segment'
                );

                // Count connections
                const connections = data.features.filter(f =>
                    f.properties.type === 'segment_connection'
                );

                const connectedPairs = connections.filter(f =>
                    f.properties.connected
                );

                const disconnectedPairs = connections.filter(f =>
                    !f.properties.connected
                );

                // Find segments connected to source/target
                const connectedFromSource = new Set();
                const connectedToTarget = new Set();

                connectedPairs.forEach(conn => {
                    if (conn.properties.from_segment === this.markerASegment.id) {
                        connectedFromSource.add(conn.properties.to_segment);
                    }
                    if (conn.properties.to_segment === this.markerBSegment.id) {
                        connectedToTarget.add(conn.properties.from_segment);
                    }
                });

                // Find potential missing connections
                const missingConnections = [];

                disconnectedPairs.forEach(conn => {
                    const distance = conn.properties.distance;

                    // Check if these segments are very close but not connected
                    if (distance < 10) { // 10 meters
                        missingConnections.push({
                            from: conn.properties.from_segment,
                            to: conn.properties.to_segment,
                            distance: distance
                        });

                        issues.push(
                            `Segments ${conn.properties.from_segment} and ${conn.properties.to_segment} ` +
                            `are only ${distance.toFixed(1)}m apart but not connected`
                        );

                        suggestions.push(
                            `Add explicit connection between segments ${conn.properties.from_segment} ` +
                            `and ${conn.properties.to_segment}`
                        );
                    }
                });

                // Look for shared nodes without connections
                data.features.filter(f => f.properties.type === 'no_path')
                    .forEach(feature => {
                        if (feature.properties.reason === 'Search depth limit reached') {
                            issues.push('Path finding depth limit reached - network may be too complex');
                            suggestions.push('Increase the search depth limit or optimize the graph structure');
                        }
                    });

                // Create a visualization of connectivity for the UI
                const visualData = this.createNetworkVisualization(
                    connectedFromSource,
                    connectedToTarget,
                    missingConnections
                );

                // Store network analysis results
                this.networkAnalysis = {
                    segmentCount: segments.length,
                    connectedFromSource: Array.from(connectedFromSource),
                    connectedToTarget: Array.from(connectedToTarget),
                    missingConnections: missingConnections,
                    issues: issues,
                    suggestions: suggestions,
                    visualData: visualData
                };

                // Update network layer on map
                this.updateNetworkLayer(data);

            } catch (err) {
                console.error('Failed to analyze network connectivity:', err);
            }
        },

        createNetworkVisualization(connectedFromSource, connectedToTarget, missingConnections) {
            // Create a simple visualization for the UI
            const nodes = [];
            const connections = [];

            // This is a placeholder for what would be a more sophisticated visualization
            // In a real implementation, you would use actual segment positions

            // Source and target points are fixed
            const sourceX = 50;
            const targetX = 350;
            const centerY = 75;

            // Add some intermediate nodes
            const fromCount = Math.min(connectedFromSource.size, 3);
            const toCount = Math.min(connectedToTarget.size, 3);

            // Create nodes connected to source
            for (let i = 0; i < fromCount; i++) {
                const segId = Array.from(connectedFromSource)[i];
                nodes.push({
                    x: 150,
                    y: centerY - 30 + i * 30,
                    type: 'connected',
                    label: `${segId}`
                });

                // Connection from source
                connections.push({
                    x1: sourceX,
                    y1: centerY,
                    x2: 150,
                    y2: centerY - 30 + i * 30,
                    type: 'connected'
                });
            }

            // Create nodes connected to target
            for (let i = 0; i < toCount; i++) {
                const segId = Array.from(connectedToTarget)[i];
                nodes.push({
                    x: 250,
                    y: centerY - 30 + i * 30,
                    type: 'connected',
                    label: `${segId}`
                });

                // Connection to target
                connections.push({
                    x1: 250,
                    y1: centerY - 30 + i * 30,
                    x2: targetX,
                    y2: centerY,
                    type: 'connected'
                });
            }

            // Add missing connections
            missingConnections.slice(0, 3).forEach((conn, i) => {
                connections.push({
                    x1: 150,
                    y1: centerY - 20 + i * 20,
                    x2: 250,
                    y2: centerY - 20 + i * 20,
                    type: 'missing'
                });
            });

            return {
                nodes,
                connections
            };
        },

        updateNetworkLayer(data) {
            // Update the network layer with connectivity data
            this.$emit('updateMapSourceData', 'route-debug-network', data);

            // Show the layer
            this.$emit('setLayerVisibility', 'route-debug-network',
                this.showNetworkDetails ? 'visible' : 'none');
        },

        highlightSegmentOnMap(segmentId) {
            // Find the segment in the route result
            const segment = this.routeResult.route?.find(s => s.id === segmentId);

            if (!segment) return;

            // Highlight the segment by updating the segment layer
            const features = [{
                type: 'Feature',
                properties: {
                    segmentId: segment.id,
                    wayId: segment.osm_way_id,
                    description: `Segment ${segment.id} (${segment.highway_type})`
                },
                geometry: {
                    type: 'LineString',
                    coordinates: segment.coordinates.map(c => [c.lon, c.lat])
                }
            }];

            // Update source data
            this.$emit('updateMapSourceData', 'route-debug-segments', {
                type: 'FeatureCollection',
                features
            });

            // Fly to the segment
            const coords = segment.coordinates;
            if (coords && coords.length > 0) {
                const midIdx = Math.floor(coords.length / 2);
                this.$emit('flyTo', {
                    center: [coords[midIdx].lon, coords[midIdx].lat],
                    zoom: 16,
                    speed: 1.2
                });
            }
        },

        resetDebugger() {
            // Reset all state
            this.mode = 'placeA';
            this.markerA = null;
            this.markerB = null;
            this.markerASegment = null;
            this.markerBSegment = null;
            this.routeResult = null;
            this.networkAnalysis = null;
            this.error = null;
            this.showNetworkDetails = false;

            // Clear map layers
            this.$emit('updateMapSourceData', 'route-debug-markers', {
                type: 'FeatureCollection',
                features: []
            });

            this.$emit('updateMapSourceData', 'route-debug-route', {
                type: 'FeatureCollection',
                features: []
            });

            this.$emit('updateMapSourceData', 'route-debug-segments', {
                type: 'FeatureCollection',
                features: []
            });

            this.$emit('updateMapSourceData', 'route-debug-network', {
                type: 'FeatureCollection',
                features: []
            });

            this.$emit('setLayerVisibility', 'route-debug-network', 'none');
        },

        cleanup() {
            // Remove map click listener
            if (this.mapClickListener) {
                this.$emit('removeMapClickListener', this.mapClickListener);
            }

            // Clear map layers
            this.resetDebugger();
        }
    }
});
</script>

<style scoped>
.route-debugger {
    position: absolute;
    top: 10px;
    left: 10px;
    width: 360px;
    max-height: calc(100vh - 20px);
    background-color: rgba(33, 33, 33, 0.9);
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
    color: #fff;
    z-index: 100;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.debugger-controls {
    padding: 10px;
    background-color: #333;
    border-bottom: 1px solid #444;
}

.debugger-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.debugger-header h3 {
    margin: 0;
    font-size: 16px;
    font-weight: bold;
    color: #fff;
}

.mode-indicator {
    font-size: 12px;
    color: #aaa;
}

.button-group {
    display: flex;
    gap: 8px;
}

.debug-button {
    background-color: #444;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-size: 12px;
    cursor: pointer;
    transition: background-color 0.2s;
}

.debug-button:hover {
    background-color: #555;
}

.debug-button.active {
    background-color: #2196f3;
}

.debug-button:disabled {
    background-color: #333;
    color: #777;
    cursor: not-allowed;
}

.error-message {
    margin: 10px;
    padding: 8px;
    background-color: #f44336;
    color: white;
    border-radius: 4px;
    font-size: 12px;
}

.results-panel {
    padding: 10px;
    overflow-y: auto;
    max-height: calc(100vh - 120px);
}

.results-summary {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
}

.summary-box {
    flex: 1;
    background-color: #333;
    border-radius: 4px;
    padding: 8px;
}

.summary-title {
    font-size: 11px;
    color: #aaa;
    margin-bottom: 4px;
}

.summary-content {
    font-size: 14px;
    font-weight: bold;
}

.summary-content.success {
    color: #4caf50;
}

.summary-content.failure {
    color: #f44336;
}

.failure-reason {
    margin-top: 4px;
    font-size: 11px;
    color: #ff9800;
}

.debug-section {
    background-color: #333;
    border-radius: 4px;
    padding: 10px;
    margin-bottom: 12px;
}

.debug-section h4 {
    margin: 0 0 8px;
    font-size: 14px;
    color: #2196f3;
}

.path-stats {
    margin-bottom: 12px;
}

.stat-row {
    display: flex;
    margin-bottom: 4px;
    font-size: 12px;
}

.stat-label {
    flex: 0 0 120px;
    font-weight: bold;
    color: #ddd;
}

.stat-value {
    flex: 1;
    color: #fff;
}

.pathfinding-failure {
    background-color: rgba(244, 67, 54, 0.1);
    border-radius: 4px;
    padding: 8px;
    margin-top: 8px;
}

.failure-title {
    font-size: 13px;
    font-weight: bold;
    color: #f44336;
    margin-bottom: 6px;
}

.failure-details {
    color: #ddd;
}

.failure-step {
    display: flex;
    margin-bottom: 6px;
    font-size: 12px;
}

.step-number {
    flex: 0 0 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #f44336;
    color: white;
    border-radius: 50%;
    font-size: 10px;
    font-weight: bold;
    margin-right: 8px;
}

.step-description {
    flex: 1;
}

.no-steps {
    font-style: italic;
    color: #aaa;
    font-size: 12px;
}

.network-stats {
    margin-bottom: 10px;
}

.network-issues,
.network-suggestions {
    margin-top: 12px;
}

.issues-title,
.suggestions-title {
    font-size: 13px;
    font-weight: bold;
    margin-bottom: 6px;
}

.issues-title {
    color: #f44336;
}

.suggestions-title {
    color: #4caf50;
}

.issues-list,
.suggestions-list {
    font-size: 12px;
}

.issue-item,
.suggestion-item {
    display: flex;
    margin-bottom: 6px;
}

.issue-icon,
.suggestion-icon {
    flex: 0 0 20px;
    margin-right: 6px;
}

.issue-text,
.suggestion-text {
    flex: 1;
}

.connection-map {
    margin-top: 12px;
    background-color: #222;
    border-radius: 4px;
    padding: 8px;
}

.connection-map-title {
    font-size: 13px;
    font-weight: bold;
    color: #aaa;
    margin-bottom: 8px;
}

.connection-visualization {
    height: 150px;
    border: 1px solid #444;
    border-radius: 3px;
    overflow: hidden;
}

.connection-svg {
    background-color: #1a1a1a;
}

.source-node {
    fill: #4caf50;
    stroke: #fff;
    stroke-width: 1;
}

.target-node {
    fill: #f44336;
    stroke: #fff;
    stroke-width: 1;
}

.node-label {
    fill: #fff;
    font-size: 12px;
    font-weight: bold;
}

.node-label.small {
    font-size: 9px;
    font-weight: normal;
    fill: #ddd;
}

line.connected {
    stroke: #4caf50;
    stroke-width: 2;
}

line.missing {
    stroke: #f44336;
    stroke-width: 2;
    stroke-dasharray: 4 2;
}

circle.connected {
    fill: #2196f3;
    stroke: #fff;
    stroke-width: 1;
}

.placeholder-text {
    fill: #555;
    font-size: 12px;
}

.route-segments-section {
    background-color: #333;
    border-radius: 4px;
    padding: 10px;
}

.route-segments-section h4 {
    margin: 0 0 8px;
    font-size: 14px;
    color: #2196f3;
}

.segments-list {
    max-height: 200px;
    overflow-y: auto;
}

.segment-item {
    display: flex;
    padding: 8px;
    background-color: #404040;
    border-radius: 4px;
    margin-bottom: 6px;
    cursor: pointer;
    transition: background-color 0.2s;
}

.segment-item:hover {
    background-color: #4a4a4a;
}

.segment-number {
    flex: 0 0 25px;
    height: 25px;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #2196f3;
    color: white;
    border-radius: 50%;
    font-size: 12px;
    font-weight: bold;
    margin-right: 8px;
}

.segment-details {
    flex: 1;
}

.segment-id {
    font-size: 12px;
    font-weight: bold;
    color: #fff;
    margin-bottom: 2px;
}

.segment-type {
    font-size: 11px;
    color: #ddd;
    margin-bottom: 2px;
}

.segment-stats {
    font-size: 10px;
    color: #aaa;
}

.no-segments {
    font-style: italic;
    color: #aaa;
    font-size: 12px;
    padding: 8px;
    text-align: center;
}
</style>
