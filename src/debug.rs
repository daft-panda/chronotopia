use anyhow::Result;
use geo::Point;
use serde_json::{Value, json};
use std::collections::HashMap;

use crate::osm_preprocessing::WaySegment;
use crate::route_matcher::{
    MatchedWaySegment, PathfindingDebugInfo, PathfindingResult, RouteMatchJob,
};

/// Unified debug visualization utilities
pub struct DebugVisualizer;

impl DebugVisualizer {
    /// Generate debugging GeoJSON for segments
    pub fn generate_segments_geojson(
        segments: &[WaySegment],
        properties: Option<HashMap<&str, Value>>,
    ) -> Value {
        let mut features = Vec::new();

        // Default properties if none provided
        let default_props = properties.unwrap_or_else(|| {
            let mut props = HashMap::new();
            props.insert("color", json!("#3388ff"));
            props.insert("weight", json!(4));
            props.insert("opacity", json!(0.8));
            props
        });

        // Add each segment
        for segment in segments {
            let coords: Vec<Vec<f64>> =
                segment.coordinates.iter().map(|c| vec![c.x, c.y]).collect();

            // Create properties
            let mut segment_props = serde_json::Map::new();
            segment_props.insert("segment_id".to_string(), json!(segment.id));
            segment_props.insert("osm_way_id".to_string(), json!(segment.osm_way_id));
            segment_props.insert(
                "highway_type".to_string(),
                json!(segment.highway_type.clone()),
            );
            segment_props.insert("is_oneway".to_string(), json!(segment.is_oneway));

            if let Some(name) = &segment.name {
                segment_props.insert("name".to_string(), json!(name));
            }

            // Add custom properties
            for (key, value) in &default_props {
                segment_props.insert(key.to_string(), value.clone());
            }

            // Description
            segment_props.insert(
                "description".to_string(),
                json!(format!(
                    "Segment ID: {} (OSM: {}), Type: {}, Name: {}",
                    segment.id,
                    segment.osm_way_id,
                    segment.highway_type,
                    segment.name.as_deref().unwrap_or("Unnamed")
                )),
            );

            features.push(json!({
                "type": "Feature",
                "properties": segment_props,
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                }
            }));
        }

        // Return GeoJSON
        json!({
            "type": "FeatureCollection",
            "features": features
        })
    }

    /// Generate debugging GeoJSON for matched segments
    pub fn generate_matched_segments_geojson(
        segments: &[MatchedWaySegment],
        properties: Option<HashMap<&str, Value>>,
    ) -> Value {
        let mut features = Vec::new();

        // Default properties if none provided
        let default_props = properties.unwrap_or_else(|| {
            let mut props = HashMap::new();
            props.insert("color", json!("#ff0000"));
            props.insert("weight", json!(4));
            props.insert("opacity", json!(0.8));
            props
        });

        // First create the combined route feature
        let mut combined_coords = Vec::new();

        // Collect all coordinates
        for matched_segment in segments {
            let coords = matched_segment.coordinates();
            for coord in &coords {
                combined_coords.push(vec![coord.x, coord.y]);
            }
        }

        // Create main route feature
        let mut route_props = serde_json::Map::new();
        route_props.insert("type".to_string(), json!("matched_route"));
        route_props.insert("description".to_string(), json!("Matched Route"));

        // Add custom properties
        for (key, value) in &default_props {
            route_props.insert(key.to_string(), json!(value));
        }

        let main_route = json!({
            "type": "Feature",
            "properties": route_props,
            "geometry": {
                "type": "LineString",
                "coordinates": combined_coords
            }
        });

        features.push(main_route);

        // Add segment features
        for (i, matched_segment) in segments.iter().enumerate() {
            let coords = matched_segment.coordinates();
            let coords_json: Vec<Vec<f64>> = coords.iter().map(|c| vec![c.x, c.y]).collect();

            // Create properties
            let mut segment_props = serde_json::Map::new();
            segment_props.insert("type".to_string(), json!("route_segment"));
            segment_props.insert("index".to_string(), json!(i));
            segment_props.insert("segment_id".to_string(), json!(matched_segment.segment.id));
            segment_props.insert(
                "osm_way_id".to_string(),
                json!(matched_segment.segment.osm_way_id),
            );
            segment_props.insert(
                "highway_type".to_string(),
                json!(matched_segment.segment.highway_type.clone()),
            );

            // Show entry/exit nodes if present
            if let Some(entry) = matched_segment.entry_node {
                segment_props.insert("entry_node".to_string(), json!(entry));
            }

            if let Some(exit) = matched_segment.exit_node {
                segment_props.insert("exit_node".to_string(), json!(exit));
            }

            // Add custom properties with slightly different color
            let mut seg_props = default_props.clone();

            // Make alternate segments slightly different colors
            if i % 2 == 1 {
                if let Some(color) = seg_props.get_mut("color") {
                    // Use different color for odd segments
                    *color = json!("#cc3300");
                }
            }

            for (key, value) in &seg_props {
                segment_props.insert(key.to_string(), value.clone());
            }

            // Description
            segment_props.insert(
                "description".to_string(),
                json!(format!(
                    "Segment #{}: ID {} (OSM: {}), Type: {}",
                    i,
                    matched_segment.segment.id,
                    matched_segment.segment.osm_way_id,
                    matched_segment.segment.highway_type
                )),
            );

            features.push(json!({
                "type": "Feature",
                "properties": segment_props,
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords_json
                }
            }));

            // Add entry/exit nodes
            if let Some(entry_idx) = matched_segment.entry_node {
                if entry_idx < coords.len() {
                    features.push(json!({
                        "type": "Feature",
                        "properties": {
                            "type": "entry_node",
                            "segment_id": matched_segment.segment.id,
                            "index": entry_idx,
                            "color": "#00ff00",
                            "radius": 5
                        },
                        "geometry": {
                            "type": "Point",
                            "coordinates": [coords[entry_idx].x, coords[entry_idx].y]
                        }
                    }));
                }
            }

            if let Some(exit_idx) = matched_segment.exit_node {
                if exit_idx < coords.len() {
                    features.push(json!({
                        "type": "Feature",
                        "properties": {
                            "type": "exit_node",
                            "segment_id": matched_segment.segment.id,
                            "index": exit_idx,
                            "color": "#0000ff",
                            "radius": 5
                        },
                        "geometry": {
                            "type": "Point",
                            "coordinates": [coords[exit_idx].x, coords[exit_idx].y]
                        }
                    }));
                }
            }
        }

        // Add junction points between segments
        if segments.len() > 1 {
            for i in 0..segments.len() - 1 {
                if let (Some(end_node), Some(start_node)) =
                    (segments[i].end_node(), segments[i + 1].start_node())
                {
                    if end_node == start_node {
                        // This is a junction point
                        let coords = segments[i].coordinates();
                        if let Some(coord) = coords.last() {
                            features.push(json!({
                                "type": "Feature",
                                "properties": {
                                    "type": "junction",
                                    "node_id": end_node,
                                    "color": "#ffff00",
                                    "radius": 6
                                },
                                "geometry": {
                                    "type": "Point",
                                    "coordinates": [coord.x, coord.y]
                                }
                            }));
                        }
                    }
                }
            }
        }

        // Return GeoJSON
        json!({
            "type": "FeatureCollection",
            "features": features
        })
    }

    /// Generate GeoJSON for point candidates
    pub fn generate_point_candidates_geojson(
        job: &RouteMatchJob,
        point_idx: usize,
    ) -> Result<Value> {
        let candidates = &job.all_candidates.borrow()[point_idx];
        let point = job.gps_points[point_idx];
        let mut features = Vec::new();

        // Add the GPS point as a feature
        features.push(json!({
            "type": "Feature",
            "properties": {
                "type": "gps_point",
                "index": point_idx,
                "description": format!("GPS Point #{}", point_idx)
            },
            "geometry": {
                "type": "Point",
                "coordinates": [point.x(), point.y()]
            }
        }));

        // Add each candidate segment
        for (i, candidate) in candidates.iter().enumerate() {
            let segment = &candidate.segment;
            let coords: Vec<Vec<f64>> =
                segment.coordinates.iter().map(|c| vec![c.x, c.y]).collect();

            // Segment feature
            features.push(json!({
                "type": "Feature",
                "properties": {
                    "type": "candidate_segment",
                    "segment_id": segment.id,
                    "osm_way_id": segment.osm_way_id,
                    "rank": i,
                    "cost": candidate.cost,
                    "distance": candidate.distance,
                    "highway_type": segment.highway_type,
                    "color": if i == 0 { "#00ff00" } else { "#3388ff" },
                    "opacity": 1.0 - (i as f64 * 0.15).min(0.8),
                    "weight": 4 - i.min(3),
                    "description": format!("Segment ID: {} (OSM: {}), Rank: {}, Cost: {:.2}, Distance: {:.2}m", 
                                        segment.id, segment.osm_way_id, i, candidate.cost, candidate.distance)
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                }
            }));

            // Projection point feature
            features.push(json!({
                "type": "Feature",
                "properties": {
                    "type": "projection",
                    "segment_id": segment.id,
                    "osm_way_id": segment.osm_way_id,
                    "color": if i == 0 { "#00ff00" } else { "#3388ff" },
                    "radius": 5,
                    "description": format!("Projection to segment {} (OSM: {})", segment.id, segment.osm_way_id)
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [candidate.projection.x(), candidate.projection.y()]
                }
            }));

            // Add line from point to projection
            features.push(json!({
                "type": "Feature",
                "properties": {
                    "type": "projection_line",
                    "segment_id": segment.id,
                    "distance": candidate.distance,
                    "color": if i == 0 { "#00ff00" } else { "#3388ff" },
                    "opacity": 0.5,
                    "weight": 2,
                    "dashArray": "5,5"
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [point.x(), point.y()],
                        [candidate.projection.x(), candidate.projection.y()]
                    ]
                }
            }));
        }

        // Create GeoJSON
        let geojson = json!({
            "type": "FeatureCollection",
            "features": features
        });

        Ok(geojson)
    }

    /// Generate GeoJSON for segment connectivity analysis
    pub fn generate_connectivity_geojson(
        start_segment: &WaySegment,
        end_segment: &WaySegment,
        path: Option<&[WaySegment]>,
        connection_info: Option<&str>,
    ) -> Value {
        let mut features = Vec::new();

        // Add start segment
        let start_coords: Vec<Vec<f64>> = start_segment
            .coordinates
            .iter()
            .map(|c| vec![c.x, c.y])
            .collect();

        features.push(json!({
            "type": "Feature",
            "properties": {
                "type": "start_segment",
                "segment_id": start_segment.id,
                "osm_way_id": start_segment.osm_way_id,
                "highway_type": start_segment.highway_type,
                "color": "#00ff00",
                "weight": 5,
                "opacity": 0.8,
                "description": format!("Start Segment: ID {} (OSM: {})", start_segment.id, start_segment.osm_way_id)
            },
            "geometry": {
                "type": "LineString",
                "coordinates": start_coords
            }
        }));

        // Add end segment
        let end_coords: Vec<Vec<f64>> = end_segment
            .coordinates
            .iter()
            .map(|c| vec![c.x, c.y])
            .collect();

        features.push(json!({
            "type": "Feature",
            "properties": {
                "type": "end_segment",
                "segment_id": end_segment.id,
                "osm_way_id": end_segment.osm_way_id,
                "highway_type": end_segment.highway_type,
                "color": "#ff0000",
                "weight": 5,
                "opacity": 0.8,
                "description": format!("End Segment: ID {} (OSM: {})", end_segment.id, end_segment.osm_way_id)
            },
            "geometry": {
                "type": "LineString",
                "coordinates": end_coords
            }
        }));

        // Add path if exists
        if let Some(path_segments) = path {
            // Draw path segments
            for (i, segment) in path_segments.iter().enumerate() {
                // Skip start and end segments in path
                if segment.id == start_segment.id || segment.id == end_segment.id {
                    continue;
                }

                let coords: Vec<Vec<f64>> =
                    segment.coordinates.iter().map(|c| vec![c.x, c.y]).collect();

                features.push(json!({
                    "type": "Feature",
                    "properties": {
                        "type": "path_segment",
                        "segment_id": segment.id,
                        "osm_way_id": segment.osm_way_id,
                        "highway_type": segment.highway_type,
                        "index": i,
                        "color": "#0088ff",
                        "weight": 4,
                        "opacity": 0.7,
                        "description": format!("Path Segment #{}: ID {} (OSM: {})", i, segment.id, segment.osm_way_id)
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords
                    }
                }));
            }

            // Add path line connecting centroids
            let path_centroids: Vec<Vec<f64>> = path_segments
                .iter()
                .map(|s| {
                    let c = s.centroid();
                    vec![c.x(), c.y()]
                })
                .collect();

            features.push(json!({
                "type": "Feature",
                "properties": {
                    "type": "path_line",
                    "segments": path_segments.len(),
                    "color": "#00ffff",
                    "weight": 3,
                    "opacity": 0.6,
                    "dashArray": "5,5",
                    "description": format!("Path: {} segments", path_segments.len())
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": path_centroids
                }
            }));
        } else {
            // No path found - draw line between segment centroids
            let start_center = start_segment.centroid();
            let end_center = end_segment.centroid();

            features.push(json!({
                "type": "Feature",
                "properties": {
                    "type": "no_path",
                    "color": "#ff00ff",
                    "weight": 3,
                    "opacity": 0.6,
                    "dashArray": "10,10",
                    "description": connection_info.unwrap_or("No path found")
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [start_center.x(), start_center.y()],
                        [end_center.x(), end_center.y()]
                    ]
                }
            }));
        }

        // Add marker points for segment endpoints
        if let Some(first) = start_segment.coordinates.first() {
            features.push(json!({
                "type": "Feature",
                "properties": {
                    "type": "segment_start",
                    "segment_id": start_segment.id,
                    "color": "#00ff00",
                    "radius": 5
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [first.x, first.y]
                }
            }));
        }

        if let Some(last) = start_segment.coordinates.last() {
            features.push(json!({
                "type": "Feature",
                "properties": {
                    "type": "segment_end",
                    "segment_id": start_segment.id,
                    "color": "#88ff00",
                    "radius": 5
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [last.x, last.y]
                }
            }));
        }

        if let Some(first) = end_segment.coordinates.first() {
            features.push(json!({
                "type": "Feature",
                "properties": {
                    "type": "segment_start",
                    "segment_id": end_segment.id,
                    "color": "#ff8800",
                    "radius": 5
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [first.x, first.y]
                }
            }));
        }

        if let Some(last) = end_segment.coordinates.last() {
            features.push(json!({
                "type": "Feature",
                "properties": {
                    "type": "segment_end",
                    "segment_id": end_segment.id,
                    "color": "#ff0000",
                    "radius": 5
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [last.x, last.y]
                }
            }));
        }

        // Find shared nodes
        let shared_nodes: Vec<u64> = start_segment
            .nodes
            .iter()
            .filter(|n| end_segment.nodes.contains(n))
            .cloned()
            .collect();

        for node_id in shared_nodes {
            // Find node position in each segment
            if let Some(idx1) = start_segment.nodes.iter().position(|&n| n == node_id) {
                if idx1 < start_segment.coordinates.len() {
                    let coord = start_segment.coordinates[idx1];

                    features.push(json!({
                        "type": "Feature",
                        "properties": {
                            "type": "shared_node",
                            "node_id": node_id,
                            "color": "#ffff00",
                            "radius": 8,
                            "description": format!("Shared Node: {}", node_id)
                        },
                        "geometry": {
                            "type": "Point",
                            "coordinates": [coord.x, coord.y]
                        }
                    }));
                }
            }
        }

        json!({
            "type": "FeatureCollection",
            "features": features
        })
    }

    /// Generate detailed debug info for pathfinding
    pub fn generate_pathfinding_debug_geojson(debug_info: &PathfindingDebugInfo) -> Value {
        let mut features = Vec::new();

        // Add start candidates
        for (i, candidate) in debug_info.start_candidates.iter().enumerate() {
            let segment = &candidate.segment;
            let coords: Vec<Vec<f64>> =
                segment.coordinates.iter().map(|c| vec![c.x, c.y]).collect();

            features.push(json!({
                "type": "Feature",
                "properties": {
                    "type": "start_candidate",
                    "segment_id": segment.id,
                    "osm_way_id": segment.osm_way_id,
                    "rank": i,
                    "distance": candidate.distance,
                    "color": "#00ff00",
                    "weight": 4 - i.min(3),
                    "opacity": 0.8 - (i as f64 * 0.1).min(0.6),
                    "description": format!("Start Candidate #{}: ID {} (OSM: {}), Distance: {:.2}m",
                        i, segment.id, segment.osm_way_id, candidate.distance)
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                }
            }));

            // Add projection point
            features.push(json!({
                "type": "Feature",
                "properties": {
                    "type": "start_projection",
                    "segment_id": segment.id,
                    "color": "#00ff00",
                    "radius": 5
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [candidate.projection.x(), candidate.projection.y()]
                }
            }));
        }

        // Add end candidates
        for (i, candidate) in debug_info.end_candidates.iter().enumerate() {
            let segment = &candidate.segment;
            let coords: Vec<Vec<f64>> =
                segment.coordinates.iter().map(|c| vec![c.x, c.y]).collect();

            features.push(json!({
                "type": "Feature",
                "properties": {
                    "type": "end_candidate",
                    "segment_id": segment.id,
                    "osm_way_id": segment.osm_way_id,
                    "rank": i,
                    "distance": candidate.distance,
                    "color": "#ff0000",
                    "weight": 4 - i.min(3),
                    "opacity": 0.8 - (i as f64 * 0.1).min(0.6),
                    "description": format!("End Candidate #{}: ID {} (OSM: {}), Distance: {:.2}m",
                        i, segment.id, segment.osm_way_id, candidate.distance)
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                }
            }));

            // Add projection point
            features.push(json!({
                "type": "Feature",
                "properties": {
                    "type": "end_projection",
                    "segment_id": segment.id,
                    "color": "#ff0000",
                    "radius": 5
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [candidate.projection.x(), candidate.projection.y()]
                }
            }));
        }

        // Add pathfinding attempts
        for (i, attempt) in debug_info.attempted_pairs.iter().enumerate() {
            // Skip if we've already added too many
            if i >= 10 {
                break;
            }

            // Pathfinding attempt visualization depends on result
            match &attempt.result {
                PathfindingResult::Success(path, cost) => {
                    // For successful paths, draw the route
                    let path_coords: Vec<Vec<f64>> = path
                        .iter()
                        .map(|segment| {
                            let center = segment.centroid();
                            vec![center.x(), center.y()]
                        })
                        .collect();

                    features.push(json!({
                        "type": "Feature",
                        "properties": {
                            "type": "successful_path",
                            "from_segment": attempt.from_segment,
                            "to_segment": attempt.to_segment,
                            "segments": path.len(),
                            "cost": cost,
                            "color": "#00ffff",
                            "weight": 3,
                            "opacity": 0.7,
                            "description": format!("Successful Path #{}: {} segments, cost: {:.2}",
                                i, path.len(), cost)
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": path_coords
                        }
                    }));

                    // Also add individual segments for the first successful path
                    if i == 0 {
                        for (j, matched) in path.iter().enumerate() {
                            let seg_coords: Vec<Vec<f64>> = matched
                                .segment
                                .coordinates
                                .iter()
                                .map(|c| vec![c.x, c.y])
                                .collect();

                            features.push(json!({
                                "type": "Feature",
                                "properties": {
                                    "type": "path_segment",
                                    "segment_id": matched.segment.id,
                                    "osm_way_id": matched.segment.osm_way_id,
                                    "index": j,
                                    "color": "#00cccc",
                                    "weight": 3,
                                    "opacity": 0.8,
                                    "description": format!("Path Segment #{}: ID {} (OSM: {})",
                                        j, matched.segment.id, matched.segment.osm_way_id)
                                },
                                "geometry": {
                                    "type": "LineString",
                                    "coordinates": seg_coords
                                }
                            }));
                        }
                    }
                }
                PathfindingResult::TooFar(max_dist, actual_dist) => {
                    // For too far results, draw a dashed line
                    let start_segment_center = Point::new(0.0, 0.0); // Will be filled later
                    let end_segment_center = Point::new(0.0, 0.0); // Will be filled later

                    features.push(json!({
                        "type": "Feature",
                        "properties": {
                            "type": "too_far_path",
                            "from_segment": attempt.from_segment,
                            "to_segment": attempt.to_segment,
                            "max_distance": max_dist,
                            "actual_distance": actual_dist,
                            "color": "#ff00ff",
                            "weight": 2,
                            "opacity": 0.5,
                            "dashArray": "5,10",
                            "description": format!("Too Far Path #{}: {:.2}m exceeds limit of {:.2}m",
                                i, actual_dist, max_dist)
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [0.0, 0.0], // Placeholders
                                [0.0, 0.0]  // Will be filled later
                            ]
                        }
                    }));
                }
                PathfindingResult::NoConnection => {
                    // For no connection, draw a red X
                    features.push(json!({
                        "type": "Feature",
                        "properties": {
                            "type": "no_connection",
                            "from_segment": attempt.from_segment,
                            "to_segment": attempt.to_segment,
                            "color": "#ff0000",
                            "weight": 2,
                            "opacity": 0.5,
                            "dashArray": "2,8",
                            "description": format!("No Connection #{}: Segments not connected",
                                i)
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [0.0, 0.0], // Placeholders
                                [0.0, 0.0]  // Will be filled later
                            ]
                        }
                    }));
                }
                PathfindingResult::NoPathFound(reason) => {
                    // For general path finding failures
                    features.push(json!({
                        "type": "Feature",
                        "properties": {
                            "type": "no_path_found",
                            "from_segment": attempt.from_segment,
                            "to_segment": attempt.to_segment,
                            "reason": reason,
                            "color": "#ff8800",
                            "weight": 2,
                            "opacity": 0.5,
                            "dashArray": "3,6",
                            "description": format!("Path Finding Failed #{}: {}",
                                i, reason)
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [0.0, 0.0], // Placeholders
                                [0.0, 0.0]  // Will be filled later
                            ]
                        }
                    }));
                }
            }
        }

        // Add constraint points
        for (point_idx, segment_id) in &debug_info.constraints {
            features.push(json!({
                "type": "Feature",
                "properties": {
                    "type": "constraint_point",
                    "point_idx": point_idx,
                    "segment_id": segment_id,
                    "color": "#ffff00",
                    "radius": 8,
                    "description": format!("Constraint: Point {} must use segment {}",
                        point_idx, segment_id)
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [0.0, 0.0] // Placeholder
                }
            }));
        }

        json!({
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "debug_reason": debug_info.reason
            }
        })
    }
}
