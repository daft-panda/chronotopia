use anyhow::{Result, anyhow, bail};
use chrono::{DateTime, Utc};
use geo::{Closest, ClosestPoint, Haversine, LineString, algorithm::Distance};
use geo_types::Point;
use log::{debug, info};
use ordered_float::OrderedFloat;
use petgraph::prelude::UnGraphMap;
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    path::Path,
};

use crate::{
    mapmatcher::TileConfig,
    osm_preprocessing::{OsmProcessor, RoadSegment},
    tile_loader::TileLoader,
};

/// Enhanced configuration for route-based map matching
#[derive(Debug, Clone)]
pub struct RouteMatcherConfig {
    /// Path to OpenStreetMap PBF file
    pub osm_pbf_path: String,
    /// Directory for storing preprocessed tiles
    pub tile_cache_dir: String,
    /// Tile size in degrees
    pub tile_config: TileConfig,
    /// Maximum number of tiles to keep in memory
    pub max_cached_tiles: usize,
    /// Maximum distance for point-to-edge matching (meters)
    pub max_matching_distance: f64,
    /// Maximum number of candidates per GPS point
    pub max_candidates_per_point: usize,
    /// Maximum number of route hypotheses to maintain
    pub max_route_hypotheses: usize,
    /// Maximum number of tiles to load per depth level
    pub max_tiles_per_depth: usize,
    /// Route complexity penalty factor (per segment change)
    pub route_complexity_penalty: f64,
    /// Weight factor for route length in scoring
    pub route_length_weight: f64,
    /// Heading consistency importance factor
    pub heading_consistency_weight: f64,
    /// Weight for main roads preference
    pub main_road_preference_weight: f64,
}

impl Default for RouteMatcherConfig {
    fn default() -> Self {
        Self {
            osm_pbf_path: String::new(),
            tile_cache_dir: String::new(),
            tile_config: TileConfig::default(),
            max_cached_tiles: 100,
            max_matching_distance: 100.0,
            max_candidates_per_point: 5,
            max_route_hypotheses: 20,
            max_tiles_per_depth: 50,
            route_complexity_penalty: 10.0,
            route_length_weight: 0.3,
            heading_consistency_weight: 0.3,
            main_road_preference_weight: 0.4,
        }
    }
}

/// Represents a candidate road segment for a GPS point
#[derive(Clone, Debug)]
pub struct RouteCandidate {
    pub segment: RoadSegment,
    pub point_on_edge: Point<f64>,
    pub distance: f64,
    pub heading_diff: f64,
    pub road_class_score: f64,
}

/// A weighted point in the trajectory
#[derive(Clone, Debug)]
pub struct WeightedPoint {
    pub point: Point<f64>,
    pub timestamp: DateTime<Utc>,
    pub weight: f64, // Higher weight = more important (key shape points)
    pub candidates: Vec<RouteCandidate>,
}

/// A route hypothesis
#[derive(Clone, Debug)]
pub struct RouteHypothesis {
    pub segments: Vec<RoadSegment>,
    pub segment_ids: Vec<u64>,
    pub total_length: f64,
    pub avg_distance_to_points: f64,
    pub avg_heading_consistency: f64,
    pub road_class_score: f64,
    pub complexity_score: f64,
    pub total_score: f64,
}

impl RouteHypothesis {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            segment_ids: Vec::new(),
            total_length: 0.0,
            avg_distance_to_points: 0.0,
            avg_heading_consistency: 0.0,
            road_class_score: 0.0,
            complexity_score: 0.0,
            total_score: 0.0,
        }
    }
}

impl Eq for RouteHypothesis {}

impl PartialEq for RouteHypothesis {
    fn eq(&self, other: &Self) -> bool {
        self.total_score.eq(&other.total_score)
    }
}

impl Ord for RouteHypothesis {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher score is better for routes
        other
            .total_score
            .partial_cmp(&self.total_score)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for RouteHypothesis {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Route-based map matcher implementation
pub struct RouteMatcher {
    config: RouteMatcherConfig,
    tile_loader: TileLoader,
}

impl RouteMatcher {
    /// Create a new route matcher with given configuration
    pub fn new(config: RouteMatcherConfig) -> Result<Self> {
        // Validate configuration
        if !Path::new(&config.osm_pbf_path).exists() {
            bail!("OSM PBF file not found");
        }

        let tile_loader = TileLoader::new(
            config.tile_cache_dir.clone(),
            config.max_cached_tiles,
            config.tile_config.clone(),
        );

        Ok(Self {
            config,
            tile_loader,
        })
    }

    /// Preprocess OSM data into tiles
    pub fn preprocess(&self) -> Result<()> {
        info!("Starting OSM data preprocessing");
        std::fs::create_dir_all(&self.config.tile_cache_dir)?;
        let processor = OsmProcessor::new(self.config.tile_config.clone());
        processor.process_pbf(&self.config.osm_pbf_path, &self.config.tile_cache_dir)?;
        info!("OSM preprocessing finished");
        Ok(())
    }

    /// Main map matching entry point with hierarchical refinement
    pub fn match_trace(
        &mut self,
        gps_points: &[Point<f64>],
        timestamps: &[DateTime<Utc>],
        debug_way_ids: Option<&[u64]>,
    ) -> Result<Vec<RoadSegment>> {
        if gps_points.len() < 2 {
            return Ok(Vec::new());
        }

        info!(
            "Starting hierarchical route-based map matching for {} points",
            gps_points.len()
        );
        let start_time = std::time::Instant::now();

        // Step 1: Load tiles covering the entire route area
        let trace_bbox = self.calculate_trace_bbox(gps_points);
        let buffer = self.config.max_matching_distance * 2.0 / 111_000.0; // Convert meters to approx degrees

        info!("Loading tiles for route area");
        let loaded_tiles = self.tile_loader.load_tile_range(
            trace_bbox,
            buffer,
            self.config.max_tiles_per_depth,
        )?;

        // Step 2: Build road network graph for path finding
        info!("Building road network graph");
        let (graph, segment_map) = self.build_road_network(&loaded_tiles)?;

        // Step 3: Create high-level route using start and end points
        info!("Building high-level route from start to end points");
        let high_level_route = self.build_high_level_route(
            &[
                gps_points.first().unwrap().clone(),
                gps_points.last().unwrap().clone(),
            ],
            &[
                timestamps.first().unwrap().clone(),
                timestamps.last().unwrap().clone(),
            ],
            &loaded_tiles,
            &graph,
            &segment_map,
        )?;

        if high_level_route.is_empty() {
            return Err(anyhow!(
                "Failed to build high-level route between endpoints"
            ));
        }

        info!(
            "High-level route built successfully with {} segments",
            high_level_route.len()
        );

        // Step 4: Process intermediate points and refine the route
        let mut refined_route = high_level_route;
        let intermediate_points = &gps_points[1..gps_points.len() - 1];
        let intermediate_timestamps = &timestamps[1..timestamps.len() - 1];

        if !intermediate_points.is_empty() {
            info!(
                "Refining route with {} intermediate points",
                intermediate_points.len()
            );
            refined_route = self.refine_route_with_intermediate_points(
                refined_route,
                intermediate_points,
                intermediate_timestamps,
                &loaded_tiles,
                &graph,
                &segment_map,
            )?;
        }

        info!(
            "Map matching completed in {:.2?} with {} segments",
            start_time.elapsed(),
            refined_route.len()
        );

        // Debug information about specified way IDs
        if let Some(way_ids) = debug_way_ids {
            self.debug_way_ids(&refined_route, way_ids, &loaded_tiles)?;
        }

        Ok(refined_route)
    }

    /// Build high-level route using just start and end points
    fn build_high_level_route(
        &mut self,
        endpoints: &[Point<f64>],
        timestamps: &[DateTime<Utc>],
        loaded_tiles: &HashSet<String>,
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
    ) -> Result<Vec<RoadSegment>> {
        // Generate candidates for start and end points
        let weighted_endpoints = endpoints
            .iter()
            .zip(timestamps)
            .map(|(p, t)| WeightedPoint {
                point: *p,
                timestamp: *t,
                weight: 3.0, // High weight for endpoints
                candidates: Vec::new(),
            })
            .collect::<Vec<_>>();

        // Generate candidates for endpoints
        let main_corridors = self.identify_route_corridors(&weighted_endpoints, loaded_tiles)?;
        let endpoints_with_candidates =
            self.generate_point_candidates(&weighted_endpoints, loaded_tiles, &main_corridors)?;

        // Generate route hypotheses
        let route_hypotheses =
            self.generate_route_hypotheses(&endpoints_with_candidates, graph, segment_map)?;

        if route_hypotheses.is_empty() {
            return Err(anyhow!(
                "No valid route hypotheses found for high-level route"
            ));
        }

        // Select best route
        let best_route = route_hypotheses[0].clone();

        Ok(best_route.segments)
    }

    /// Refine route with intermediate points one by one
    fn refine_route_with_intermediate_points(
        &mut self,
        initial_route: Vec<RoadSegment>,
        intermediate_points: &[Point<f64>],
        intermediate_timestamps: &[DateTime<Utc>],
        loaded_tiles: &HashSet<String>,
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
    ) -> Result<Vec<RoadSegment>> {
        let mut current_route = initial_route;

        // Process each intermediate point in sequence
        for (i, (point, timestamp)) in intermediate_points
            .iter()
            .zip(intermediate_timestamps.iter())
            .enumerate()
        {
            debug!(
                "Processing intermediate point {}/{}",
                i + 1,
                intermediate_points.len()
            );

            // Try to match the point to the current route
            let (matched_to_route, matched_segment_idx, dist_to_route) =
                self.match_point_to_route(*point, &current_route);

            // If point is close enough to the current route, continue
            if matched_to_route && dist_to_route <= self.config.max_matching_distance * 0.7 {
                debug!(
                    "Point {}/{} matched to existing route (distance: {:.2}m)",
                    i + 1,
                    intermediate_points.len(),
                    dist_to_route
                );
                continue;
            }

            // Point needs a route adjustment
            debug!(
                "Point {}/{} requires route adjustment (distance to route: {:.2}m)",
                i + 1,
                intermediate_points.len(),
                dist_to_route
            );

            // Find candidate segments for this point
            let weighted_point = WeightedPoint {
                point: *point,
                timestamp: *timestamp,
                weight: 2.0, // Medium weight for refinement point
                candidates: Vec::new(),
            };

            let corridors = Vec::new(); // No need for corridors at refinement stage
            let mut point_with_candidates =
                self.generate_point_candidates(&[weighted_point], loaded_tiles, &corridors)?;

            if point_with_candidates[0].candidates.is_empty() {
                debug!(
                    "No candidates found for point {}/{}, keeping current route",
                    i + 1,
                    intermediate_points.len()
                );
                continue;
            }

            // Sort candidates by a combination of distance and directional consistency
            self.sort_candidates_by_route_context(
                &mut point_with_candidates[0].candidates,
                &current_route,
            );

            // Try each candidate until we find one that doesn't create a loop
            let mut adjusted_route = None;

            // If there was a matched segment on the route (even if far), use it as reference
            let route_split_idx = if matched_to_route {
                matched_segment_idx
            } else {
                // Find closest segment in route as split point
                self.find_closest_segment_idx(*point, &current_route)
            };

            // Try each candidate
            for candidate in &point_with_candidates[0].candidates {
                // Check if this segment is already in the route (would create a loop)
                if current_route
                    .iter()
                    .any(|seg| seg.id == candidate.segment.id)
                {
                    debug!(
                        "Skipping candidate {} as it would create a loop",
                        candidate.segment.id
                    );
                    continue;
                }

                // Try to adjust the route with this candidate
                match self.adjust_route_with_candidate(
                    &current_route,
                    route_split_idx,
                    &candidate.segment,
                    graph,
                    segment_map,
                ) {
                    Ok(new_route) => {
                        // Check for loops in the new route
                        if self.has_loops(&new_route) {
                            debug!(
                                "Adjusted route with candidate {} has loops, trying next candidate",
                                candidate.segment.id
                            );
                            continue;
                        }

                        adjusted_route = Some(new_route);
                        debug!(
                            "Successfully adjusted route with candidate {} (distance: {:.2}m)",
                            candidate.segment.id, candidate.distance
                        );
                        break;
                    }
                    Err(e) => {
                        debug!(
                            "Failed to adjust route with candidate {}: {}",
                            candidate.segment.id, e
                        );
                        continue;
                    }
                }
            }

            // Update current route if adjustment was successful
            if let Some(new_route) = adjusted_route {
                current_route = new_route;
                debug!(
                    "Route updated for point {}/{}, new route has {} segments",
                    i + 1,
                    intermediate_points.len(),
                    current_route.len()
                );
            } else {
                debug!(
                    "No valid route adjustment found for point {}/{}, keeping current route",
                    i + 1,
                    intermediate_points.len()
                );
            }
        }

        Ok(current_route)
    }

    /// Adjust route to include a specific candidate segment
    fn adjust_route_with_candidate(
        &self,
        current_route: &[RoadSegment],
        split_idx: usize,
        candidate_segment: &RoadSegment,
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
    ) -> Result<Vec<RoadSegment>> {
        // We'll split the route at split_idx and try to connect:
        // 1. From the split point to the candidate segment
        // 2. From the candidate segment to the rest of the route

        // Ensure split_idx is valid
        if split_idx >= current_route.len() {
            return Err(anyhow!("Invalid split index"));
        }

        let first_part = &current_route[0..=split_idx];
        let last_part = &current_route[split_idx..];

        // Connect first part to candidate
        let first_to_candidate = match self.find_path(
            first_part.last().unwrap().id,
            candidate_segment.id,
            graph,
            segment_map,
        ) {
            Ok((_, path)) => path,
            Err(e) => return Err(anyhow!("Failed to connect first part to candidate: {}", e)),
        };

        // Connect candidate to last part
        let candidate_to_last = match self.find_path(
            candidate_segment.id,
            last_part.first().unwrap().id,
            graph,
            segment_map,
        ) {
            Ok((_, path)) => path,
            Err(e) => return Err(anyhow!("Failed to connect candidate to last part: {}", e)),
        };

        // Combine the parts to form new route
        let mut new_route = Vec::with_capacity(
            first_part.len() + first_to_candidate.len() + candidate_to_last.len() - 2,
        );

        // Add first part of original route (up to split point)
        new_route.extend_from_slice(first_part);

        // Add path from split point to candidate (skip first segment which is already in new_route)
        for segment in first_to_candidate.iter().skip(1) {
            new_route.push(segment.clone());
        }

        // Add path from candidate to rest of route (skip first segment which is last of previous path)
        // Also skip the last segment which will be duplicated with the rest of the original route
        for segment in candidate_to_last
            .iter()
            .skip(1)
            .take(candidate_to_last.len() - 2)
        {
            new_route.push(segment.clone());
        }

        // Add remaining part of original route (from split_idx onwards)
        new_route.extend_from_slice(&last_part[1..]);

        Ok(new_route)
    }

    /// Check if a route contains duplicate segment IDs (loops)
    fn has_loops(&self, route: &[RoadSegment]) -> bool {
        let mut seen_ids = HashSet::new();

        for segment in route {
            if !seen_ids.insert(segment.id) {
                return true; // Duplicate found
            }
        }

        false
    }

    /// Debug function to track why specific way IDs were not chosen
    fn debug_way_ids(
        &mut self,
        final_route: &[RoadSegment],
        way_ids: &[u64],
        loaded_tiles: &HashSet<String>,
    ) -> Result<()> {
        // First check if any of the specified way IDs are in the final route
        let route_segment_ids: HashSet<u64> = final_route.iter().map(|seg| seg.id).collect();

        for &way_id in way_ids {
            if route_segment_ids.contains(&way_id) {
                info!("Debug: Way ID {} is included in the final route", way_id);
                continue;
            }

            // Way ID not in final route, investigate why
            info!(
                "Debug: Way ID {} is NOT included in the final route",
                way_id
            );

            // Check if the way ID exists in loaded tiles
            let mut way_exists = false;
            let mut way_segment: Option<RoadSegment> = None;

            for tile_id in loaded_tiles {
                let tile = self.tile_loader.load_tile(tile_id)?;
                if let Some(segment) = tile.road_segments.iter().find(|s| s.id == way_id) {
                    way_exists = true;
                    way_segment = Some(segment.clone());
                    info!("Debug: Way ID {} exists in tile {}", way_id, tile_id);
                    break;
                }
            }

            if !way_exists {
                info!("Debug: Way ID {} was not found in any loaded tile", way_id);
                continue;
            }

            // Analyze closest GPS points to this segment
            if let Some(segment) = way_segment {
                let mut closest_distance = f64::MAX;
                let mut closest_point_idx = 0;

                for (i, point) in final_route.iter().enumerate() {
                    // Use centroid of segment as reference point
                    let segment_point = segment.centroid();
                    let route_segment_point = point.centroid();

                    let distance = Haversine.distance(segment_point, route_segment_point);
                    if distance < closest_distance {
                        closest_distance = distance;
                        closest_point_idx = i;
                    }
                }

                info!(
                    "Debug: Way ID {} is closest to segment #{} in route (distance: {:.2}m)",
                    way_id, closest_point_idx, closest_distance
                );

                // Check connections to nearby segments
                let nearby_segment = &final_route[closest_point_idx];
                let is_connected = segment.connections.contains(&nearby_segment.id)
                    || nearby_segment.connections.contains(&segment.id);

                if is_connected {
                    info!(
                        "Debug: Way ID {} is connected to segment {} in the route",
                        way_id, nearby_segment.id
                    );
                } else {
                    info!(
                        "Debug: Way ID {} is NOT connected to segment {} in the route",
                        way_id, nearby_segment.id
                    );
                }

                // Analyze highway type and other attributes
                info!(
                    "Debug: Way ID {} is type '{}' (route segment is type '{}')",
                    way_id, segment.highway_type, nearby_segment.highway_type
                );

                // Check if it would create a loop
                let mut test_route = final_route.to_vec();
                test_route.insert(closest_point_idx + 1, segment.clone());

                if self.has_loops(&test_route) {
                    info!(
                        "Debug: Adding Way ID {} would create a loop in the route",
                        way_id
                    );
                }
            }
        }

        Ok(())
    }

    /// Try to match a point to an existing route
    /// Returns (matched_successfully, segment_index, distance)
    fn match_point_to_route(&self, point: Point<f64>, route: &[RoadSegment]) -> (bool, usize, f64) {
        let mut min_distance = f64::MAX;
        let mut matched_idx = 0;
        let mut matched = false;

        for (i, segment) in route.iter().enumerate() {
            let projection = self.project_point_to_segment(point, segment);
            let distance = Haversine.distance(point, projection);

            if distance < min_distance {
                min_distance = distance;
                matched_idx = i;
                matched = true;
            }
        }

        (matched, matched_idx, min_distance)
    }

    /// Find the index of the closest segment in a route to a point
    fn find_closest_segment_idx(&self, point: Point<f64>, route: &[RoadSegment]) -> usize {
        let mut min_distance = f64::MAX;
        let mut closest_idx = 0;

        for (i, segment) in route.iter().enumerate() {
            let projection = self.project_point_to_segment(point, segment);
            let distance = Haversine.distance(point, projection);

            if distance < min_distance {
                min_distance = distance;
                closest_idx = i;
            }
        }

        closest_idx
    }

    /// Sort candidates by context of the existing route (direction, distance, etc.)
    fn sort_candidates_by_route_context(
        &self,
        candidates: &mut Vec<RouteCandidate>,
        route: &[RoadSegment],
    ) {
        // If route is empty, just sort by distance
        if route.is_empty() {
            candidates.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(Ordering::Equal)
            });
            return;
        }

        // Get average direction of the route
        let avg_direction = self.calculate_route_direction(route);

        // Update heading_diff for each candidate based on route context
        for candidate in candidates.iter_mut() {
            let segment_direction = self.calculate_segment_direction(&candidate.segment);
            candidate.heading_diff = angle_difference(avg_direction, segment_direction);
        }

        // Sort by a combination of distance and heading consistency
        candidates.sort_by(|a, b| {
            let a_score = a.distance + a.heading_diff * 2.0;
            let b_score = b.distance + b.heading_diff * 2.0;
            a_score.partial_cmp(&b_score).unwrap_or(Ordering::Equal)
        });
    }

    /// Calculate average direction of a route
    fn calculate_route_direction(&self, route: &[RoadSegment]) -> f64 {
        if route.is_empty() {
            return 0.0;
        }

        let mut total_direction = 0.0;
        let mut total_weight = 0.0;

        for segment in route {
            let dir = self.calculate_segment_direction(segment);
            let weight = segment.length(); // Longer segments have more weight
            total_direction += dir * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            total_direction / total_weight
        } else {
            0.0
        }
    }

    /// Calculate average direction of a segment
    fn calculate_segment_direction(&self, segment: &RoadSegment) -> f64 {
        if segment.coordinates.len() < 2 {
            return 0.0;
        }

        let first = segment.coordinates.first().unwrap();
        let last = segment.coordinates.last().unwrap();

        let start_point = Point::new(first.x, first.y);
        let end_point = Point::new(last.x, last.y);

        calculate_heading(start_point, end_point)
    }

    /// Identify main route corridors
    fn identify_route_corridors(
        &mut self,
        weighted_points: &[WeightedPoint],
        loaded_tiles: &HashSet<String>,
    ) -> Result<Vec<RoadSegment>> {
        // Find high-importance roads that could form the backbone of the route
        let mut corridors = Vec::new();
        let mut visited_road_types = HashSet::new();

        // Look only at key shape points (with higher weight)
        let key_points: Vec<_> = weighted_points.iter().filter(|p| p.weight > 1.0).collect();

        if key_points.is_empty() {
            return Ok(Vec::new());
        }

        // Look for major roads near key points
        for point in key_points {
            for tile_id in loaded_tiles {
                let segments = {
                    let tile = self.tile_loader.load_tile(tile_id)?;
                    tile.road_segments.clone() // Clone to avoid borrow checker issues
                };

                for segment in &segments {
                    // Only consider major roads for corridors
                    let road_score = calculate_road_class_score(&segment.highway_type);
                    if road_score > 3.0 {
                        continue; // Skip minor roads
                    }

                    // Calculate distance to point
                    let distance = self.calculate_distance_to_segment(point.point, segment);

                    // If within reasonable distance and not already processed this road type
                    if distance < self.config.max_matching_distance * 1.5
                        && !visited_road_types.contains(&segment.highway_type)
                    {
                        corridors.push(segment.clone());
                        visited_road_types.insert(segment.highway_type.clone());
                    }
                }
            }
        }

        // Sort corridors by road importance
        corridors.sort_by(|a, b| {
            let score_a = calculate_road_class_score(&a.highway_type);
            let score_b = calculate_road_class_score(&b.highway_type);
            score_a.partial_cmp(&score_b).unwrap_or(Ordering::Equal)
        });

        // Limit to most important corridors
        if corridors.len() > 10 {
            corridors.truncate(10);
        }

        Ok(corridors)
    }

    /// Generate candidates for each weighted point, with preference for corridors
    fn generate_point_candidates(
        &mut self,
        weighted_points: &[WeightedPoint],
        loaded_tiles: &HashSet<String>,
        corridors: &[RoadSegment],
    ) -> Result<Vec<WeightedPoint>> {
        let mut result = Vec::with_capacity(weighted_points.len());

        for (i, point) in weighted_points.iter().enumerate() {
            if i % 50 == 0 {
                debug!(
                    "Generating candidates for point {}/{}",
                    i + 1,
                    weighted_points.len()
                );
            }

            let mut candidates = Vec::new();

            // First check corridors (give priority to major roads that form corridors)
            for corridor in corridors {
                let projection = self.project_point_to_segment(point.point, corridor);
                let distance = Haversine.distance(point.point, projection);

                if distance <= self.config.max_matching_distance {
                    // Calculate heading similarity if applicable
                    let prev_point = if i > 0 {
                        Some(weighted_points[i - 1].point)
                    } else {
                        None
                    };
                    let heading_diff = if let Some(pp) = prev_point {
                        let user_heading = calculate_heading(pp, point.point);
                        let seg_heading = corridor.direction_at_point(projection);
                        angle_difference(user_heading, seg_heading)
                    } else {
                        0.0
                    };

                    candidates.push(RouteCandidate {
                        segment: corridor.clone(),
                        point_on_edge: projection,
                        distance,
                        heading_diff,
                        road_class_score: calculate_road_class_score(&corridor.highway_type),
                    });
                }
            }

            // Then look for other segments if we don't have enough candidates
            if candidates.len() < self.config.max_candidates_per_point {
                for tile_id in loaded_tiles {
                    let segments = {
                        let tile = self.tile_loader.load_tile(tile_id)?;
                        tile.road_segments.clone() // Clone to avoid borrow checker issues
                    };

                    for segment in &segments {
                        // Skip segments we already have from corridors
                        if candidates.iter().any(|c| c.segment.id == segment.id) {
                            continue;
                        }

                        let projection = self.project_point_to_segment(point.point, segment);
                        let distance = Haversine.distance(point.point, projection);

                        if distance <= self.config.max_matching_distance {
                            // Calculate heading similarity if applicable
                            let prev_point = if i > 0 {
                                Some(weighted_points[i - 1].point)
                            } else {
                                None
                            };
                            let heading_diff = if let Some(pp) = prev_point {
                                let user_heading = calculate_heading(pp, point.point);
                                let seg_heading = segment.direction_at_point(projection);
                                angle_difference(user_heading, seg_heading)
                            } else {
                                0.0
                            };

                            candidates.push(RouteCandidate {
                                segment: segment.clone(),
                                point_on_edge: projection,
                                distance,
                                heading_diff,
                                road_class_score: calculate_road_class_score(&segment.highway_type),
                            });
                        }
                    }

                    // Break if we have enough candidates
                    if candidates.len() >= self.config.max_candidates_per_point {
                        break;
                    }
                }
            }

            // Sort candidates by a combination of distance and road class
            candidates.sort_by(|a, b| {
                let score_a = a.distance + a.road_class_score * 10.0;
                let score_b = b.distance + b.road_class_score * 10.0;
                score_a.partial_cmp(&score_b).unwrap_or(Ordering::Equal)
            });

            // Limit to max candidates
            if candidates.len() > self.config.max_candidates_per_point {
                candidates.truncate(self.config.max_candidates_per_point);
            }

            if candidates.is_empty() {
                debug!("No candidates found for point {}", i);
            }

            // Create new weighted point with candidates
            let mut point_with_candidates = point.clone();
            point_with_candidates.candidates = candidates;
            result.push(point_with_candidates);
        }

        Ok(result)
    }

    /// Build road network graph for path finding
    fn build_road_network(
        &mut self,
        loaded_tiles: &HashSet<String>,
    ) -> Result<(UnGraphMap<u64, f64>, HashMap<u64, RoadSegment>)> {
        let mut graph = UnGraphMap::new();
        let mut segment_map = HashMap::new();

        // Collect all segments
        for tile_id in loaded_tiles {
            let tile = self.tile_loader.load_tile(tile_id)?;

            for segment in &tile.road_segments {
                segment_map.insert(segment.id, segment.clone());
            }
        }

        // Build graph connections
        for segment in segment_map.values() {
            for &conn_id in &segment.connections {
                if !graph.contains_edge(segment.id, conn_id) {
                    // Calculate transition cost
                    let cost = if let Some(conn_segment) = segment_map.get(&conn_id) {
                        calculate_transition_cost(segment, conn_segment)
                    } else {
                        // Default cost
                        1.0
                    };

                    graph.add_edge(segment.id, conn_id, cost);
                }
            }
        }

        debug!(
            "Built road network graph with {} nodes and {} edges",
            graph.node_count(),
            graph.edge_count()
        );

        Ok((graph, segment_map))
    }

    /// Generate route hypotheses by finding paths between candidate segments
    fn generate_route_hypotheses(
        &self,
        points_with_candidates: &[WeightedPoint],
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
    ) -> Result<Vec<RouteHypothesis>> {
        // First, find points that actually have candidates
        let valid_points: Vec<_> = points_with_candidates
            .iter()
            .filter(|p| !p.candidates.is_empty())
            .collect();

        if valid_points.len() < 2 {
            return Err(anyhow!("Not enough points with valid candidates"));
        }

        // We'll start with a set of initial hypotheses using the best candidates from key points
        let key_points: Vec<_> = valid_points
            .iter()
            .filter(|p| p.weight > 1.0)
            .cloned()
            .collect();

        let start_points = if !key_points.is_empty() {
            key_points
        } else {
            valid_points.clone()
        };

        // Create initial empty hypothesis
        let initial_hypothesis = RouteHypothesis::new();
        let mut hypotheses = vec![initial_hypothesis];

        // For each key point, expand hypotheses
        for (i, point) in start_points.iter().enumerate() {
            debug!("Processing key point {}/{}", i + 1, start_points.len());

            let mut new_hypotheses = Vec::new();

            // For each existing hypothesis, try to extend with each candidate
            for hypothesis in &hypotheses {
                for candidate in &point.candidates {
                    // If hypothesis is empty, just add the segment
                    if hypothesis.segments.is_empty() {
                        let mut new_hyp = hypothesis.clone();
                        new_hyp.segments.push(candidate.segment.clone());
                        new_hyp.segment_ids.push(candidate.segment.id);
                        new_hyp.total_length = candidate.segment.length();
                        new_hyp.road_class_score = candidate.road_class_score;
                        new_hyp.avg_distance_to_points = candidate.distance;
                        new_hyp.avg_heading_consistency = 100.0 - candidate.heading_diff;

                        // Initial score
                        new_hyp.total_score = self.score_route_hypothesis(&new_hyp);
                        new_hypotheses.push(new_hyp);
                    } else {
                        // We need to find a path from the last segment to this candidate
                        let last_segment_id = *hypothesis.segment_ids.last().unwrap();

                        // Skip if it's the same segment (already there)
                        if last_segment_id == candidate.segment.id {
                            continue;
                        }

                        // Find path between segments
                        match self.find_path(
                            last_segment_id,
                            candidate.segment.id,
                            graph,
                            segment_map,
                        ) {
                            Ok((_path_cost, path_segments)) => {
                                // Only add if we found a valid path
                                if !path_segments.is_empty() {
                                    let mut new_hyp = hypothesis.clone();

                                    // First segment is already in the hypothesis, so skip it
                                    let new_segments = &path_segments[1..];

                                    // Add path segments
                                    for seg in new_segments {
                                        new_hyp.segments.push(seg.clone());
                                        new_hyp.segment_ids.push(seg.id);
                                        new_hyp.total_length += seg.length();
                                    }

                                    // Update complexity score - penalize each segment change
                                    new_hyp.complexity_score += path_segments.len() as f64
                                        * self.config.route_complexity_penalty;

                                    // Update other scores
                                    new_hyp.avg_distance_to_points =
                                        (new_hyp.avg_distance_to_points * (i as f64)
                                            + candidate.distance)
                                            / (i as f64 + 1.0);
                                    new_hyp.avg_heading_consistency =
                                        (new_hyp.avg_heading_consistency * (i as f64)
                                            + (100.0 - candidate.heading_diff))
                                            / (i as f64 + 1.0);
                                    new_hyp.road_class_score = (new_hyp.road_class_score
                                        * hypothesis.segments.len() as f64
                                        + candidate.road_class_score * new_segments.len() as f64)
                                        / (hypothesis.segments.len() + new_segments.len()) as f64;

                                    // Calculate total score
                                    new_hyp.total_score = self.score_route_hypothesis(&new_hyp);
                                    new_hypotheses.push(new_hyp);
                                }
                            }
                            Err(e) => {
                                debug!("Failed to find path: {}", e);
                                // We don't add this hypothesis
                            }
                        }
                    }
                }
            }

            // Sort and limit hypotheses
            new_hypotheses.sort_by(|a, b| {
                b.total_score
                    .partial_cmp(&a.total_score)
                    .unwrap_or(Ordering::Equal)
            });
            if new_hypotheses.len() > self.config.max_route_hypotheses {
                new_hypotheses.truncate(self.config.max_route_hypotheses);
            }

            // Replace hypotheses with new expanded ones
            hypotheses = new_hypotheses;

            if hypotheses.is_empty() {
                return Err(anyhow!("No valid hypotheses after processing point {}", i));
            }
        }

        // After processing key points, fill in details for the intermediate points if needed

        // Sort final hypotheses by score
        hypotheses.sort_by(|a, b| {
            b.total_score
                .partial_cmp(&a.total_score)
                .unwrap_or(Ordering::Equal)
        });

        // Log info about top hypotheses
        for (i, hyp) in hypotheses.iter().take(3).enumerate() {
            debug!(
                "Hypothesis #{}: score={:.2}, segments={}, length={:.1}m, avg_dist={:.1}m",
                i + 1,
                hyp.total_score,
                hyp.segments.len(),
                hyp.total_length,
                hyp.avg_distance_to_points
            );
        }

        Ok(hypotheses)
    }

    /// Score a route hypothesis based on multiple factors
    fn score_route_hypothesis(&self, hypothesis: &RouteHypothesis) -> f64 {
        // Normalize length (shorter is better, but not the only factor)
        // We use negative values because lower is better for length
        let length_score = -hypothesis.total_length / 1000.0; // Convert to km

        // Heading consistency (higher is better)
        let heading_score = hypothesis.avg_heading_consistency;

        // Road class score (lower is better)
        let road_score = -hypothesis.road_class_score;

        // Complexity penalty (lower is better)
        let complexity_score = -hypothesis.complexity_score;

        // Combine scores with appropriate weights
        let total_score = self.config.route_length_weight * length_score
            + self.config.heading_consistency_weight * heading_score
            + self.config.main_road_preference_weight * road_score
            + complexity_score; // Always apply complexity penalty

        total_score
    }

    /// Find path between two segments using A* search
    fn find_path(
        &self,
        from: u64,
        to: u64,
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
    ) -> Result<(f64, Vec<RoadSegment>)> {
        if from == to {
            if let Some(segment) = segment_map.get(&from) {
                return Ok((0.0, vec![segment.clone()]));
            }
        }

        // A* search
        let mut open_set = BinaryHeap::new();
        let mut closed_set = HashSet::new();
        let mut g_scores = HashMap::new();
        let mut came_from = HashMap::new();

        // Get destination coordinates for heuristic
        let to_segment = segment_map
            .get(&to)
            .ok_or_else(|| anyhow!("Destination segment not found"))?;
        let goal_point = to_segment.centroid();

        // Initialize
        g_scores.insert(from, 0.0);
        open_set.push(OrderedFloat(0.0), from);

        while let Some((_, current)) = open_set.pop() {
            if current == to {
                // Reconstruct path
                let mut path = Vec::new();
                let cost = *g_scores.get(&to).unwrap_or(&0.0);
                let mut current_node = to;

                while current_node != from {
                    if let Some(segment) = segment_map.get(&current_node) {
                        path.push(segment.clone());
                    }

                    if let Some(&prev) = came_from.get(&current_node) {
                        current_node = prev;
                    } else {
                        return Err(anyhow!("Path reconstruction failed"));
                    }
                }

                // Add the starting segment
                if let Some(segment) = segment_map.get(&from) {
                    path.push(segment.clone());
                }

                // Reverse to get correct order
                path.reverse();

                return Ok((cost, path));
            }

            if closed_set.contains(&current) {
                continue;
            }
            closed_set.insert(current);

            // Get current segment
            let current_segment = match segment_map.get(&current) {
                Some(seg) => seg,
                None => continue, // Skip if segment not found
            };

            // Process neighbors
            for &neighbor in &current_segment.connections {
                if closed_set.contains(&neighbor) {
                    continue;
                }

                // Check if it exists in the graph
                if !graph.contains_edge(current, neighbor) {
                    continue;
                }

                let neighbor_segment = match segment_map.get(&neighbor) {
                    Some(seg) => seg,
                    None => continue, // Skip if segment not found
                };

                // Calculate cost (edge cost + turn penalty)
                let edge_cost = *graph.edge_weight(current, neighbor).unwrap_or(&1.0);
                let new_g_score = g_scores[&current] + edge_cost;

                // Only consider if better path
                if !g_scores.contains_key(&neighbor) || new_g_score < g_scores[&neighbor] {
                    // Update path
                    came_from.insert(neighbor, current);
                    g_scores.insert(neighbor, new_g_score);

                    // Calculate heuristic
                    let neighbor_point = neighbor_segment.centroid();
                    let h_score = Haversine.distance(neighbor_point, goal_point) / 1000.0; // km
                    let f_score = new_g_score + h_score;

                    // Add to open set
                    open_set.push(OrderedFloat(-f_score), neighbor); // Negative because we want min heap
                }
            }
        }

        // No path found
        Err(anyhow!(
            "No path found between segments {} and {}",
            from,
            to
        ))
    }

    /// Calculate bounding box for the trace
    fn calculate_trace_bbox(&self, points: &[Point<f64>]) -> geo::Rect<f64> {
        if points.is_empty() {
            return geo::Rect::new(
                geo::Coord {
                    x: -180.0,
                    y: -90.0,
                },
                geo::Coord { x: 180.0, y: 90.0 },
            );
        }

        let mut min_x = points[0].x();
        let mut max_x = points[0].x();
        let mut min_y = points[0].y();
        let mut max_y = points[0].y();

        for point in points {
            min_x = min_x.min(point.x());
            max_x = max_x.max(point.x());
            min_y = min_y.min(point.y());
            max_y = max_y.max(point.y());
        }

        // Add buffer
        let buffer = self.config.max_matching_distance * 2.0 / 111_000.0;

        geo::Rect::new(
            geo::Coord {
                x: min_x - buffer,
                y: min_y - buffer,
            },
            geo::Coord {
                x: max_x + buffer,
                y: max_y + buffer,
            },
        )
    }

    /// Project a point to a road segment and return projected point
    fn project_point_to_segment(&self, point: Point<f64>, segment: &RoadSegment) -> Point<f64> {
        let line = LineString::from(segment.coordinates.clone());
        match line.closest_point(&point) {
            Closest::SinglePoint(projected) => projected,
            _ => point, // Fallback, should not happen with valid segments
        }
    }

    /// Calculate distance from point to segment
    fn calculate_distance_to_segment(&self, point: Point<f64>, segment: &RoadSegment) -> f64 {
        let projection = self.project_point_to_segment(point, segment);
        Haversine.distance(point, projection)
    }
}

/// Custom binary heap implementation with key-value pairs
struct BinaryHeap<K: Ord, V> {
    heap: Vec<(K, V)>,
}

impl<K: Ord, V> BinaryHeap<K, V> {
    fn new() -> Self {
        Self { heap: Vec::new() }
    }

    fn push(&mut self, key: K, value: V) {
        self.heap.push((key, value));
        let len = self.heap.len();
        if len > 1 {
            self.heap.sort_by(|a, b| a.0.cmp(&b.0));
        }
    }

    fn pop(&mut self) -> Option<(K, V)> {
        self.heap.pop()
    }

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

/// Calculate perpendicular distance from point to line segment
fn perpendicular_distance(point: Point<f64>, start: Point<f64>, end: Point<f64>) -> f64 {
    let a = end.y() - start.y();
    let b = start.x() - end.x();
    let c = end.x() * start.y() - start.x() * end.y();

    let numerator = (a * point.x() + b * point.y() + c).abs();
    let denominator = (a * a + b * b).sqrt();

    if denominator.abs() < 1e-10 {
        // Line is actually a point
        Haversine.distance(point, start)
    } else {
        numerator / denominator
    }
}

/// Calculate transition cost between segments
fn calculate_transition_cost(from_seg: &RoadSegment, to_seg: &RoadSegment) -> f64 {
    // Calculate geographic distance between segments
    let from_end = from_seg.coordinates.last().unwrap();
    let to_start = to_seg.coordinates.first().unwrap();

    let from_point = Point::new(from_end.x, from_end.y);
    let to_point = Point::new(to_start.x, to_start.y);

    let distance = Haversine.distance(from_point, to_point);

    // Calculate heading change penalty with more severe penalty for U-turns
    let from_heading = from_seg.direction_at_point(from_point);
    let to_heading = to_seg.direction_at_point(to_point);
    let heading_diff = angle_difference(from_heading, to_heading);

    // Increased penalties for sharp turns
    let turn_penalty = if heading_diff > 150.0 {
        10.0 // U-turn penalty is very severe
    } else if heading_diff > 90.0 {
        5.0 // Major turn penalty is higher
    } else if heading_diff > 45.0 {
        2.0 // Minor turn penalty is increased
    } else {
        1.0 // No penalty for slight turns
    };

    // Consider road type compatibility with stronger preference for staying on same road type
    let type_penalty = if from_seg.highway_type == to_seg.highway_type {
        0.8 // Bonus for staying on same road type
    } else {
        // Stronger penalty for changing road types
        1.5
    };

    // Consider road name - strongly prefer staying on the same named road
    let name_factor = if from_seg.name.is_some() && from_seg.name == to_seg.name {
        0.5 // Significant bonus for staying on same named road
    } else {
        1.0 // No bonus for different named roads
    };

    // Combine factors
    (distance / 100.0) * turn_penalty * type_penalty * name_factor
}

pub fn calculate_heading(from: Point<f64>, to: Point<f64>) -> f64 {
    let dx = to.x() - from.x();
    let dy = to.y() - from.y();
    dy.atan2(dx).to_degrees()
}

fn angle_difference(a: f64, b: f64) -> f64 {
    let diff = ((a - b) % 360.0).abs();
    diff.min(360.0 - diff)
}

/// Calculate score for road class - lower values are better (major roads)
pub fn calculate_road_class_score(highway_type: &str) -> f64 {
    match highway_type {
        "motorway" => 0.0,
        "motorway_link" => 0.5,
        "trunk" => 1.0,
        "trunk_link" => 1.5,
        "primary" => 2.0,
        "primary_link" => 2.5,
        "secondary" => 3.0,
        "secondary_link" => 3.5,
        "tertiary" => 4.0,
        "tertiary_link" => 4.5,
        "unclassified" => 6.0,
        "residential" => 7.0,
        "service" => 8.0,
        "track" => 9.0,
        "path" => 10.0,
        _ => 5.0, // Default for unknown types
    }
}
