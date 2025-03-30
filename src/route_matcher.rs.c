use anyhow::{Result, anyhow, bail};
use chrono::{DateTime, Duration, Utc};
use geo::{Closest, ClosestPoint, Haversine, LineString, algorithm::Distance};
use geo_types::Point;
use log::{debug, info, warn};
use ordered_float::{Float, OrderedFloat};
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

    /// Main map matching entry point with sliding window approach
    pub fn match_trace(
        &mut self,
        gps_points: &[Point<f64>],
        timestamps: &[DateTime<Utc>],
        debug_way_ids: Option<&[u64]>,
    ) -> Result<Vec<RoadSegment>> {
        if gps_points.len() < 2 || gps_points.len() != timestamps.len() {
            return Err(anyhow!("Invalid GPS points or timestamps"));
        }

        info!(
            "Starting sliding window route matching for {} points",
            gps_points.len()
        );
        let start_time = std::time::Instant::now();

        // Step 1: Load tiles covering the entire route area with extra buffer for GPS inaccuracy
        let trace_bbox = self.calculate_trace_bbox(gps_points);
        let buffer = (self.config.max_matching_distance * 2.0) / 111_000.0; // Convert meters to approx degrees

        info!("Loading tiles for route area");
        let loaded_tiles = self.tile_loader.load_tile_range(
            trace_bbox,
            buffer,
            self.config.max_tiles_per_depth,
        )?;

        // Step 2: Build road network graph for path finding
        info!("Building road network graph");
        let (graph, segment_map) = self.build_road_network(&loaded_tiles)?;

        // Step 3: For each GPS point, find all candidate segments within accuracy range
        info!("Finding candidate segments for all GPS points");
        let all_candidates = self.find_all_candidate_segments(gps_points, loaded_tiles.clone())?;

        // Step 4: Build route using sliding window approach
        info!("Building route using sliding window approach");
        let route = self.build_route_with_sliding_window(
            gps_points,
            timestamps,
            &all_candidates,
            &graph,
            &segment_map,
            &loaded_tiles,
        )?;

        info!(
            "Map matching completed in {:.2?} with {} segments",
            start_time.elapsed(),
            route.len()
        );

        // Debug information about specified way IDs
        if let Some(way_ids) = debug_way_ids {
            self.debug_way_ids(&route, way_ids, &loaded_tiles)?;
        }

        Ok(route)
    }

    /// Find all candidate segments within accuracy range for each GPS point
    fn find_all_candidate_segments(
        &mut self,
        gps_points: &[Point<f64>],
        loaded_tiles: HashSet<String>,
    ) -> Result<Vec<Vec<SegmentCandidate>>> {
        const GPS_ACCURACY: f64 = 75.0; // 75m accuracy
        let max_distance = GPS_ACCURACY * 1.5; // Allow up to 1.5x the accuracy

        let mut all_candidates = Vec::with_capacity(gps_points.len());

        for (i, &point) in gps_points.iter().enumerate() {
            let mut candidates = Vec::new();

            // For each tile, find segments within range
            for tile_id in &loaded_tiles {
                // Load tile and clone segments to avoid borrow checker issues
                let segments = {
                    let tile = self.tile_loader.load_tile(tile_id)?;
                    tile.road_segments.clone()
                };

                for segment in segments {
                    // Project point to segment and calculate distance
                    let projection = self.project_point_to_segment(point, &segment);
                    let distance = Haversine.distance(point, projection);

                    // If within accuracy range, add as candidate
                    if distance <= max_distance {
                        // Basic score based on distance
                        let score = distance / GPS_ACCURACY;

                        candidates.push(SegmentCandidate {
                            segment,
                            distance,
                            projection,
                            score,
                        });
                    }
                }
            }

            // If no candidates found, try with increased range
            if candidates.is_empty() {
                debug!(
                    "No candidates found for point {} within normal range, increasing search distance",
                    i
                );

                let increased_max = max_distance * 1.5; // Further increase search distance

                for tile_id in &loaded_tiles {
                    let segments = {
                        let tile = self.tile_loader.load_tile(tile_id)?;
                        tile.road_segments.clone()
                    };

                    for segment in segments {
                        let projection = self.project_point_to_segment(point, &segment);
                        let distance = Haversine.distance(point, projection);

                        if distance <= increased_max {
                            // Higher score due to being outside normal range
                            let score = distance / GPS_ACCURACY * 1.5;

                            candidates.push(SegmentCandidate {
                                segment,
                                distance,
                                projection,
                                score,
                            });
                        }
                    }
                }
            }

            // Sort candidates by score
            candidates.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal));

            // Limit number of candidates per point to improve performance
            let max_candidates = 10;
            if candidates.len() > max_candidates {
                candidates.truncate(max_candidates);
            }

            debug!("Found {} candidates for GPS point {}", candidates.len(), i);
            all_candidates.push(candidates);
        }

        Ok(all_candidates)
    }

    /// Build route using sliding window approach
    fn build_route_with_sliding_window(
        &mut self,
        gps_points: &[Point<f64>],
        timestamps: &[DateTime<Utc>],
        all_candidates: &[Vec<SegmentCandidate>],
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
        loaded_tiles: &HashSet<String>,
    ) -> Result<Vec<RoadSegment>> {
        // Define window size - adapt based on GPS density
        let window_size = if gps_points.len() < 10 {
            // For very few points, use larger window
            gps_points.len().min(7)
        } else if gps_points.len() < 20 {
            // For moderate number of points
            5
        } else {
            // For many points, use smaller window
            4
        };

        debug!("Using sliding window of size {}", window_size);

        // Special case: if we have fewer points than window size, just do one window
        if gps_points.len() <= window_size {
            return self.match_single_window(
                gps_points,
                timestamps,
                all_candidates,
                graph,
                segment_map,
                0,
                gps_points.len() - 1,
            );
        }

        // Process data in overlapping windows
        let mut complete_route = Vec::new();
        let mut last_end_segment: Option<RoadSegment> = None;

        // Step through the data with overlapping windows
        let step_size = window_size / 2; // 50% overlap between windows
        let mut window_start = 0;

        while window_start < gps_points.len() {
            let window_end = (window_start + window_size - 1).min(gps_points.len() - 1);

            debug!(
                "Processing window from point {} to {} (of {})",
                window_start,
                window_end,
                gps_points.len() - 1
            );

            // Match this window
            let window_route = self.match_window_with_context(
                gps_points,
                timestamps,
                all_candidates,
                graph,
                segment_map,
                window_start,
                window_end,
                last_end_segment.as_ref(),
            )?;

            if window_route.is_empty() {
                debug!("No route found for window {}-{}", window_start, window_end);

                // Try with a smaller window before giving up
                if window_size > 3 && window_end > window_start + 2 {
                    let smaller_end = window_start + 2;
                    debug!("Trying smaller window {}-{}", window_start, smaller_end);

                    let smaller_route = self.match_window_with_context(
                        gps_points,
                        timestamps,
                        all_candidates,
                        graph,
                        segment_map,
                        window_start,
                        smaller_end,
                        last_end_segment.as_ref(),
                    )?;

                    if !smaller_route.is_empty() {
                        // Add to complete route
                        self.add_to_route(&mut complete_route, smaller_route);

                        // Update last segment
                        last_end_segment = complete_route.last().cloned();

                        // Adjust window for next iteration
                        window_start = smaller_end;
                        continue;
                    }
                }

                // If we can't find any route, we need to bridge the gap directly
                if let Some(last_segment) = &last_end_segment {
                    let next_valid_point =
                        self.find_next_valid_point(window_start, gps_points, all_candidates);

                    if let Some((next_idx, next_candidate)) = next_valid_point {
                        debug!(
                            "Bridging gap from point {} to {} with direct connection",
                            window_start, next_idx
                        );

                        // Try to connect directly
                        let bridge = self.bridge_gap(
                            last_segment,
                            &next_candidate.segment,
                            graph,
                            segment_map,
                        )?;

                        if !bridge.is_empty() {
                            // Add bridge to route
                            self.add_to_route(&mut complete_route, bridge);

                            // Update last segment
                            last_end_segment = complete_route.last().cloned();

                            // Skip to next valid point
                            window_start = next_idx;
                            continue;
                        }
                    }
                }

                // If we can't find any route or bridge, just advance window
                window_start += 1;
                continue;
            }

            // Add window route to complete route
            self.add_to_route(&mut complete_route, window_route);

            // Update last segment for next window
            last_end_segment = complete_route.last().cloned();

            // Move window forward
            window_start += step_size;
        }

        // Check for loops in the final route
        let final_route = self.remove_loops(complete_route)?;

        if final_route.is_empty() {
            return Err(anyhow!("Failed to build valid route"));
        }

        Ok(final_route)
    }

    /// Match a window of GPS points, considering context from previous window
    fn match_window_with_context(
        &mut self,
        gps_points: &[Point<f64>],
        timestamps: &[DateTime<Utc>],
        all_candidates: &[Vec<SegmentCandidate>],
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
        start_idx: usize,
        end_idx: usize,
        previous_segment: Option<&RoadSegment>,
    ) -> Result<Vec<RoadSegment>> {
        if start_idx > end_idx || end_idx >= gps_points.len() {
            return Err(anyhow!("Invalid window indices"));
        }

        // If there are no candidates for any point, return empty
        for i in start_idx..=end_idx {
            if all_candidates[i].is_empty() {
                debug!("No candidates for point {} in window", i);
                return Ok(Vec::new());
            }
        }

        // If we have context from previous window, constrain the first point's candidates
        let mut all_candidates_in_window = all_candidates[start_idx..=end_idx].to_vec();

        if let Some(prev_segment) = previous_segment {
            // Prioritize candidates that are connected to or close to previous segment
            let mut first_point_candidates = all_candidates_in_window[0].clone();

            // Check if any candidates are directly connected to previous segment
            let connected_candidates: Vec<_> = first_point_candidates
                .iter()
                .filter(|c| {
                    prev_segment.connections.contains(&c.segment.id)
                        || c.segment.connections.contains(&prev_segment.id)
                })
                .cloned()
                .collect();

            if !connected_candidates.is_empty() {
                // Use only connected candidates with their original scores
                debug!(
                    "Found {} candidates connected to previous segment",
                    connected_candidates.len()
                );
                all_candidates_in_window[0] = connected_candidates;
            } else {
                // If no connected candidates, adjust scores based on distance to previous segment
                let prev_end = Point::new(
                    prev_segment.coordinates.last().unwrap().x,
                    prev_segment.coordinates.last().unwrap().y,
                );

                for candidate in &mut first_point_candidates {
                    let candidate_start = Point::new(
                        candidate.segment.coordinates.first().unwrap().x,
                        candidate.segment.coordinates.first().unwrap().y,
                    );

                    let distance_to_prev = Haversine.distance(prev_end, candidate_start);

                    // Adjust score based on distance to previous segment
                    // Allow "bridging" gaps up to 50 meters without much penalty
                    if distance_to_prev <= 50.0 {
                        // Small adjustment for small gaps
                        candidate.score += distance_to_prev / 500.0;
                    } else {
                        // Larger penalty for bigger gaps
                        candidate.score += distance_to_prev / 100.0;
                    }
                }

                // Sort by adjusted scores
                first_point_candidates
                    .sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal));

                // Replace with adjusted candidates
                all_candidates_in_window[0] = first_point_candidates;
            }
        }

        // Find best route for this window
        let route = self.find_best_route_in_window(
            &gps_points[start_idx..=end_idx],
            &timestamps[start_idx..=end_idx],
            &all_candidates_in_window,
            graph,
            segment_map,
        )?;

        Ok(route)
    }

    /// Match a single window of GPS points when we have too few points for sliding
    fn match_single_window(
        &mut self,
        gps_points: &[Point<f64>],
        timestamps: &[DateTime<Utc>],
        all_candidates: &[Vec<SegmentCandidate>],
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
        start_idx: usize,
        end_idx: usize,
    ) -> Result<Vec<RoadSegment>> {
        // Get window candidates
        let window_candidates = &all_candidates[start_idx..=end_idx];

        // Find best route
        let route = self.find_best_route_in_window(
            &gps_points[start_idx..=end_idx],
            &timestamps[start_idx..=end_idx],
            window_candidates,
            graph,
            segment_map,
        )?;

        Ok(route)
    }

    /// Find the best route through a window of GPS points
    fn find_best_route_in_window(
        &self,
        window_points: &[Point<f64>],
        window_timestamps: &[DateTime<Utc>],
        window_candidates: &[Vec<SegmentCandidate>],
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
    ) -> Result<Vec<RoadSegment>> {
        if window_points.is_empty() || window_candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Strategy: evaluate all combinations of first and last point candidates
        // and find the shortest valid route between them that passes near intermediate points

        // Get candidates for first and last points
        let first_candidates = &window_candidates[0];
        let last_candidates = &window_candidates[window_candidates.len() - 1];

        if first_candidates.is_empty() || last_candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Track best route
        let mut best_route = Vec::new();
        let mut best_score = f64::MAX;

        // Try each combination of first and last candidates
        for first_candidate in first_candidates {
            for last_candidate in last_candidates {
                // Don't try to route from a segment to itself if window has multiple points
                if first_candidate.segment.id == last_candidate.segment.id
                    && window_points.len() > 1
                {
                    continue;
                }

                // Find shortest route between these segments
                let route_result = self.find_route_between_candidates(
                    &first_candidate.segment,
                    &last_candidate.segment,
                    window_points,
                    window_timestamps,
                    window_candidates,
                    graph,
                    segment_map,
                );

                if let Ok((route, score)) = route_result {
                    if score < best_score {
                        best_route = route;
                        best_score = score;
                    }
                }
            }
        }

        if best_route.is_empty() {
            debug!("Could not find any valid route in window");

            // Fall back to using just the best segments with no connectivity requirement
            if !first_candidates.is_empty() {
                best_route.push(first_candidates[0].segment.clone());

                // If first and last are different and we have multiple points, also add last
                if window_points.len() > 1
                    && !last_candidates.is_empty()
                    && first_candidates[0].segment.id != last_candidates[0].segment.id
                {
                    best_route.push(last_candidates[0].segment.clone());
                }
            }
        }

        Ok(best_route)
    }

    /// Find a route between two candidate segments that passes near intermediate points
    fn find_route_between_candidates(
        &self,
        from_segment: &RoadSegment,
        to_segment: &RoadSegment,
        window_points: &[Point<f64>],
        window_timestamps: &[DateTime<Utc>],
        window_candidates: &[Vec<SegmentCandidate>],
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
    ) -> Result<(Vec<RoadSegment>, f64)> {
        // If segments are the same or directly connected, return simple route
        if from_segment.id == to_segment.id {
            return Ok((vec![from_segment.clone()], 0.0));
        } else if from_segment.connections.contains(&to_segment.id) {
            return Ok((vec![from_segment.clone(), to_segment.clone()], 0.0));
        }

        // Calculate max allowed route length based on time and direct distance
        let time_diff = if window_timestamps.len() >= 2 {
            (*window_timestamps.last().unwrap() - *window_timestamps.first().unwrap()).num_seconds()
                as f64
        } else {
            0.0
        };

        // Calculate direct distance between endpoints
        let start_point = from_segment.centroid();
        let end_point = to_segment.centroid();
        let direct_distance = Haversine.distance(start_point, end_point);

        // Max length is max of:
        // 1. Direct distance * 2.5 for detour factor
        // 2. Time-based distance with reasonable speed (120 km/h = 33.3 m/s)
        const REASONABLE_SPEED: f64 = 33.3; // m/s

        let max_route_length = if time_diff > 0.0 {
            let speed_based = REASONABLE_SPEED * time_diff;
            let detour_based = direct_distance * 2.5;
            speed_based.min(detour_based.max(500.0)) // At least 500m to find some route
        } else {
            direct_distance * 2.5
        };

        // For intermediate waypoints, consider the best candidates at middle points
        let mut waypoints = HashSet::new();

        // If we have more than 2 points, consider intermediate waypoints
        if window_points.len() > 2 {
            // Get candidates for intermediate points
            for i in 1..window_candidates.len() - 1 {
                if !window_candidates[i].is_empty() {
                    // Take top candidates from each intermediate point
                    let num_waypoints = window_candidates[i].len().min(3); // Top 3 at most

                    for j in 0..num_waypoints {
                        waypoints.insert(window_candidates[i][j].segment.id);
                    }
                }
            }
        }

        // Try to find a path through waypoints
        let path_result = self.find_path_via_segment_waypoints(
            from_segment.id,
            to_segment.id,
            &waypoints,
            graph,
            segment_map,
            max_route_length,
        );

        match path_result {
            Ok((path, path_cost)) => {
                // Score is based on path cost and coverage of GPS points
                let mut score = path_cost;

                // Calculate how well the path covers GPS points
                if window_points.len() > 2 {
                    let coverage_score = self.calculate_path_coverage(&path, window_points);

                    // Add coverage score to total
                    score += coverage_score;
                }

                Ok((path, score))
            }
            Err(_) => {
                // If no path through waypoints, try direct path
                match self.find_path_with_distance_limit(
                    from_segment.id,
                    to_segment.id,
                    graph,
                    segment_map,
                    &HashSet::new(),
                    max_route_length,
                ) {
                    Ok((path_cost, path)) => {
                        let score = path_cost;
                        Ok((path, score))
                    }
                    Err(_) => {
                        // If still no path, consider bridging gap if distance is reasonable
                        if direct_distance <= 100.0 {
                            // Create a route with just the two segments
                            Ok((
                                vec![from_segment.clone(), to_segment.clone()],
                                direct_distance,
                            ))
                        } else {
                            Err(anyhow!("Could not find path between segments"))
                        }
                    }
                }
            }
        }
    }

    /// Calculate how well a path covers GPS points
    fn calculate_path_coverage(&self, path: &[RoadSegment], points: &[Point<f64>]) -> f64 {
        // Convert path to LineString for efficient distance calculations
        let mut path_coordinates = Vec::new();

        for segment in path {
            for coord in &segment.coordinates {
                path_coordinates.push(coord.clone());
            }
        }

        // Create LineString
        let path_line = LineString::from(path_coordinates);

        // Calculate sum of distances from GPS points to path
        let mut total_distance = 0.0;

        for point in points {
            let projected = match path_line.closest_point(point) {
                Closest::SinglePoint(p) => p,
                _ => *point, // Fallback
            };

            let distance = Haversine.distance(*point, projected);
            total_distance += distance;
        }

        // Return average distance
        total_distance / points.len() as f64
    }

    /// Find a route through specified waypoint segments
    fn find_path_via_segment_waypoints(
        &self,
        from_id: u64,
        to_id: u64,
        waypoint_ids: &HashSet<u64>,
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
        max_distance: f64,
    ) -> Result<(Vec<RoadSegment>, f64)> {
        if waypoint_ids.is_empty() {
            // No waypoints, just find direct path
            let result = self.find_path_with_distance_limit(
                from_id,
                to_id,
                graph,
                segment_map,
                &HashSet::new(),
                max_distance,
            )?;
            return Ok((result.1, result.0)); // Swap to match expected return type
        }

        // First try: find a path that includes at least one waypoint
        let mut best_path = Vec::new();
        let mut best_path_cost = f64::MAX;

        for &waypoint_id in waypoint_ids {
            // Skip if waypoint is the same as from or to
            if waypoint_id == from_id || waypoint_id == to_id {
                continue;
            }

            // Find path from start to waypoint
            let start_to_waypoint = match self.find_path_with_distance_limit(
                from_id,
                waypoint_id,
                graph,
                segment_map,
                &HashSet::new(),
                max_distance / 2.0,
            ) {
                Ok((cost, path)) => (cost, path),
                Err(_) => continue,
            };

            // Find path from waypoint to end
            let waypoint_to_end = match self.find_path_with_distance_limit(
                waypoint_id,
                to_id,
                graph,
                segment_map,
                &HashSet::new(),
                max_distance - start_to_waypoint.0,
            ) {
                Ok((cost, path)) => (cost, path),
                Err(_) => continue,
            };

            // Combine paths
            let total_cost = start_to_waypoint.0 + waypoint_to_end.0;

            // Skip first segment of second path (it's duplicated)
            let mut combined_path = start_to_waypoint.1;
            combined_path.extend(waypoint_to_end.1.into_iter().skip(1));

            // Update best path if this one is better
            if total_cost < best_path_cost {
                best_path = combined_path;
                best_path_cost = total_cost;
            }
        }

        if !best_path.is_empty() {
            return Ok((best_path, best_path_cost));
        }

        // If no path through waypoints, try a direct path
        let result = self.find_path_with_distance_limit(
            from_id,
            to_id,
            graph,
            segment_map,
            &HashSet::new(),
            max_distance,
        )?;
        Ok((result.1, result.0)) // Swap to match expected return type
    }

    /// Find the next point with valid candidates after a given index
    fn find_next_valid_point(
        &self,
        start_idx: usize,
        gps_points: &[Point<f64>],
        all_candidates: &[Vec<SegmentCandidate>],
    ) -> Option<(usize, SegmentCandidate)> {
        for i in start_idx + 1..gps_points.len() {
            if !all_candidates[i].is_empty() {
                return Some((i, all_candidates[i][0].clone()));
            }
        }

        None
    }

    /// Add segments to route, avoiding duplicates
    fn add_to_route(&self, route: &mut Vec<RoadSegment>, new_segments: Vec<RoadSegment>) {
        // If route is empty, just add all segments
        if route.is_empty() {
            route.extend(new_segments);
            return;
        }

        // Get last segment in route
        let last_segment_id = route.last().unwrap().id;

        // Skip first segment of new segments if it's the same as the last segment of route
        let start_idx = if !new_segments.is_empty() && new_segments[0].id == last_segment_id {
            1
        } else {
            0
        };

        // Add remaining segments
        for segment in new_segments.into_iter().skip(start_idx) {
            route.push(segment);
        }
    }

    /// Bridge a gap between two unconnected segments
    fn bridge_gap(
        &self,
        from_segment: &RoadSegment,
        to_segment: &RoadSegment,
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
    ) -> Result<Vec<RoadSegment>> {
        // If segments are the same, return just one
        if from_segment.id == to_segment.id {
            return Ok(vec![from_segment.clone()]);
        }

        // If segments are directly connected, return both
        if from_segment.connections.contains(&to_segment.id) {
            return Ok(vec![from_segment.clone(), to_segment.clone()]);
        }

        // Calculate direct distance
        let from_end = Point::new(
            from_segment.coordinates.last().unwrap().x,
            from_segment.coordinates.last().unwrap().y,
        );

        let to_start = Point::new(
            to_segment.coordinates.first().unwrap().x,
            to_segment.coordinates.first().unwrap().y,
        );

        let direct_distance = Haversine.distance(from_end, to_start);

        // If distance is small enough (50m), allow direct bridge
        if direct_distance <= 50.0 {
            return Ok(vec![from_segment.clone(), to_segment.clone()]);
        }

        // Try to find a path
        match self.find_path_with_distance_limit(
            from_segment.id,
            to_segment.id,
            graph,
            segment_map,
            &HashSet::new(),
            direct_distance * 2.0,
        ) {
            Ok((_, path)) => Ok(path),
            Err(_) => {
                // If no path and distance is reasonable, allow direct bridge
                if direct_distance <= 100.0 {
                    Ok(vec![from_segment.clone(), to_segment.clone()])
                } else {
                    Err(anyhow!("Could not bridge gap between segments"))
                }
            }
        }
    }

    /// Alternative matching approach when GPS speeds are unrealistic
    fn match_trace_with_outlier_removal(
        &mut self,
        gps_points: &[Point<f64>],
        timestamps: &[DateTime<Utc>],
        debug_way_ids: Option<&[u64]>,
    ) -> Result<Vec<RoadSegment>> {
        info!("Attempting match with outlier removal");

        // Find speed outliers
        let speed_outliers = self.identify_speed_outliers(gps_points, timestamps);

        if speed_outliers.is_empty() {
            return Err(anyhow!("Failed to identify speed outliers"));
        }

        info!("Identified {} speed outliers", speed_outliers.len());

        // Create filtered points and timestamps
        let mut filtered_points = Vec::new();
        let mut filtered_timestamps = Vec::new();

        for i in 0..gps_points.len() {
            if !speed_outliers.contains(&i) {
                filtered_points.push(gps_points[i]);
                filtered_timestamps.push(timestamps[i]);
            }
        }

        // Ensure we still have enough points
        if filtered_points.len() < 2 {
            return Err(anyhow!("Insufficient points after outlier removal"));
        }

        info!(
            "Retrying match with {} points after removing outliers",
            filtered_points.len()
        );

        // Retry matching with filtered points
        let start_time = std::time::Instant::now();

        // Step 1: Analyze GPS metrics for filtered points
        let metrics = self.analyze_gps_metrics(&filtered_points, &filtered_timestamps);

        // Check if we still have unreasonable speeds
        const MAX_ALLOWED_SPEED: f64 = 36.1; // 130 km/h in m/s

        if metrics.max_speed > MAX_ALLOWED_SPEED {
            warn!(
                "Filtered GPS trace still contains speeds exceeding 130 km/h (max: {:.1} km/h), proceeding anyway",
                metrics.max_speed * 3.6
            );
        }

        debug!(
            "Filtered GPS metrics - avg speed: {:.1} km/h, max speed: {:.1} km/h, total distance: {:.2} km",
            metrics.avg_speed * 3.6,
            metrics.max_speed * 3.6,
            metrics.total_distance / 1000.0
        );

        // Step 2: Load tiles covering the entire route area
        let trace_bbox = self.calculate_trace_bbox(&filtered_points);
        let buffer = self.config.max_matching_distance * 2.0 / 111_000.0;

        info!("Loading tiles for filtered route area");
        let loaded_tiles = self.tile_loader.load_tile_range(
            trace_bbox,
            buffer,
            self.config.max_tiles_per_depth,
        )?;

        // Step 3: Build road network graph
        info!("Building road network graph");
        let (graph, segment_map) = self.build_road_network(&loaded_tiles)?;

        // Step 4: Generate candidate segments for key points
        info!("Identifying key points and generating candidates");
        let (key_points, all_candidates) = self.generate_candidates_for_key_points(
            &filtered_points,
            &filtered_timestamps,
            &loaded_tiles,
            &metrics,
        )?;

        if key_points.is_empty() {
            return Err(anyhow!("No valid key points identified after filtering"));
        }

        // Step 5: Build route using shortest path with speed constraints
        info!("Building optimal route through filtered key points");

        // Use more relaxed constraints since we've filtered outliers
        let relaxed_metrics = GpsMetrics {
            reasonable_max_speed: metrics.reasonable_max_speed * 1.2, // Allow 20% higher speeds
            ..metrics
        };

        let initial_route = self.build_optimal_route(
            &key_points,
            &all_candidates,
            &graph,
            &segment_map,
            &relaxed_metrics,
        )?;

        // Step 6: Validate and correct the route
        info!("Validating and correcting filtered route");
        let validated_route = self.validate_and_correct_route(
            initial_route,
            &filtered_points,
            &filtered_timestamps,
            &graph,
            &segment_map,
            &loaded_tiles,
        )?;

        info!(
            "Map matching with outlier removal completed in {:.2?} with {} segments",
            start_time.elapsed(),
            validated_route.len()
        );

        // Debug information about specified way IDs
        if let Some(way_ids) = debug_way_ids {
            self.debug_way_ids(&validated_route, way_ids, &loaded_tiles)?;
        }

        Ok(validated_route)
    }

    /// Add this function to perform post-processing validation and correction
    fn validate_and_correct_route(
        &mut self,
        route: Vec<RoadSegment>,
        gps_points: &[Point<f64>],
        timestamps: &[DateTime<Utc>],
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
        loaded_tiles: &HashSet<String>,
    ) -> Result<Vec<RoadSegment>> {
        if route.len() < 2 {
            return Ok(route);
        }

        info!("Performing post-processing validation and correction");

        // Step 1: Identify problematic route segments with implausible "bridges"
        let problematic_segments = self.find_problematic_segments(&route, graph, segment_map);

        if problematic_segments.is_empty() {
            info!("Route validation successful - no problematic segments found");
            return Ok(route);
        }

        info!(
            "Found {} problematic segment transitions in route",
            problematic_segments.len()
        );

        // Step 2: Segment the route into valid chunks
        let chunks = self.split_route_at_problems(&route, &problematic_segments);
        info!("Split route into {} valid chunks", chunks.len());

        // Step 3: Find best GPS point to match for each chunk
        let chunk_assignments = self.assign_gps_points_to_chunks(&chunks, gps_points, timestamps);

        // Step 4: Rebuild the route by reconnecting chunks with better transitions
        let corrected_route = self.rebuild_route_from_chunks(
            chunks,
            &chunk_assignments,
            gps_points,
            timestamps,
            graph,
            segment_map,
            loaded_tiles,
        )?;

        // Final validation - check if we still have problems
        let remaining_problems =
            self.find_problematic_segments(&corrected_route, graph, segment_map);

        if !remaining_problems.is_empty() {
            info!(
                "Route still has {} problematic segments after correction, performing additional fixes",
                remaining_problems.len()
            );

            // Try more aggressive correction for remaining problems
            return self.fix_remaining_problems(
                corrected_route,
                &remaining_problems,
                gps_points,
                timestamps,
                graph,
                segment_map,
                loaded_tiles,
            );
        }

        info!("Route correction successful - all problems resolved");
        Ok(corrected_route)
    }

    /// Find problematic segments in the route
    fn find_problematic_segments(
        &self,
        route: &[RoadSegment],
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
    ) -> Vec<usize> {
        let mut problematic_indices = Vec::new();

        for i in 0..route.len() - 1 {
            let current = &route[i];
            let next = &route[i + 1];

            // Check 1: Check if segments are disconnected
            let disconnected = !current.connections.contains(&next.id);

            if disconnected {
                // Calculate direct distance between segment endpoints
                let current_end = Point::new(
                    current.coordinates.last().unwrap().x,
                    current.coordinates.last().unwrap().y,
                );

                let next_start = Point::new(
                    next.coordinates.first().unwrap().x,
                    next.coordinates.first().unwrap().y,
                );

                let bridge_distance = Haversine.distance(current_end, next_start);

                // Problem if bridge is too long (more than 50 meters)
                if bridge_distance > 50.0 {
                    debug!(
                        "Problematic transition detected at segments {}-{}: bridge distance {:.1}m",
                        i,
                        i + 1,
                        bridge_distance
                    );
                    problematic_indices.push(i);
                }
            }

            // Check 2: Check for unreasonable turns (U-turns or sharp turns on high-speed roads)
            let current_dir = if current.coordinates.len() >= 2 {
                let first = current.coordinates[current.coordinates.len() - 2];
                let last = current.coordinates.last().unwrap();

                calculate_heading(Point::new(first.x, first.y), Point::new(last.x, last.y))
            } else {
                0.0
            };

            let next_dir = if next.coordinates.len() >= 2 {
                let first = next.coordinates.first().unwrap();
                let second = next.coordinates[1];

                calculate_heading(Point::new(first.x, first.y), Point::new(second.x, second.y))
            } else {
                0.0
            };

            let turn_angle = angle_difference(current_dir, next_dir);

            // Check if this is a sharp turn on a major road
            let is_major_road = calculate_road_class_score(&current.highway_type) < 3.0
                || calculate_road_class_score(&next.highway_type) < 3.0;

            if is_major_road && turn_angle > 120.0 {
                debug!(
                    "Problematic turn detected at segments {}-{}: angle {:.1}° on major road",
                    i,
                    i + 1,
                    turn_angle
                );

                if !problematic_indices.contains(&i) {
                    problematic_indices.push(i);
                }
            }
        }

        problematic_indices
    }

    /// Split route into valid chunks
    fn split_route_at_problems(
        &self,
        route: &[RoadSegment],
        problematic_indices: &[usize],
    ) -> Vec<Vec<RoadSegment>> {
        let mut chunks = Vec::new();
        let mut start_idx = 0;

        for &prob_idx in problematic_indices {
            // Create chunk from start to problem (inclusive)
            let chunk = route[start_idx..=prob_idx].to_vec();
            chunks.push(chunk);

            // Next chunk starts after the problem
            start_idx = prob_idx + 1;
        }

        // Add final chunk if needed
        if start_idx < route.len() {
            chunks.push(route[start_idx..].to_vec());
        }

        chunks
    }

    /// Assign GPS points to route chunks
    fn assign_gps_points_to_chunks(
        &self,
        chunks: &[Vec<RoadSegment>],
        gps_points: &[Point<f64>],
        timestamps: &[DateTime<Utc>],
    ) -> Vec<ChunkAssignment> {
        let mut assignments = Vec::new();

        // Skip if no chunks or no GPS points
        if chunks.is_empty() || gps_points.is_empty() {
            return assignments;
        }

        let mut remaining_points: Vec<(usize, Point<f64>, DateTime<Utc>)> = gps_points
            .iter()
            .zip(timestamps)
            .enumerate()
            .map(|(i, (p, t))| (i, *p, *t))
            .collect();

        // For each chunk, find the GPS points that best match
        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            // Skip empty chunks
            if chunk.is_empty() {
                continue;
            }

            let mut points_for_chunk = Vec::new();
            let mut best_matches = Vec::new();

            // Create a set of segment IDs for this chunk
            let segment_ids: HashSet<u64> = chunk.iter().map(|s| s.id).collect();

            // For each segment in the chunk, find best matching GPS points
            for segment in chunk {
                // Convert segment to linestring for distance calculation
                let line = LineString::from(segment.coordinates.clone());

                // Score each remaining GPS point
                let mut point_scores = Vec::new();

                for (idx, (gps_idx, point, _)) in remaining_points.iter().enumerate() {
                    // Calculate distance to segment
                    let distance = match line.closest_point(point) {
                        Closest::SinglePoint(projected) => Haversine.distance(*point, projected),
                        _ => f64::MAX,
                    };

                    // Score is simply distance (lower is better)
                    if distance <= 100.0 {
                        // Only consider points within 100m
                        point_scores.push((idx, *gps_idx, distance));
                    }
                }

                // Sort by distance
                point_scores.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

                // Take best match for this segment if available
                if !point_scores.is_empty() {
                    best_matches.push(point_scores[0].1); // Add GPS index

                    // Remove this point from remaining points to avoid duplicates
                    let idx_to_remove = point_scores[0].0;
                    if idx_to_remove < remaining_points.len() {
                        points_for_chunk.push(remaining_points.remove(idx_to_remove));
                    }
                }
            }

            // If no points were matched directly, try to match by proximity
            if points_for_chunk.is_empty() && !remaining_points.is_empty() {
                // Find centroid of the chunk
                let chunk_points: Vec<Point<f64>> = chunk
                    .iter()
                    .flat_map(|s| s.coordinates.iter().map(|c| Point::new(c.x, c.y)))
                    .collect();

                let chunk_centroid = if !chunk_points.is_empty() {
                    let sum_x: f64 = chunk_points.iter().map(|p| p.x()).sum();
                    let sum_y: f64 = chunk_points.iter().map(|p| p.y()).sum();
                    Point::new(
                        sum_x / chunk_points.len() as f64,
                        sum_y / chunk_points.len() as f64,
                    )
                } else {
                    // Fallback to first point in chunk
                    chunk[0].centroid()
                };

                // Score remaining points by distance to centroid
                let mut centroid_scores = Vec::new();

                for (idx, (gps_idx, point, time)) in remaining_points.iter().enumerate() {
                    let distance = Haversine.distance(*point, chunk_centroid);
                    centroid_scores.push((idx, *gps_idx, *time, distance));
                }

                // Sort by distance
                centroid_scores.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(Ordering::Equal));

                // Take closest point
                if !centroid_scores.is_empty() {
                    let idx_to_remove = centroid_scores[0].0;
                    best_matches.push(centroid_scores[0].1); // Add GPS index

                    if idx_to_remove < remaining_points.len() {
                        points_for_chunk.push(remaining_points.remove(idx_to_remove));
                    }
                }
            }

            // Record assignment if we have points
            if !points_for_chunk.is_empty() {
                // Sort GPS indices to maintain order
                best_matches.sort_unstable();

                // Get min/max timestamps
                let start_time = points_for_chunk
                    .iter()
                    .map(|(_, _, t)| *t)
                    .min()
                    .unwrap_or(timestamps[0]);

                let end_time = points_for_chunk
                    .iter()
                    .map(|(_, _, t)| *t)
                    .max()
                    .unwrap_or(timestamps[0]);

                assignments.push(ChunkAssignment {
                    chunk_idx,
                    gps_indices: best_matches,
                    start_time,
                    end_time,
                });
            } else {
                // No points matched - still create assignment with time estimation
                let prev_assignment = if chunk_idx > 0 && !assignments.is_empty() {
                    assignments.last()
                } else {
                    None
                };

                let next_points = if !remaining_points.is_empty() {
                    Some(&remaining_points[0])
                } else {
                    None
                };

                // Estimate times based on surrounding data
                let (start_time, end_time) = match (prev_assignment, next_points) {
                    (Some(prev), Some(&(_, _, next_time))) => {
                        let time_diff = (next_time - prev.end_time).num_seconds() as f64 / 2.0;
                        let mid_time = prev.end_time + Duration::seconds(time_diff as i64);
                        (prev.end_time, mid_time)
                    }
                    (Some(prev), None) => (prev.end_time, prev.end_time),
                    (None, Some(&(_, _, time))) => (time, time),
                    (None, None) => (timestamps[0], timestamps[0]),
                };

                assignments.push(ChunkAssignment {
                    chunk_idx,
                    gps_indices: Vec::new(),
                    start_time,
                    end_time,
                });
            }
        }

        // Sort assignments by chunk index
        assignments.sort_by_key(|a| a.chunk_idx);

        assignments
    }

    /// Rebuild route from chunks with better transitions
    fn rebuild_route_from_chunks(
        &mut self,
        chunks: Vec<Vec<RoadSegment>>,
        assignments: &[ChunkAssignment],
        gps_points: &[Point<f64>],
        timestamps: &[DateTime<Utc>],
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
        loaded_tiles: &HashSet<String>,
    ) -> Result<Vec<RoadSegment>> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        if chunks.len() == 1 {
            return Ok(chunks[0].clone());
        }

        let mut corrected_route = Vec::new();

        // Add first chunk as-is
        corrected_route.extend(chunks[0].clone());

        // Process remaining chunks
        for i in 1..chunks.len() {
            let prev_chunk = &chunks[i - 1];
            let curr_chunk = &chunks[i];

            if prev_chunk.is_empty() || curr_chunk.is_empty() {
                continue;
            }

            // Get last segment of previous chunk and first segment of current chunk
            let prev_end_segment = prev_chunk.last().unwrap();
            let curr_start_segment = curr_chunk.first().unwrap();

            // Find transition between chunks
            let transition = self.find_better_transition(
                prev_end_segment,
                curr_start_segment,
                &assignments[i - 1], // Previous chunk assignment
                &assignments[i],     // Current chunk assignment
                gps_points,
                timestamps,
                graph,
                segment_map,
                loaded_tiles,
            )?;

            // Add transition segments
            for segment in transition {
                // Check if segment is already in route
                if corrected_route.last().is_some_and(|s| s.id == segment.id) {
                    continue;
                }

                corrected_route.push(segment);
            }

            // Add current chunk segments (except first if it's already added)
            let start_idx = if !curr_chunk.is_empty()
                && !corrected_route.is_empty()
                && corrected_route.last().unwrap().id == curr_chunk[0].id
            {
                1 // Skip first segment
            } else {
                0 // Include all segments
            };

            for segment in &curr_chunk[start_idx..] {
                // Check if segment is already in route
                if corrected_route.last().is_some_and(|s| s.id == segment.id) {
                    continue;
                }

                corrected_route.push(segment.clone());
            }
        }

        // Verify no loops in final route
        let mut unique_ids = HashSet::new();
        for segment in &corrected_route {
            if !unique_ids.insert(segment.id) {
                warn!("Loop detected in corrected route, attempting to fix");
                return self.remove_loops(corrected_route);
            }
        }

        Ok(corrected_route)
    }

    /// Find better transition between chunks
    fn find_better_transition(
        &mut self,
        prev_end_segment: &RoadSegment,
        curr_start_segment: &RoadSegment,
        prev_assignment: &ChunkAssignment,
        curr_assignment: &ChunkAssignment,
        gps_points: &[Point<f64>],
        timestamps: &[DateTime<Utc>],
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
        loaded_tiles: &HashSet<String>,
    ) -> Result<Vec<RoadSegment>> {
        // If segments are already connected, no transition needed
        if prev_end_segment
            .connections
            .contains(&curr_start_segment.id)
        {
            return Ok(vec![curr_start_segment.clone()]);
        }

        // Calculate max route length based on time between assignments
        let time_diff =
            (curr_assignment.start_time - prev_assignment.end_time).num_seconds() as f64;

        // Reasonable max speed in m/s (120 km/h = 33.3 m/s)
        const REASONABLE_SPEED: f64 = 33.3;

        // Direct distance between segments
        let prev_point = prev_end_segment.centroid();
        let curr_point = curr_start_segment.centroid();
        let direct_distance = Haversine.distance(prev_point, curr_point);

        // Max route length is max of:
        // 1. Speed-based max distance (reasonable speed * time)
        // 2. Direct distance with detour factor
        let max_route_length = if time_diff > 0.0 {
            // Allow reasonable speed plus some buffer
            let speed_max = REASONABLE_SPEED * time_diff;

            // Allow up to 2x the direct distance
            let detour_max = direct_distance * 2.0;

            // Use the smaller of the two limits
            speed_max.min(detour_max.max(500.0)) // At least 500m to find some route
        } else {
            // If time is too small, use direct distance + buffer
            direct_distance * 2.0
        };

        // Try to find path using A* with distance limit
        match self.find_path_with_distance_limit(
            prev_end_segment.id,
            curr_start_segment.id,
            graph,
            segment_map,
            &HashSet::new(), // No segments to avoid yet
            max_route_length,
        ) {
            Ok((_, path)) => {
                // Found path within constraints
                return Ok(path);
            }
            Err(_) => {
                // Path not found within constraints, try with GPS points
                debug!("No direct path found, trying to find transition with GPS points");
            }
        }

        // Try to find alternative route using GPS points between chunks
        let transition_gps_indices =
            self.find_transition_gps_indices(prev_assignment, curr_assignment, timestamps);

        // Get GPS points that might help find transition
        let transition_points: Vec<Point<f64>> = transition_gps_indices
            .iter()
            .map(|&idx| gps_points[idx])
            .collect();

        // If we have GPS points, try to find a route through them
        if !transition_points.is_empty() {
            debug!(
                "Using {} GPS points to help find transition",
                transition_points.len()
            );

            // Find candidate segments near these points
            let mut candidate_segments = HashSet::new();

            for point in &transition_points {
                let nearby = self.find_segments_near_point(
                    *point,
                    loaded_tiles,
                    100.0, // 100m search radius
                )?;

                for segment in nearby {
                    candidate_segments.insert(segment.id);
                }
            }

            // Try to find a path using these candidates as waypoints
            if !candidate_segments.is_empty() {
                let path = self.find_path_via_waypoints(
                    prev_end_segment.id,
                    curr_start_segment.id,
                    &candidate_segments,
                    graph,
                    segment_map,
                    max_route_length,
                );

                if let Ok(route) = path {
                    return Ok(route);
                }
            }
        }

        // Fallback: just try direct connection with higher limit
        debug!("Fallback: trying direct connection with higher distance limit");

        match self.find_path_with_distance_limit(
            prev_end_segment.id,
            curr_start_segment.id,
            graph,
            segment_map,
            &HashSet::new(),
            max_route_length * 2.0, // Double the limit for fallback
        ) {
            Ok((_, path)) => Ok(path),
            Err(_) => {
                // Still no path, just return empty vector and let caller handle it
                debug!(
                    "No path found between segments {} and {}",
                    prev_end_segment.id, curr_start_segment.id
                );
                Ok(Vec::new())
            }
        }
    }

    /// Find GPS indices that might help with the transition
    fn find_transition_gps_indices(
        &self,
        prev_assignment: &ChunkAssignment,
        curr_assignment: &ChunkAssignment,
        timestamps: &[DateTime<Utc>],
    ) -> Vec<usize> {
        let mut transition_indices = Vec::new();

        // If we have GPS indices in assignments, find points between
        if !prev_assignment.gps_indices.is_empty() && !curr_assignment.gps_indices.is_empty() {
            let prev_max = *prev_assignment.gps_indices.iter().max().unwrap();
            let curr_min = *curr_assignment.gps_indices.iter().min().unwrap();

            // Get indices between assignments
            for idx in prev_max + 1..curr_min {
                transition_indices.push(idx);
            }
        }

        // If we don't have any points, try to find points by timestamp
        if transition_indices.is_empty() {
            for i in 0..timestamps.len() {
                // Check if timestamp is between assignments
                if timestamps[i] > prev_assignment.end_time
                    && timestamps[i] < curr_assignment.start_time
                {
                    transition_indices.push(i);
                }
            }
        }

        transition_indices
    }

    /// Find segments near a GPS point
    fn find_segments_near_point(
        &mut self,
        point: Point<f64>,
        loaded_tiles: &HashSet<String>,
        max_distance: f64,
    ) -> Result<Vec<RoadSegment>> {
        let mut nearby_segments = Vec::new();

        for tile_id in loaded_tiles {
            // Load tile and get segments
            let segments_in_tile = {
                let tile = self.tile_loader.load_tile(tile_id)?;
                tile.road_segments.clone()
            };

            // Check each segment
            for segment in &segments_in_tile {
                let projection = self.project_point_to_segment(point, segment);
                let distance = Haversine.distance(point, projection);

                if distance <= max_distance {
                    nearby_segments.push((segment.clone(), distance));
                }
            }
        }

        // Sort by distance
        nearby_segments.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Return segments
        Ok(nearby_segments.into_iter().map(|(s, _)| s).collect())
    }

    /// Find a path through candidate waypoints
    fn find_path_via_waypoints(
        &self,
        start_id: u64,
        end_id: u64,
        candidate_waypoints: &HashSet<u64>,
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
        max_distance: f64,
    ) -> Result<Vec<RoadSegment>> {
        // If no candidates, just try direct path
        if candidate_waypoints.is_empty() {
            return self
                .find_path_with_distance_limit(
                    start_id,
                    end_id,
                    graph,
                    segment_map,
                    &HashSet::new(),
                    max_distance,
                )
                .map(|(_, path)| path);
        }

        // Try each candidate as a waypoint
        let mut best_path = Vec::new();
        let mut best_length = f64::MAX;

        for &waypoint_id in candidate_waypoints {
            // Skip if same as start or end
            if waypoint_id == start_id || waypoint_id == end_id {
                continue;
            }

            // Find path from start to waypoint
            let start_to_waypoint = match self.find_path_with_distance_limit(
                start_id,
                waypoint_id,
                graph,
                segment_map,
                &HashSet::new(),
                max_distance / 2.0,
            ) {
                Ok((_, path)) => path,
                Err(_) => continue,
            };

            // Calculate total length so far
            let first_length: f64 = start_to_waypoint.iter().map(|s| s.length()).sum();

            // If already too long, skip
            if first_length > max_distance {
                continue;
            }

            // Find path from waypoint to end
            let waypoint_to_end = match self.find_path_with_distance_limit(
                waypoint_id,
                end_id,
                graph,
                segment_map,
                &HashSet::new(),
                max_distance - first_length,
            ) {
                Ok((_, path)) => path,
                Err(_) => continue,
            };

            // Calculate total length
            let mut full_path = start_to_waypoint.clone();

            // Add the waypoint to end path (skip first segment as it's already in start_to_waypoint)
            for segment in waypoint_to_end.iter().skip(1) {
                full_path.push(segment.clone());
            }

            let total_length: f64 = full_path.iter().map(|s| s.length()).sum();

            // If better than current best, update
            if total_length < best_length {
                best_path = full_path;
                best_length = total_length;
            }
        }

        if !best_path.is_empty() {
            Ok(best_path)
        } else {
            Err(anyhow!("No path found via waypoints"))
        }
    }

    /// Fix remaining problematic segments more aggressively
    fn fix_remaining_problems(
        &mut self,
        route: Vec<RoadSegment>,
        problematic_indices: &[usize],
        gps_points: &[Point<f64>],
        timestamps: &[DateTime<Utc>],
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
        loaded_tiles: &HashSet<String>,
    ) -> Result<Vec<RoadSegment>> {
        if route.is_empty() || problematic_indices.is_empty() {
            return Ok(route);
        }

        // More aggressive approach: simply remove problematic segments
        // and try to find direct connections between valid segments

        let mut fixed_route = Vec::new();
        let mut last_valid_idx = 0;

        // Add first segment
        fixed_route.push(route[0].clone());

        // Handle each problematic segment
        for &prob_idx in problematic_indices {
            // Skip if first or last segment (can't remove those)
            if prob_idx == 0 || prob_idx >= route.len() - 1 {
                continue;
            }

            // Add segments between last valid and current problem
            for i in last_valid_idx + 1..prob_idx {
                fixed_route.push(route[i].clone());
            }

            // Find next valid segment after problem
            let next_valid_idx = prob_idx + 1;

            if next_valid_idx >= route.len() {
                // No more valid segments
                break;
            }

            // Get last valid segment and next valid segment
            let last_valid = fixed_route.last().unwrap();
            let next_valid = &route[next_valid_idx];

            // Try to find connecting path
            let connecting_path = self.find_better_transition(
                last_valid,
                next_valid,
                &ChunkAssignment {
                    chunk_idx: 0,
                    gps_indices: Vec::new(),
                    start_time: timestamps[0],
                    end_time: timestamps[0],
                },
                &ChunkAssignment {
                    chunk_idx: 1,
                    gps_indices: Vec::new(),
                    start_time: timestamps[0],
                    end_time: timestamps[0],
                },
                gps_points,
                timestamps,
                graph,
                segment_map,
                loaded_tiles,
            )?;

            // Add connecting path
            for segment in connecting_path {
                // Check if segment is already in route
                if fixed_route.last().is_some_and(|s| s.id == segment.id) {
                    continue;
                }

                fixed_route.push(segment);
            }

            // Update last valid index
            last_valid_idx = next_valid_idx;
        }

        // Add remaining segments
        for i in last_valid_idx + 1..route.len() {
            fixed_route.push(route[i].clone());
        }

        // Verify no loops
        let mut unique_ids = HashSet::new();
        for segment in &fixed_route {
            if !unique_ids.insert(segment.id) {
                warn!("Loop detected in final fixed route, removing duplicates");
                return self.remove_loops(fixed_route);
            }
        }

        Ok(fixed_route)
    }

    /// Identify indices of GPS points that create unrealistic speeds
    fn identify_speed_outliers(
        &self,
        gps_points: &[Point<f64>],
        timestamps: &[DateTime<Utc>],
    ) -> HashSet<usize> {
        let mut outliers = HashSet::new();
        const MAX_ALLOWED_SPEED: f64 = 36.1; // 130 km/h in m/s

        // Find points that create high speeds (either as start or end of a segment)
        for i in 1..gps_points.len() {
            let prev_point = gps_points[i - 1];
            let curr_point = gps_points[i];

            let distance = Haversine.distance(prev_point, curr_point);
            let time_diff = (timestamps[i] - timestamps[i - 1]).num_seconds() as f64;

            if time_diff > 0.0 {
                let speed = distance / time_diff; // m/s

                if speed > MAX_ALLOWED_SPEED {
                    // Determine which point is more likely to be an outlier
                    if i > 1 && i < gps_points.len() - 1 {
                        // Check speeds with adjacent points
                        let prev_prev_point = gps_points[i - 2];
                        let next_point = gps_points[i];

                        let prev_segment_dist = Haversine.distance(prev_prev_point, prev_point);
                        let next_segment_dist = Haversine.distance(curr_point, next_point);

                        let prev_time_diff =
                            (timestamps[i - 1] - timestamps[i - 2]).num_seconds() as f64;
                        let next_time_diff =
                            (timestamps[i] - timestamps[i - 1]).num_seconds() as f64;

                        let prev_speed = if prev_time_diff > 0.0 {
                            prev_segment_dist / prev_time_diff
                        } else {
                            0.0
                        };
                        let next_speed = if next_time_diff > 0.0 {
                            next_segment_dist / next_time_diff
                        } else {
                            0.0
                        };

                        // Mark the point that appears more anomalous
                        if prev_speed > MAX_ALLOWED_SPEED && next_speed <= MAX_ALLOWED_SPEED {
                            outliers.insert(i - 1); // Previous point is outlier
                        } else if prev_speed <= MAX_ALLOWED_SPEED && next_speed > MAX_ALLOWED_SPEED
                        {
                            outliers.insert(i); // Current point is outlier
                        } else {
                            // Both speeds are problematic or both are ok, mark both points
                            outliers.insert(i - 1);
                            outliers.insert(i);
                        }
                    } else {
                        // Edge points, mark both as potential outliers
                        outliers.insert(i - 1);
                        outliers.insert(i);
                    }
                }
            }
        }

        // Avoid removing first and last points if possible
        if outliers.len() > 2 {
            outliers.remove(&0);
            outliers.remove(&(gps_points.len() - 1));
        }

        outliers
    }

    /// Extract speed and distance metrics from GPS points
    fn analyze_gps_metrics(
        &self,
        gps_points: &[Point<f64>],
        timestamps: &[DateTime<Utc>],
    ) -> GpsMetrics {
        let mut total_distance = 0.0;
        let mut total_time = 0.0;
        let mut max_speed = 0.0;
        let mut speeds = Vec::new();
        let mut point_distances = Vec::new();

        // Calculate distances and speeds between consecutive points
        for i in 1..gps_points.len() {
            let prev_point = gps_points[i - 1];
            let curr_point = gps_points[i];

            let distance = Haversine.distance(prev_point, curr_point);
            point_distances.push(distance);
            total_distance += distance;

            let time_diff = (timestamps[i] - timestamps[i - 1]).num_seconds() as f64;
            if time_diff > 0.0 {
                let speed = distance / time_diff; // m/s
                speeds.push(speed);
                total_time += time_diff;

                max_speed = max_speed.max(speed);
            }
        }

        // Calculate average speed and other metrics
        let avg_speed = if total_time > 0.0 {
            total_distance / total_time
        } else {
            0.0
        };

        // Set reasonable maximum speed (1.5x avg speed or at least 25 m/s which is 90 km/h)
        let reasonable_max_speed = (avg_speed * 1.5).max(25.0);

        // Average distance between consecutive points
        let avg_point_distance = if !point_distances.is_empty() {
            point_distances.iter().sum::<f64>() / point_distances.len() as f64
        } else {
            0.0
        };

        GpsMetrics {
            avg_speed,
            max_speed,
            reasonable_max_speed,
            min_travel_time: 1.0, // Minimum 1 second between points
            gps_error: 75.0,      // 75m GPS accuracy as specified
            total_distance,
            avg_point_distance,
        }
    }

    /// Generate candidates for key GPS points
    fn generate_candidates_for_key_points(
        &mut self,
        gps_points: &[Point<f64>],
        timestamps: &[DateTime<Utc>],
        loaded_tiles: &HashSet<String>,
        metrics: &GpsMetrics,
    ) -> Result<(Vec<KeyPoint>, Vec<HashSet<u64>>)> {
        // Determine which points to use as key points
        // For sparse data (fewer than 20 points), use all points
        // For dense data, sample intelligently

        let mut key_point_indices = Vec::new();

        if gps_points.len() <= 20 {
            // Use all points for sparse data
            key_point_indices.extend(0..gps_points.len());
        } else {
            // Always include first and last points
            key_point_indices.push(0);
            key_point_indices.push(gps_points.len() - 1);

            // Add points where significant direction changes occur
            for i in 1..gps_points.len() - 1 {
                if self.is_significant_turn(gps_points, i) {
                    key_point_indices.push(i);
                }
            }

            // Add additional points to ensure reasonable coverage
            let max_gap = (gps_points.len() / 10).max(5); // Max 10% or at least 5 points

            let mut i = 0;
            while i < gps_points.len() {
                if !key_point_indices.contains(&i) {
                    key_point_indices.push(i);
                }
                i += max_gap;
            }

            // Sort indices
            key_point_indices.sort_unstable();
        }

        debug!(
            "Selected {} key points out of {} total points",
            key_point_indices.len(),
            gps_points.len()
        );

        // Generate candidates for each key point
        let mut key_points = Vec::new();
        let mut all_candidates = vec![HashSet::new(); gps_points.len()];

        for &idx in &key_point_indices {
            let point = gps_points[idx];
            let timestamp = timestamps[idx];

            // Get previous and next points for direction context
            let prev_idx = if idx > 0 { idx - 1 } else { 0 };
            let next_idx = if idx < gps_points.len() - 1 {
                idx + 1
            } else {
                idx
            };

            let prev_point = gps_points[prev_idx];
            let next_point = gps_points[next_idx];

            // Calculate approach and departure directions
            let approach_dir = if idx > 0 {
                calculate_heading(prev_point, point)
            } else {
                calculate_heading(point, next_point)
            };

            let depart_dir = if idx < gps_points.len() - 1 {
                calculate_heading(point, next_point)
            } else {
                calculate_heading(prev_point, point)
            };

            // Find candidate segments for this point
            let mut candidates = Vec::new();

            for tile_id in loaded_tiles {
                // First, load the tile and clone the segments to avoid borrow checker issues
                let segments_in_tile = {
                    let tile = self.tile_loader.load_tile(tile_id)?;
                    tile.road_segments.clone()
                };

                // Now use the cloned segments
                for segment in &segments_in_tile {
                    let projection = self.project_point_to_segment(point, segment);
                    let distance = Haversine.distance(point, projection);

                    // Consider GPS error in matching distance
                    if distance <= metrics.gps_error * 1.5 {
                        // Calculate segment direction at projection
                        let segment_dir = segment.direction_at_point(projection);

                        // Direction difference (average of approach and departure)
                        let approach_diff = angle_difference(approach_dir, segment_dir);
                        let depart_diff = angle_difference(depart_dir, segment_dir);
                        let direction_diff = (approach_diff + depart_diff) / 2.0;

                        // Score based on distance and direction
                        // - Distance is weighted by GPS error
                        // - Direction is important but less so for short segments

                        let distance_factor = distance / metrics.gps_error;
                        let direction_factor = direction_diff / 45.0; // Normalize to make 45° = 1.0

                        // Road class factor - prefer major roads slightly
                        let road_class_score = calculate_road_class_score(&segment.highway_type);
                        let road_factor = road_class_score / 10.0;

                        // Combined score (lower is better)
                        let score = distance_factor + direction_factor + road_factor;

                        candidates.push(SegmentCandidate {
                            segment: segment.clone(),
                            distance,
                            score,
                            projection,
                        });

                        // Track all candidate segments for this point
                        all_candidates[idx].insert(segment.id);
                    }
                }
            }

            // Sort candidates by score
            candidates.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal));

            // Limit number of candidates
            let max_candidates = self.config.max_candidates_per_point.min(candidates.len());
            candidates.truncate(max_candidates);

            // Create key point
            let key_point = KeyPoint {
                index: idx,
                point,
                timestamp,
                candidates,
            };

            key_points.push(key_point);
        }

        if key_points.is_empty() {
            return Err(anyhow!("No valid key points with candidates"));
        }

        Ok((key_points, all_candidates))
    }

    /// Check if point represents a significant turn in the GPS trace
    fn is_significant_turn(&self, gps_points: &[Point<f64>], index: usize) -> bool {
        if index == 0 || index >= gps_points.len() - 1 {
            return false;
        }

        // Get points before and after
        let prev_point = gps_points[index - 1];
        let curr_point = gps_points[index];
        let next_point = gps_points[index + 1];

        // Calculate directions
        let incoming_direction = calculate_heading(prev_point, curr_point);
        let outgoing_direction = calculate_heading(curr_point, next_point);

        // Calculate angle difference
        let angle_diff = angle_difference(incoming_direction, outgoing_direction);

        // Consider it significant if the turn is more than 30 degrees
        angle_diff > 30.0
    }

    /// Build optimal route through key points with speed constraints
    fn build_optimal_route(
        &self,
        key_points: &[KeyPoint],
        all_candidates: &[HashSet<u64>],
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
        metrics: &GpsMetrics,
    ) -> Result<Vec<RoadSegment>> {
        if key_points.is_empty() {
            return Err(anyhow!("No key points provided"));
        }

        // Start with a route from the first key point's best candidate
        if key_points[0].candidates.is_empty() {
            return Err(anyhow!("First key point has no candidates"));
        }

        let mut route = vec![key_points[0].candidates[0].segment.clone()];
        let mut used_segment_ids = HashSet::new();
        used_segment_ids.insert(route[0].id);

        // Process each subsequent key point
        for i in 1..key_points.len() {
            let prev_key_point = &key_points[i - 1];
            let curr_key_point = &key_points[i];

            debug!(
                "Connecting key points {}/{} (GPS indices {} to {})",
                i,
                key_points.len() - 1,
                prev_key_point.index,
                curr_key_point.index
            );

            // Calculate time difference between key points
            let time_diff =
                (curr_key_point.timestamp - prev_key_point.timestamp).num_seconds() as f64;

            // Direct distance between key points
            let direct_distance = Haversine.distance(prev_key_point.point, curr_key_point.point);

            // Calculate reasonable maximum route distance based on speed constraints
            // Allow for some detour, but prevent unreasonably long routes
            let max_allowed_distance = if time_diff > 0.0 {
                // Based on reasonable max speed
                let speed_based_max = metrics.reasonable_max_speed * time_diff;

                // Allow detour based on direct distance
                let detour_max = direct_distance * 2.5; // Allow up to 2.5x direct distance

                // Take the smaller of the two constraints
                speed_based_max.min(detour_max)
            } else {
                // Fallback if time difference is too small
                direct_distance * 2.5
            };

            debug!(
                "Key points {}-{}: time {}s, direct distance {:.1}m, max allowed {:.1}m",
                i - 1,
                i,
                time_diff,
                direct_distance,
                max_allowed_distance
            );

            // Get last segment in current route
            let last_segment = route.last().unwrap();

            // Find best connection to current key point
            let next_segments = self.find_best_connection(
                last_segment,
                curr_key_point,
                &used_segment_ids,
                graph,
                segment_map,
                max_allowed_distance,
            )?;

            if next_segments.is_empty() {
                debug!(
                    "Could not find connection to key point {}, using best candidate directly",
                    i
                );

                // Fall back to best candidate at key point
                if !curr_key_point.candidates.is_empty() {
                    let best_candidate = &curr_key_point.candidates[0].segment;

                    // Only add if not creating a loop
                    if !used_segment_ids.contains(&best_candidate.id) {
                        route.push(best_candidate.clone());
                        used_segment_ids.insert(best_candidate.id);
                    }
                }

                continue;
            }

            // Add connecting segments to route
            for segment in next_segments {
                if !used_segment_ids.contains(&segment.id) {
                    route.push(segment.clone());
                    used_segment_ids.insert(segment.id);
                }
            }
        }

        // Verify no loops in final route
        let mut unique_ids = HashSet::new();
        for segment in &route {
            if !unique_ids.insert(segment.id) {
                warn!("Loop detected in final route, attempting to fix");
                return self.remove_loops(route);
            }
        }

        Ok(route)
    }

    /// Find best connection between segments with speed constraints
    fn find_best_connection(
        &self,
        current_segment: &RoadSegment,
        target_key_point: &KeyPoint,
        used_segment_ids: &HashSet<u64>,
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
        max_allowed_distance: f64,
    ) -> Result<Vec<RoadSegment>> {
        if target_key_point.candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Try each candidate at the target key point
        let mut best_path = Vec::new();
        let mut best_path_cost = f64::MAX;

        for candidate in &target_key_point.candidates {
            // Skip if would create a loop
            if used_segment_ids.contains(&candidate.segment.id) {
                continue;
            }

            // Find path to this candidate
            match self.find_path_with_distance_limit(
                current_segment.id,
                candidate.segment.id,
                graph,
                segment_map,
                used_segment_ids,
                max_allowed_distance,
            ) {
                Ok((path_cost, path)) => {
                    // Check if this is a better path
                    if path_cost < best_path_cost {
                        best_path = path;
                        best_path_cost = path_cost;
                    }
                }
                Err(_) => continue,
            }
        }

        if best_path.is_empty() {
            debug!("No valid path found to key point candidates");
        } else {
            debug!(
                "Found path to key point with cost {:.1} and {} segments",
                best_path_cost,
                best_path.len()
            );
        }

        Ok(best_path)
    }

    /// A* path finding with distance limit and loop avoidance
    fn find_path_with_distance_limit(
        &self,
        from: u64,
        to: u64,
        graph: &UnGraphMap<u64, f64>,
        segment_map: &HashMap<u64, RoadSegment>,
        used_segments: &HashSet<u64>,
        max_distance: f64,
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
        let mut total_distances = HashMap::new();

        // Get destination coordinates for heuristic
        let to_segment = segment_map
            .get(&to)
            .ok_or_else(|| anyhow!("Destination segment not found"))?;
        let goal_point = to_segment.centroid();

        // Initialize
        g_scores.insert(from, 0.0);
        total_distances.insert(from, 0.0);
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
                // Skip if already processed
                if closed_set.contains(&neighbor) {
                    continue;
                }

                // Skip if segment would create a loop (except for destination)
                if neighbor != to && used_segments.contains(&neighbor) {
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

                // Calculate new distance total
                let segment_length = neighbor_segment.length();
                let new_total_distance = total_distances[&current] + segment_length;

                // Skip if exceeds maximum allowed distance
                if new_total_distance > max_distance {
                    continue;
                }

                // Calculate edge cost
                let edge_cost = *graph.edge_weight(current, neighbor).unwrap_or(&1.0);
                let new_g_score = g_scores[&current] + edge_cost;

                // Only consider if better path
                if !g_scores.contains_key(&neighbor) || new_g_score < g_scores[&neighbor] {
                    // Update path
                    came_from.insert(neighbor, current);
                    g_scores.insert(neighbor, new_g_score);
                    total_distances.insert(neighbor, new_total_distance);

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
            "No path found between segments {} and {} within distance limit",
            from,
            to
        ))
    }

    /// Remove any loops in the route
    fn remove_loops(&self, route: Vec<RoadSegment>) -> Result<Vec<RoadSegment>> {
        let mut clean_route = Vec::new();
        let mut seen_ids = HashSet::new();

        for segment in route {
            if seen_ids.insert(segment.id) {
                clean_route.push(segment);
            } else {
                // Found a duplicate, remove all segments from first occurrence to here
                let first_index = clean_route
                    .iter()
                    .position(|s| s.id == segment.id)
                    .unwrap_or(0);

                warn!(
                    "Removing loop: segments #{} to #{} (segment ID: {})",
                    first_index,
                    clean_route.len() - 1,
                    segment.id
                );

                clean_route.truncate(first_index);

                // Re-add the current segment (it's now the first occurrence)
                clean_route.push(segment);

                // Update seen IDs
                seen_ids.clear();
                for seg in &clean_route {
                    seen_ids.insert(seg.id);
                }
            }
        }

        Ok(clean_route)
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
        // Always apply complexity penalty

        self.config.route_length_weight * length_score
            + self.config.heading_consistency_weight * heading_score
            + self.config.main_road_preference_weight * road_score
            + complexity_score
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
            }
        }

        Ok(())
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

/// GPS trace metrics for speed-aware route matching
struct GpsMetrics {
    avg_speed: f64,            // Average speed in m/s
    max_speed: f64,            // Maximum speed in m/s
    reasonable_max_speed: f64, // Reasonable maximum speed with buffer (m/s)
    min_travel_time: f64,      // Minimum reasonable travel time between points (seconds)
    gps_error: f64,            // Estimated GPS error radius (meters)
    total_distance: f64,       // Total GPS trace distance (meters)
    avg_point_distance: f64,   // Average distance between consecutive points (meters)
}

/// Key point with candidate segments and timestamps
struct KeyPoint {
    index: usize,
    point: Point<f64>,
    timestamp: DateTime<Utc>,
    candidates: Vec<SegmentCandidate>,
}

/// Structure to represent a candidate segment for a GPS point
#[derive(Clone)]
struct SegmentCandidate {
    segment: RoadSegment,
    distance: f64,          // Distance from GPS point to segment
    projection: Point<f64>, // Projected point on segment
    score: f64,             // Overall score (lower is better)
}

/// Structure to track which GPS points correspond to which route chunks
struct ChunkAssignment {
    chunk_idx: usize,
    gps_indices: Vec<usize>,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
}
