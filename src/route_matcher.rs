use anyhow::{Result, anyhow, bail};
use chrono::{DateTime, Utc};
use geo::{Closest, ClosestPoint, Haversine, LineString, algorithm::Distance};
use geo_types::Point;
use log::{debug, info, trace, warn};
use ordered_float::OrderedFloat;
use petgraph::prelude::UnGraphMap;
use std::{
    cell::RefCell,
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
    /// Maximum number of tiles to load per depth level
    pub max_tiles_per_depth: usize,
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
            max_tiles_per_depth: 50,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub(crate) struct WindowTrace {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) segments: Vec<RoadSegment>,
    pub(crate) bridge: bool,
}

pub struct RouteMatchJob {
    // inputs
    pub(crate) gps_points: Vec<Point<f64>>,
    pub(crate) timestamps: Vec<DateTime<Utc>>,
    debug_way_ids: Option<Vec<u64>>,
    // state
    graph: RefCell<Option<UnGraphMap<u64, f64>>>,
    segment_map: RefCell<HashMap<u64, RoadSegment>>,
    loaded_tiles: RefCell<HashSet<String>>,
    // trackers
    all_candidates: RefCell<Vec<Vec<SegmentCandidate>>>,
    // misc
    tracing: bool,
    pub(crate) window_trace: RefCell<Vec<WindowTrace>>,
}

impl RouteMatchJob {
    pub fn new(
        gps_points: Vec<Point<f64>>,
        timestamps: Vec<DateTime<Utc>>,
        debug_way_ids: Option<Vec<u64>>,
    ) -> Self {
        Self {
            all_candidates: RefCell::new(Vec::with_capacity(gps_points.len())),
            gps_points,
            timestamps,
            debug_way_ids,
            tracing: false,
            graph: RefCell::new(None),
            segment_map: RefCell::new(HashMap::new()),
            loaded_tiles: RefCell::new(HashSet::new()),
            window_trace: RefCell::new(Vec::new()),
        }
    }

    pub fn activate_tracing(&mut self) {
        self.tracing = true;
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
    pub fn match_trace(&mut self, job: &mut RouteMatchJob) -> Result<Vec<RoadSegment>> {
        if job.gps_points.len() < 2 || job.gps_points.len() != job.timestamps.len() {
            return Err(anyhow!("Invalid GPS points or timestamps"));
        }

        info!(
            "Starting sliding window route matching for {} points",
            job.gps_points.len()
        );
        let start_time = std::time::Instant::now();

        // Step 1: Load tiles covering the entire route area with extra buffer for GPS inaccuracy
        let trace_bbox = self.calculate_trace_bbox(&job.gps_points);
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

        job.loaded_tiles.replace(loaded_tiles.clone());
        job.graph.replace(Some(graph));
        job.segment_map.replace(segment_map);

        // Step 3: For each GPS point, find all candidate segments within accuracy range
        info!("Finding candidate segments for all GPS points");
        self.find_all_candidate_segments(&job, loaded_tiles.clone())?;

        // Step 4: Build route using sliding window approach
        info!("Building route using sliding window approach");
        let route = self.build_route_with_sliding_window(&job)?;

        info!(
            "Map matching completed in {:.2?} with {} segments",
            start_time.elapsed(),
            route.len()
        );

        // Debug information about specified way IDs
        if let Some(way_ids) = &job.debug_way_ids {
            self.debug_way_ids(&route, &way_ids, &loaded_tiles)?;
        }

        Ok(route)
    }

    /// Find all candidate segments within accuracy range for each GPS point
    fn find_all_candidate_segments(
        &mut self,
        job: &RouteMatchJob,
        loaded_tiles: HashSet<String>,
    ) -> Result<()> {
        const GPS_ACCURACY_METER: f64 = 75.0;
        let max_distance = GPS_ACCURACY_METER * 1.5; // Allow up to 1.5x the accuracy

        let mut all_candidates = Vec::with_capacity(job.gps_points.len());

        for (i, &point) in job.gps_points.iter().enumerate() {
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
                        let score = distance / GPS_ACCURACY_METER;

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
                            let score = distance / GPS_ACCURACY_METER * 1.5;

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
            if candidates.len() > self.config.max_candidates_per_point {
                candidates.truncate(self.config.max_candidates_per_point);
            }

            debug!("Found {} candidates for GPS point {}", candidates.len(), i);
            all_candidates.push(candidates);
        }

        job.all_candidates.replace(all_candidates);
        Ok(())
    }

    /// Build route using sliding window approach
    fn build_route_with_sliding_window(&mut self, job: &RouteMatchJob) -> Result<Vec<RoadSegment>> {
        // Define window size - adapt based on GPS density
        let window_size = if job.gps_points.len() < 10 {
            // For very few points, use larger window
            job.gps_points.len().min(7)
        } else if job.gps_points.len() < 20 {
            // For moderate number of points
            5
        } else {
            // For many points, use smaller window
            4
        };

        debug!("Using sliding window of size {}", window_size);

        // Special case: if we have fewer points than window size, just do one window
        if job.gps_points.len() <= window_size {
            return self.match_single_window(job, 0, job.gps_points.len() - 1);
        }

        // Process data in overlapping windows
        let mut complete_route = Vec::new();
        let mut last_end_segment: Option<RoadSegment> = None;

        // Step through the data with overlapping windows
        let step_size = window_size / 4; // 50% overlap between windows
        let mut window_start = 0;

        let mut window_tracing = if job.tracing {
            Some(job.window_trace.borrow_mut())
        } else {
            None
        };

        while window_start < job.gps_points.len() {
            let window_end = (window_start + window_size - 1).min(job.gps_points.len() - 1);

            trace!(
                "Processing window from point {} to {} (of {})",
                window_start,
                window_end,
                job.gps_points.len() - 1
            );

            // Match this window
            let window_route = self.match_window_with_context(
                job,
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
                        job,
                        window_start,
                        smaller_end,
                        last_end_segment.as_ref(),
                    )?;

                    if !smaller_route.is_empty() {
                        // Add to complete route
                        let new_segments =
                            self.new_segments_for_route(&mut complete_route, smaller_route);
                        if let Some(ref mut window_traces) = window_tracing {
                            window_traces.push(WindowTrace {
                                start: window_start,
                                end: window_end,
                                segments: new_segments.clone(),
                                ..Default::default()
                            });
                        }
                        complete_route.extend(new_segments);

                        // Update last segment
                        last_end_segment = complete_route.last().cloned();

                        // Adjust window for next iteration
                        window_start = smaller_end;
                        continue;
                    }
                }

                // If we can't find any route, we need to bridge the gap directly
                if let Some(last_segment) = &last_end_segment {
                    let next_valid_point = self.find_next_valid_point(job, window_start);

                    if let Some((next_idx, next_candidate)) = next_valid_point {
                        debug!(
                            "Bridging gap from point {} to {} with direct connection",
                            window_start, next_idx
                        );

                        // Try to connect directly
                        let bridge = self.bridge_gap(job, last_segment, &next_candidate.segment)?;

                        if !bridge.is_empty() {
                            // Add bridge to route
                            let new_segments =
                                self.new_segments_for_route(&mut complete_route, bridge);
                            if let Some(ref mut window_traces) = window_tracing {
                                window_traces.push(WindowTrace {
                                    start: window_start,
                                    end: window_end,
                                    segments: new_segments.clone(),
                                    bridge: true,
                                });
                            }
                            complete_route.extend(new_segments);

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
            let new_segments = self.new_segments_for_route(&complete_route, window_route);
            if let Some(ref mut window_traces) = window_tracing {
                window_traces.push(WindowTrace {
                    start: window_start,
                    end: window_end,
                    segments: new_segments.clone(),
                    ..Default::default()
                });
            }
            complete_route.extend(new_segments);

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
        job: &RouteMatchJob,
        start_idx: usize,
        end_idx: usize,
        previous_segment: Option<&RoadSegment>,
    ) -> Result<Vec<RoadSegment>> {
        if start_idx > end_idx || end_idx >= job.gps_points.len() {
            return Err(anyhow!("Invalid window indices"));
        }

        // If there are no candidates for any point, return empty
        for i in start_idx..=end_idx {
            if job.all_candidates.borrow()[i].is_empty() {
                debug!("No candidates for point {} in window", i);
                return Ok(Vec::new());
            }
        }

        // If we have context from previous window, constrain the first point's candidates
        let mut all_candidates_in_window =
            job.all_candidates.borrow()[start_idx..=end_idx].to_vec();

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
        let route = self.find_best_route_in_window(job, start_idx, end_idx)?;

        Ok(route)
    }

    /// Match a single window of GPS points when we have too few points for sliding
    fn match_single_window(
        &mut self,
        job: &RouteMatchJob,
        start_idx: usize,
        end_idx: usize,
    ) -> Result<Vec<RoadSegment>> {
        // Find best route
        let route = self.find_best_route_in_window(job, start_idx, end_idx)?;
        Ok(route)
    }

    /// Find the best route through a window of GPS points
    fn find_best_route_in_window(
        &self,
        job: &RouteMatchJob,
        start_idx: usize,
        end_idx: usize,
    ) -> Result<Vec<RoadSegment>> {
        let window_points = &job.gps_points[start_idx..=end_idx];
        let window_timestamps = &job.timestamps[start_idx..=end_idx];
        let window_candidates = &job.all_candidates.borrow()[start_idx..=end_idx];

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
                    job,
                    &first_candidate.segment,
                    &last_candidate.segment,
                    window_points,
                    window_timestamps,
                    window_candidates,
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
        job: &RouteMatchJob,
        from_segment: &RoadSegment,
        to_segment: &RoadSegment,
        window_points: &[Point<f64>],
        window_timestamps: &[DateTime<Utc>],
        window_candidates: &[Vec<SegmentCandidate>],
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
        const REASONABLE_SPEED_METER_SECOND: f64 = 33.3;

        let max_route_length = if time_diff > 0.0 {
            let speed_based = REASONABLE_SPEED_METER_SECOND * time_diff;
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
            job,
            from_segment.id,
            to_segment.id,
            &waypoints,
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
                    job,
                    from_segment.id,
                    to_segment.id,
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
                path_coordinates.push(*coord);
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
        job: &RouteMatchJob,
        from_id: u64,
        to_id: u64,
        waypoint_ids: &HashSet<u64>,
        max_distance: f64,
    ) -> Result<(Vec<RoadSegment>, f64)> {
        if waypoint_ids.is_empty() {
            // No waypoints, just find direct path
            let result = self.find_path_with_distance_limit(
                job,
                from_id,
                to_id,
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
                job,
                from_id,
                waypoint_id,
                &HashSet::new(),
                max_distance / 2.0,
            ) {
                Ok((cost, path)) => (cost, path),
                Err(_) => continue,
            };

            // Find path from waypoint to end
            let waypoint_to_end = match self.find_path_with_distance_limit(
                job,
                waypoint_id,
                to_id,
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
        let result =
            self.find_path_with_distance_limit(job, from_id, to_id, &HashSet::new(), max_distance)?;
        Ok((result.1, result.0)) // Swap to match expected return type
    }

    /// Find the next point with valid candidates after a given index
    fn find_next_valid_point(
        &self,
        job: &RouteMatchJob,
        start_idx: usize,
    ) -> Option<(usize, SegmentCandidate)> {
        for i in start_idx + 1..job.gps_points.len() {
            if !job.all_candidates.borrow()[i].is_empty() {
                return Some((i, job.all_candidates.borrow()[i][0].clone()));
            }
        }

        None
    }

    /// Return new segments for route, avoiding duplicates
    fn new_segments_for_route(
        &self,
        route: &Vec<RoadSegment>,
        matched_segments: Vec<RoadSegment>,
    ) -> Vec<RoadSegment> {
        // If route is empty, just return all segments
        if route.is_empty() {
            return matched_segments;
        }

        // Get last segment in route
        let last_segment_id = route.last().unwrap().id;

        // Skip first segment of new segments if it's the same as the last segment of route
        let start_idx = if !matched_segments.is_empty() && matched_segments[0].id == last_segment_id
        {
            1
        } else {
            0
        };

        let mut added_segments = vec![];

        // Return remaining segments
        for segment in matched_segments.into_iter().skip(start_idx) {
            added_segments.push(segment);
        }

        added_segments
    }

    /// Bridge a gap between two unconnected segments
    fn bridge_gap(
        &self,
        job: &RouteMatchJob,
        from_segment: &RoadSegment,
        to_segment: &RoadSegment,
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
            job,
            from_segment.id,
            to_segment.id,
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

    /// A* path finding with distance limit and loop avoidance
    fn find_path_with_distance_limit(
        &self,
        job: &RouteMatchJob,
        from: u64,
        to: u64,
        used_segments: &HashSet<u64>,
        max_distance: f64,
    ) -> Result<(f64, Vec<RoadSegment>)> {
        if from == to {
            if let Some(segment) = job.segment_map.borrow().get(&from) {
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
        let segment_map = job.segment_map.borrow();
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

            let graph = job.graph.borrow();

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
                if !graph.as_ref().unwrap().contains_edge(current, neighbor) {
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
                let edge_cost = *graph
                    .as_ref()
                    .unwrap()
                    .edge_weight(current, neighbor)
                    .unwrap_or(&1.0);
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

    // Combine factors
    (distance / 100.0) * turn_penalty * type_penalty
}

fn angle_difference(a: f64, b: f64) -> f64 {
    let diff = ((a - b) % 360.0).abs();
    diff.min(360.0 - diff)
}

/// Structure to represent a candidate segment for a GPS point
#[derive(Clone)]
struct SegmentCandidate {
    segment: RoadSegment,
    distance: f64,          // Distance from GPS point to segment
    projection: Point<f64>, // Projected point on segment
    score: f64,             // Overall score (lower is better)
}
