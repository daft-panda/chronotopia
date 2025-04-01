use anyhow::{Result, anyhow, bail};
use chrono::{DateTime, Utc};
use geo::{Closest, ClosestPoint, Haversine, LineString, algorithm::Distance};
use geo_types::Point;
use log::{debug, info, trace, warn};
use ordered_float::OrderedFloat;
use petgraph::prelude::UnGraphMap;
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::{HashMap, HashSet},
    path::Path,
};

use crate::{
    osm_preprocessing::{OsmProcessor, WaySegment},
    tile_loader::TileLoader,
};

#[derive(Debug, Clone)]
pub struct TileConfig {
    pub base_tile_size: f64,
    pub min_tile_density: usize, // Minimum roads per tile
    pub max_split_depth: u8,     // Prevent infinite splitting
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            base_tile_size: 0.1,
            min_tile_density: 1000,
            max_split_depth: 3,
        }
    }
}

/// Core map matching configuration
#[derive(Debug, Clone)]
pub struct MapMatcherConfig {
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
    /// Consider turn restrictions
    pub use_turn_restrictions: bool,
    /// Maximum number of candidates per GPS point
    pub max_candidates_per_point: usize,
    /// Weight for distance in scoring
    pub distance_weight: f64,
    /// Weight for heading difference in scoring
    pub heading_weight: f64,
    /// Weight for speed in scoring
    pub speed_weight: f64,
    /// Maximum number of tiles to load per depth level
    pub max_tiles_per_depth: usize,
    /// Detected loop penalty
    pub loop_penalty_weight: f64,
    /// Road continuity bonus
    pub continuity_bonus_weight: f64,
    /// Maximum allowed angle for a sharp turn (degrees)
    pub max_turn_angle: f64,
    /// Minimum speed to consider for a sharp turn (km/h)
    pub min_turn_speed: f64,
    /// Distance threshold for loop detection (meters)
    pub loop_distance_threshold: f64,
}

impl Default for MapMatcherConfig {
    fn default() -> Self {
        Self {
            osm_pbf_path: String::new(),
            tile_cache_dir: String::new(),
            tile_config: TileConfig::default(),
            max_cached_tiles: 100,
            max_matching_distance: 50.0,
            use_turn_restrictions: true,
            max_candidates_per_point: 10,
            distance_weight: 0.6,
            heading_weight: 0.3,
            speed_weight: 0.1,
            max_tiles_per_depth: 50,
            loop_penalty_weight: 50.0,
            continuity_bonus_weight: 20.0,
            max_turn_angle: 120.0, // Maximum angle for sharp turns without slowing down
            min_turn_speed: 5.0,   // Minimum speed to make sharp turns (km/h)
            loop_distance_threshold: 50.0, // Meters
        }
    }
}

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
    pub(crate) segments: Vec<WaySegment>,
    pub(crate) bridge: bool,
}

/// Structure to represent a candidate segment for a GPS point
#[derive(Clone)]
pub(crate) struct SegmentCandidate {
    segment: WaySegment,
    distance: f64,          // Distance from GPS point to segment
    projection: Point<f64>, // Projected point on segment
    score: f64,             // Overall score (lower is better)
}

pub struct RouteMatchJob {
    // inputs
    pub(crate) gps_points: Vec<Point<f64>>,
    pub(crate) timestamps: Vec<DateTime<Utc>>,
    pub(crate) debug_way_ids: Option<Vec<u64>>,
    // state
    pub(crate) graph: RefCell<Option<UnGraphMap<u64, f64>>>,
    pub(crate) segment_map: RefCell<HashMap<u64, WaySegment>>,
    pub(crate) loaded_tiles: RefCell<HashSet<String>>,
    // trackers
    pub(crate) all_candidates: RefCell<Vec<Vec<SegmentCandidate>>>,
    // debugging
    tracing: bool,
    pub(crate) window_trace: RefCell<Vec<WindowTrace>>,
    pub(crate) point_candidates_geojson: RefCell<Vec<Value>>,
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
            point_candidates_geojson: RefCell::new(Vec::new()),
        }
    }

    pub fn activate_tracing(&mut self) {
        self.tracing = true;
    }
}

/// Route-based map matcher implementation
pub struct RouteMatcher {
    config: RouteMatcherConfig,
    pub(crate) tile_loader: TileLoader,
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
    pub fn match_trace(&mut self, job: &mut RouteMatchJob) -> Result<Vec<WaySegment>> {
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
        let route = self.build_route_with_sliding_window(job)?;

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
        if job.tracing {
            for i in 0..job.gps_points.len() {
                let mut pc_geojson = job.point_candidates_geojson.borrow_mut();
                pc_geojson.push(self.debug_point_candidates(job, i)?);
            }
        }

        Ok(())
    }

    fn build_route_with_sliding_window(
        &mut self,
        job: &mut RouteMatchJob,
    ) -> Result<Vec<WaySegment>> {
        // Define window size - todo adapt based on GPS density
        let window_size = 7;
        // Process data in overlapping windows
        let step_size = window_size / 2; // More overlap for better correction

        info!(
            "Using window size {} with step size {}",
            window_size, step_size
        );

        let mut complete_route = Vec::new();
        let mut window_start = 0;
        let mut window_index: usize = 0;

        while window_start < job.gps_points.len() {
            let window_end = (window_start + window_size - 1).min(job.gps_points.len() - 1);

            info!(
                "Processing window {} from point {} to {} (of {})",
                window_index,
                window_start,
                window_end,
                job.gps_points.len() - 1
            );

            window_index += 1;

            // Find optimal path for this window
            let window_route = self.find_a_star_path_for_window(
                job,
                window_start,
                window_end,
                if complete_route.is_empty() {
                    None
                } else {
                    complete_route.last()
                },
            )?;

            if window_route.is_empty() {
                // Try smaller window if no route found
                window_start += 1;
                continue;
            }

            // If there's overlap with previous windows,
            // consider whether to replace the previous route
            if !complete_route.is_empty() && window_start > 0 {
                // Find the overlap point - the first GPS point in this window
                let overlap_gps_idx = window_start;

                // Find which segment in the complete route corresponds to this overlap point
                let overlap_segment_idx = self
                    .find_segment_for_gps_point(&complete_route, job.gps_points[overlap_gps_idx]);

                if let Some(idx) = overlap_segment_idx {
                    // We found where current window overlaps previous windows
                    if job.tracing {
                        let window_traces = job.window_trace.get_mut();
                        window_traces.push(WindowTrace {
                            start: window_start,
                            end: window_end,
                            segments: window_route.clone(),
                            bridge: false,
                        });
                    }

                    // Replace from overlap point onwards with new route
                    complete_route.truncate(idx);
                    complete_route.extend(window_route);

                    // Move window start ahead accounting for the replacement
                    window_start += step_size;
                    continue;
                }
            }

            // If no replacement happened, just add non-overlapping segments
            let new_segments = self.new_segments_for_route(&complete_route, window_route);

            if job.tracing {
                let window_traces = job.window_trace.get_mut();
                window_traces.push(WindowTrace {
                    start: window_start,
                    end: window_end,
                    segments: new_segments.clone(),
                    bridge: false,
                });
            }

            complete_route.extend(new_segments);

            // Move window forward
            window_start += step_size;
        }

        if complete_route.is_empty() {
            return Err(anyhow!("Failed to build valid route"));
        }

        Ok(complete_route)
    }

    /// Find which segment in a route matches a GPS point
    fn find_segment_for_gps_point(&self, route: &[WaySegment], point: Point<f64>) -> Option<usize> {
        if route.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_distance = f64::MAX;

        for (i, segment) in route.iter().enumerate() {
            // Project point to segment
            let projection = self.project_point_to_segment(point, segment);
            let distance = Haversine.distance(point, projection);

            if distance < best_distance {
                best_distance = distance;
                best_idx = i;
            }
        }

        // Only return if within reasonable distance
        if best_distance <= 150.0 {
            Some(best_idx)
        } else {
            None
        }
    }

    // if job.tracing {
    //                         let window_traces = job.window_trace.get_mut();
    //                         window_traces.push(WindowTrace {
    //                             start: window_start,
    //                             end: window_end,
    //                             segments: new_segments.clone(),
    //                             bridge: false,
    //                         });
    //                     }
    fn find_a_star_path_for_window(
        &self,
        job: &RouteMatchJob,
        start_idx: usize,
        end_idx: usize,
        previous_segment: Option<&WaySegment>,
    ) -> Result<Vec<WaySegment>> {
        // Get window data
        let window_points = &job.gps_points[start_idx..=end_idx];
        let window_timestamps = &job.timestamps[start_idx..=end_idx];
        let window_candidates = &job.all_candidates.borrow()[start_idx..=end_idx];

        // Validation
        if window_points.is_empty() || window_candidates.is_empty() {
            return Ok(Vec::new());
        }

        let first_candidates = &window_candidates[0];
        let last_candidates = &window_candidates[window_candidates.len() - 1];

        if first_candidates.is_empty() || last_candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Track best route
        let mut best_route = Vec::new();
        let mut best_score = f64::MAX;

        // Consider ALL candidates equally - don't prioritize connectivity
        for first_candidate in first_candidates {
            for last_candidate in last_candidates {
                // Skip if same segment for multi-point windows
                if first_candidate.segment.id == last_candidate.segment.id
                    && window_points.len() > 1
                {
                    continue;
                }

                // Find path between candidates using simplified metrics
                match self.find_simple_path(
                    job,
                    &first_candidate.segment,
                    &last_candidate.segment,
                    Self::max_route_length_from_points(window_points, window_timestamps),
                ) {
                    Ok((path, score)) => {
                        // Apply road class factor to score
                        let road_class_factor = self.evaluate_path_road_classes(&path);
                        let adjusted_score = score * road_class_factor;

                        if adjusted_score < best_score {
                            best_route = path;
                            best_score = adjusted_score;
                        }
                    }
                    Err(_) => continue,
                }
            }
        }

        Ok(best_route)
    }

    /// Find path between segments with just distance and road class
    fn find_simple_path(
        &self,
        job: &RouteMatchJob,
        from_segment: &WaySegment,
        to_segment: &WaySegment,
        max_distance: f64,
    ) -> Result<(Vec<WaySegment>, f64)> {
        // If same segment, trivial case
        if from_segment.id == to_segment.id {
            return Ok((vec![from_segment.clone()], 0.0));
        }

        // Use A* to find the shortest path
        match self.find_path_with_distance_limit(
            job,
            from_segment.id,
            to_segment.id,
            &HashSet::new(),
            max_distance,
        ) {
            Ok((path_cost, path)) => Ok((path, path_cost)),
            Err(e) => Err(e),
        }
    }

    /// Helper to calculate max route length from points
    fn max_route_length_from_points(points: &[Point<f64>], timestamps: &[DateTime<Utc>]) -> f64 {
        // Distance-based approach
        if points.len() >= 2 {
            let direct_distance =
                Haversine.distance(*points.first().unwrap(), *points.last().unwrap());

            // Time-based approach if timestamps available
            if timestamps.len() >= 2 {
                let time_diff = (*timestamps.last().unwrap() - timestamps.first().unwrap())
                    .num_seconds() as f64;
                if time_diff > 0.0 {
                    // Using high average speed (40 m/s = 144 km/h)
                    let time_based_limit = time_diff * 40.0;
                    return time_based_limit.min(direct_distance * 5.0).max(1000.0);
                }
            }

            // Fallback to just distance-based
            return direct_distance * 5.0f64.max(1000.0);
        }

        // Default if we can't calculate
        5000.0
    }

    /// Calculate how well a path covers GPS points
    fn calculate_path_coverage(&self, path: &[WaySegment], points: &[Point<f64>]) -> f64 {
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

    /// Return new segments for route, avoiding duplicates
    fn new_segments_for_route(
        &self,
        route: &Vec<WaySegment>,
        matched_segments: Vec<WaySegment>,
    ) -> Vec<WaySegment> {
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

    /// A* path finding with distance limit and loop avoidance
    /// Enhanced to better handle split segments
    fn find_path_with_distance_limit(
        &self,
        job: &RouteMatchJob,
        from: u64,
        to: u64,
        used_segments: &HashSet<u64>,
        max_distance: f64,
    ) -> Result<(f64, Vec<WaySegment>)> {
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

        // Create a mapping from original IDs to split segment IDs
        // This helps us recognize segments that came from the same original segment
        let mut original_id_map: HashMap<u64, Vec<u64>> = HashMap::new();
        for (&id, segment) in segment_map.iter() {
            // Use the segment's tracked original_id if available
            let original_id = if let Some(orig_id) = segment.original_id {
                orig_id
            } else {
                id
            };

            original_id_map.entry(original_id).or_default().push(id);
        }

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

                // Get neighbor segment
                let neighbor_segment = match segment_map.get(&neighbor) {
                    Some(seg) => seg,
                    None => continue,
                };

                // For split segments, check if we've used any segments with the same original ID
                // But allow crossing split segments if they're at different nodes
                if neighbor != to {
                    let orig_id = neighbor_segment.original_id.unwrap_or(neighbor);

                    // Check if we'd create a loop by using segments from the same original segment
                    let would_create_loop = original_id_map
                        .get(&orig_id)
                        .map(|ids| {
                            ids.iter().any(|&id| {
                                if used_segments.contains(&id) {
                                    // Check if they're connected at a different node than the current connection
                                    if let (Some(current_seg), Some(used_seg)) =
                                        (segment_map.get(&current), segment_map.get(&id))
                                    {
                                        // Find common nodes
                                        let current_split = current_seg.split_id;
                                        let used_split = used_seg.split_id;

                                        // If they have the same split node, they'd create a loop
                                        current_split.is_some() && current_split == used_split
                                    } else {
                                        true
                                    }
                                } else {
                                    false
                                }
                            })
                        })
                        .unwrap_or(false);

                    if would_create_loop {
                        continue;
                    }
                }

                // Check if it exists in the graph
                if !graph.as_ref().unwrap().contains_edge(current, neighbor) {
                    continue;
                }

                // Calculate distance
                let segment_length = neighbor_segment.length();
                let new_total_distance = total_distances[&current] + segment_length;

                // Skip if exceeds max distance
                if new_total_distance > max_distance {
                    continue;
                }

                // Use road-class aware edge cost
                let edge_cost =
                    self.calculate_edge_cost(current_segment, neighbor_segment, segment_length);

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

    /// Evaluate overall quality of path based on road classes
    fn evaluate_path_road_classes(&self, path: &[WaySegment]) -> f64 {
        if path.is_empty() {
            return 1.0;
        }

        let mut motorway_count = 0;
        let mut trunk_count = 0;
        let mut primary_count = 0;
        let mut other_count = 0;

        for segment in path {
            match segment.highway_type.as_str() {
                "motorway" | "motorway_link" => motorway_count += 1,
                "trunk" | "trunk_link" => trunk_count += 1,
                "primary" | "primary_link" => primary_count += 1,
                _ => other_count += 1,
            }
        }

        let total = path.len() as f64;

        // Calculate quality score (higher = better)
        let quality_score =
            (motorway_count as f64 * 3.0 + trunk_count as f64 * 2.0 + primary_count as f64 * 1.0)
                / total;

        // Convert to factor (lower = better for scoring)
        1.0 / (quality_score + 0.5)
    }

    /// Build road network graph for path finding
    fn build_road_network(
        &mut self,
        loaded_tiles: &HashSet<String>,
    ) -> Result<(UnGraphMap<u64, f64>, HashMap<u64, WaySegment>)> {
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
    fn project_point_to_segment(&self, point: Point<f64>, segment: &WaySegment) -> Point<f64> {
        let line = LineString::from(segment.coordinates.clone());
        match line.closest_point(&point) {
            Closest::SinglePoint(projected) => projected,
            _ => point, // Fallback, should not happen with valid segments
        }
    }

    /// Debug function to track why specific way IDs were not chosen
    fn debug_way_ids(
        &mut self,
        final_route: &[WaySegment],
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
            let mut way_segment: Option<WaySegment> = None;

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

    fn debug_point_candidates(&self, job: &RouteMatchJob, point_idx: usize) -> Result<Value> {
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

            features.push(json!({
            "type": "Feature",
            "properties": {
                "type": "candidate_segment",
                "segment_id": segment.id,
                "rank": i,
                "score": candidate.score,
                "distance": candidate.distance,
                "highway_type": segment.highway_type,
                "description": format!("Segment ID: {}, Rank: {}, Score: {:.2}, Distance: {:.2}m", 
                                       segment.id, i, candidate.score, candidate.distance)
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
                    "type": "projection",
                    "segment_id": segment.id,
                    "description": format!("Projection to segment {}", segment.id)
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [candidate.projection.x(), candidate.projection.y()]
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

    /// Apply road class weight to edge cost in A* search
    fn calculate_edge_cost(
        &self,
        from_segment: &WaySegment,
        to_segment: &WaySegment,
        distance: f64,
    ) -> f64 {
        // Base cost is the distance
        let base_cost = distance;

        // Apply road class factor
        let road_class_factor = match to_segment.highway_type.as_str() {
            "motorway" => 0.5, // Strongly prefer motorways
            "motorway_link" => 0.6,
            "trunk" => 0.7,
            "trunk_link" => 0.8,
            "primary" => 0.9,
            "primary_link" => 1.0,
            "secondary" => 1.2,
            "secondary_link" => 1.3,
            "tertiary" => 1.5,
            "tertiary_link" => 1.6,
            "residential" => 2.0, // Penalize smaller roads
            "unclassified" => 2.5,
            "service" => 3.0,
            _ => 3.5,
        };

        base_cost * road_class_factor
    }

    /// Utility method to check connectivity between a list of way IDs
    /// Returns a list of connectivity issues found
    pub fn check_way_connectivity(&mut self, way_ids: &[u64]) -> Result<Vec<String>> {
        if way_ids.len() < 2 {
            return Ok(Vec::new()); // Nothing to check with fewer than 2 segments
        }

        let mut issues = Vec::new();

        // Now check connectivity between consecutive way IDs
        for i in 0..way_ids.len() - 1 {
            let current_id = way_ids[i];
            let next_id = way_ids[i + 1];

            // Skip if they're the same segment
            if current_id == next_id {
                continue;
            }

            let current = self.tile_loader.get_segment(current_id).unwrap();
            let next = self.tile_loader.get_segment(next_id).unwrap();

            // Check if directly connected in OSM graph
            let directly_connected =
                current.connections.contains(&next_id) || next.connections.contains(&current_id);

            if !directly_connected {
                // Not directly connected, check distances between endpoints
                let current_end = current.coordinates.last().unwrap();
                let next_start = next.coordinates.first().unwrap();

                let distance = Haversine.distance(
                    Point::new(current_end.x, current_end.y),
                    Point::new(next_start.x, next_start.y),
                );

                issues.push(format!(
                    "Way IDs {} and {} at positions {}-{} are not directly connected (distance: {:.2}m). No connecting segments found.",
                    current_id,
                    next_id,
                    i,
                    i + 1,
                    distance
                ));

                // Check if segments are in reverse order
                if next.connections.contains(&current_id) && !current.connections.contains(&next_id)
                {
                    issues.push(format!(
                        "Way IDs may be in reverse order: {} connects to {} but not vice versa",
                        next_id, current_id
                    ));
                }

                // Check road classes to see if it's a routing preference issue
                issues.push(format!(
                    "Road classes: {} is '{}', {} is '{}'",
                    current_id, current.highway_type, next_id, next.highway_type
                ));
            }
        }

        // Check for potential loops
        let unique_ids: HashSet<u64> = way_ids.iter().copied().collect();
        if unique_ids.len() < way_ids.len() {
            // There are duplicates - find them
            let mut seen = HashSet::new();
            let mut duplicates = Vec::new();

            for (i, &id) in way_ids.iter().enumerate() {
                if !seen.insert(id) {
                    duplicates.push((id, i));
                }
            }

            for (id, pos) in duplicates {
                issues.push(format!(
                "Way ID {} appears multiple times (last at position {}), indicating a potential loop",
                id, pos
            ));
            }
        }

        Ok(issues)
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

fn calculate_transition_cost(from_seg: &WaySegment, to_seg: &WaySegment) -> f64 {
    // Calculate geographic distance
    let from_end = from_seg.coordinates.last().unwrap();
    let to_start = to_seg.coordinates.first().unwrap();

    let from_point = Point::new(from_end.x, from_end.y);
    let to_point = Point::new(to_start.x, to_start.y);

    let distance = Haversine.distance(from_point, to_point);

    // Apply road type factor
    let type_factor = match to_seg.highway_type.as_str() {
        "motorway" => 0.5, // Strongly prefer motorways
        "motorway_link" => 0.6,
        "trunk" => 0.7,
        "trunk_link" => 0.8,
        "primary" => 0.9,
        "primary_link" => 1.0,
        "secondary" => 1.2,
        "secondary_link" => 1.3,
        "tertiary" => 1.5,
        "tertiary_link" => 1.6,
        "residential" => 2.0,
        "unclassified" => 2.5,
        "service" => 3.0,
        _ => 3.5,
    };

    // Final cost is distance with road type factor
    (distance / 100.0) * type_factor
}

pub fn calculate_heading(from: Point<f64>, to: Point<f64>) -> f64 {
    let dx = to.x() - from.x();
    let dy = to.y() - from.y();
    dy.atan2(dx).to_degrees()
}
