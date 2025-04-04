use anyhow::{Result, anyhow, bail};
use chrono::{DateTime, Utc};
use geo::{Closest, ClosestPoint, Haversine, Intersects, LineString, algorithm::Distance};
use geo_types::Point;
use log::{debug, info};
use ordered_float::OrderedFloat;
use petgraph::prelude::UnGraphMap;
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

use crate::{
    osm_preprocessing::{OSMProcessor, TileIndex, WaySegment, are_road_types_compatible},
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
    /// User-provided expected way sequence for debugging
    pub(crate) expected_way_sequence: Option<RouteExpectation>,

    /// Explanation for why expected path was not matched
    pub(crate) matching_explanation: RefCell<Vec<String>>,
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
            expected_way_sequence: None,
            matching_explanation: RefCell::new(Vec::new()),
        }
    }

    pub fn activate_tracing(&mut self) {
        self.tracing = true;
    }

    /// Set an expected sequence of OSM way IDs that should be matched between two points
    pub fn expect_way_sequence(&mut self, start_idx: usize, end_idx: usize, way_ids: Vec<u64>) {
        self.expected_way_sequence = Some(RouteExpectation {
            way_ids,
            start_point_idx: start_idx,
            end_point_idx: end_idx,
            active: true,
            is_subsequence: false,
            result: None,
        });

        // Initialize explanation vector
        self.matching_explanation.replace(Vec::new());
    }

    pub fn expect_subsequence(&mut self, start_idx: usize, end_idx: usize, way_ids: Vec<u64>) {
        self.expected_way_sequence = Some(RouteExpectation {
            way_ids,
            start_point_idx: start_idx,
            end_point_idx: end_idx,
            active: true,
            is_subsequence: true,
            result: None,
        });

        // Initialize explanation vector
        self.matching_explanation.replace(Vec::new());
    }

    /// Check if a specific way ID was considered as a candidate for a given GPS point
    pub fn is_way_id_candidate(&self, point_idx: usize, way_id: u64) -> bool {
        if point_idx >= self.all_candidates.borrow().len() {
            return false;
        }

        let candidates = &self.all_candidates.borrow()[point_idx];

        for candidate in candidates {
            if candidate.segment.osm_way_id == way_id {
                return true;
            }
        }

        false
    }

    /// Get candidate rank for a specific way ID at a GPS point (if it exists)
    pub fn get_way_id_candidate_rank(&self, point_idx: usize, way_id: u64) -> Option<(usize, f64)> {
        if point_idx >= self.all_candidates.borrow().len() {
            return None;
        }

        let candidates = &self.all_candidates.borrow()[point_idx];

        for (rank, candidate) in candidates.iter().enumerate() {
            if candidate.segment.osm_way_id == way_id {
                return Some((rank, candidate.score));
            }
        }

        None
    }

    /// Add an explanation for why the expected path wasn't matched
    pub fn add_explanation(&self, explanation: String) {
        self.matching_explanation.borrow_mut().push(explanation);
    }

    /// Get the full explanation report for why the expected path wasn't matched
    pub fn get_matching_explanation(&self) -> String {
        if self.matching_explanation.borrow().is_empty() {
            return "No explanation available.".to_string();
        }

        let mut explanation = "Reasons why expected way sequence was not matched:\n".to_string();

        for reason in self.matching_explanation.borrow().iter() {
            explanation.push_str(&format!("{}\n", reason));
        }

        explanation
    }
}

/// Struct to track expected routing and results
#[derive(Clone, Debug)]
pub struct RouteExpectation {
    /// The expected sequence of way IDs
    pub way_ids: Vec<u64>,
    /// Start point index in the original gps_points array
    pub start_point_idx: usize,
    /// End point index in the original gps_points array
    pub end_point_idx: usize,
    /// Whether this sequence should be checked
    pub active: bool,
    /// Whether this sequence should be treated as a subsequence
    pub is_subsequence: bool,
    /// Result of the debugging
    pub result: Option<String>,
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
        // Create temporary directory for processing
        let temp_dir = "sample/temp";

        info!("Starting OSM data preprocessing");
        std::fs::create_dir_all(&self.config.tile_cache_dir)?;
        let processor = OSMProcessor::new(self.config.tile_config.clone())
            .with_chunk_size(1_000_000) // Process 1M elements per chunk
            .with_batch_sizes(5_000_000, 500_000) // 5M nodes, 500K ways at once
            .with_temp_dir(PathBuf::from(temp_dir));

        processor.process_pbf(&self.config.osm_pbf_path, &self.config.tile_cache_dir)?;

        // Clean up temporary directory
        std::fs::remove_dir_all(temp_dir)?;

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
        self.find_all_candidate_segments(job, loaded_tiles.clone())?;

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
            self.debug_way_ids(&route, way_ids, &loaded_tiles)?;
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

        // Track which segments were chosen for each GPS point in previous windows
        let mut previous_point_segments: HashMap<usize, u64> = HashMap::new();

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

            // Create a list of previously matched point constraints for this window
            let overlapping_points: Vec<(usize, u64)> = (window_start..=window_end)
                .filter_map(|idx| {
                    previous_point_segments
                        .get(&idx)
                        .map(|&seg_id| (idx, seg_id))
                })
                .collect();

            // Find optimal path for this window, passing previous point constraints
            let window_route = self.find_a_star_path_for_window(
                job,
                window_start,
                window_end,
                if complete_route.is_empty() {
                    None
                } else {
                    complete_route.last()
                },
                &overlapping_points,
            )?;

            if window_route.is_empty() {
                // Try to find a path without constraints if no valid path with constraints
                info!("No valid path found with constraints. Attempting unconstrained matching...");
                let unconstrained_route = self.find_a_star_path_for_window(
                    job,
                    window_start,
                    window_end,
                    if complete_route.is_empty() {
                        None
                    } else {
                        complete_route.last()
                    },
                    &[],
                )?;

                if unconstrained_route.is_empty() {
                    // Still no valid route, try a smaller window
                    window_start += 1;
                    continue;
                }

                // Use unconstrained route when no constrained route is possible
                if let Some(exp) = &job.expected_way_sequence {
                    if exp.active {
                        self.analyze_window_match(
                            job,
                            window_start,
                            window_end,
                            &unconstrained_route,
                            window_index,
                        );
                        job.add_explanation(format!(
                            "Window {} had to ignore previous constraints to produce a valid route",
                            window_index
                        ));
                    }
                }

                // Update complete route with unconstrained route
                if !complete_route.is_empty() && window_start > 0 {
                    // Find the overlap point - the first GPS point in this window
                    let overlap_gps_idx = window_start;

                    // Find which segment in the complete route corresponds to this overlap point
                    let overlap_segment_idx = self.find_segment_for_gps_point(
                        &complete_route,
                        job.gps_points[overlap_gps_idx],
                    );

                    if let Some(idx) = overlap_segment_idx {
                        // We found where current window overlaps previous windows
                        if job.tracing {
                            let window_traces = job.window_trace.get_mut();
                            window_traces.push(WindowTrace {
                                start: window_start,
                                end: window_end,
                                segments: unconstrained_route.clone(),
                                bridge: true, // Mark as unconstrained bridge
                            });
                        }

                        // Replace from overlap point onwards with new route
                        complete_route.truncate(idx);
                        complete_route.extend(unconstrained_route.clone());

                        // Update the point-to-segment mapping for the new segments
                        self.update_point_segment_mapping(
                            &mut previous_point_segments,
                            window_start,
                            window_end,
                            &unconstrained_route,
                            &job.gps_points,
                        );

                        // Move window start ahead accounting for the replacement
                        window_start += step_size;
                        continue;
                    }
                }

                // If no overlap logic applied, just add non-overlapping segments
                let new_segments =
                    self.new_segments_for_route(&complete_route, unconstrained_route.clone());

                if job.tracing {
                    let window_traces = job.window_trace.get_mut();
                    window_traces.push(WindowTrace {
                        start: window_start,
                        end: window_end,
                        segments: new_segments.clone(),
                        bridge: true, // Mark as unconstrained
                    });
                }

                complete_route.extend(new_segments.clone());

                // Update the point-to-segment mapping for the new segments
                self.update_point_segment_mapping(
                    &mut previous_point_segments,
                    window_start,
                    window_end,
                    &unconstrained_route,
                    &job.gps_points,
                );
            } else {
                // We have a valid constrained route
                if let Some(exp) = &job.expected_way_sequence {
                    if exp.active {
                        self.analyze_window_match(
                            job,
                            window_start,
                            window_end,
                            &window_route,
                            window_index,
                        );
                    }
                }

                // If there's overlap with previous windows,
                // consider whether to replace the previous route
                if !complete_route.is_empty() && window_start > 0 {
                    // Find the overlap point - the first GPS point in this window
                    let overlap_gps_idx = window_start;

                    // Find which segment in the complete route corresponds to this overlap point
                    let overlap_segment_idx = self.find_segment_for_gps_point(
                        &complete_route,
                        job.gps_points[overlap_gps_idx],
                    );

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
                        complete_route.extend(window_route.clone());

                        // Update the point-to-segment mapping for the new segments
                        self.update_point_segment_mapping(
                            &mut previous_point_segments,
                            window_start,
                            window_end,
                            &window_route,
                            &job.gps_points,
                        );

                        // Move window start ahead accounting for the replacement
                        window_start += step_size;
                        continue;
                    }
                }

                // If no replacement happened, just add non-overlapping segments
                let new_segments =
                    self.new_segments_for_route(&complete_route, window_route.clone());

                if job.tracing {
                    let window_traces = job.window_trace.get_mut();
                    window_traces.push(WindowTrace {
                        start: window_start,
                        end: window_end,
                        segments: new_segments.clone(),
                        bridge: false,
                    });
                }

                complete_route.extend(new_segments.clone());

                // Update the point-to-segment mapping for the new segments
                self.update_point_segment_mapping(
                    &mut previous_point_segments,
                    window_start,
                    window_end,
                    &window_route,
                    &job.gps_points,
                );
            }

            // Move window forward
            window_start += step_size;
        }

        if complete_route.is_empty() {
            return Err(anyhow!("Failed to build valid route"));
        }

        Ok(complete_route)
    }

    // Helper function to update the mapping between GPS points and their matched segments
    fn update_point_segment_mapping(
        &self,
        mapping: &mut HashMap<usize, u64>,
        window_start: usize,
        window_end: usize,
        segments: &[WaySegment],
        gps_points: &[Point<f64>],
    ) {
        if segments.is_empty() {
            return;
        }

        for point_idx in window_start..=window_end {
            if point_idx >= gps_points.len() {
                continue;
            }

            let point = gps_points[point_idx];
            let mut best_distance = f64::MAX;
            let mut best_segment_id = 0;

            for segment in segments {
                let projection = self.project_point_to_segment(point, segment);
                let distance = Haversine.distance(point, projection);

                if distance < best_distance {
                    best_distance = distance;
                    best_segment_id = segment.id;
                }
            }

            if best_segment_id != 0 && best_distance <= 150.0 {
                mapping.insert(point_idx, best_segment_id);
            }
        }
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

    fn validate_route_for_window(
        &self,
        route: &[WaySegment],
        window_points: &[Point<f64>],
    ) -> bool {
        // For each GPS point, check if it's within the maximum distance from any segment
        for &point in window_points {
            let mut min_distance = f64::MAX;

            // Find the minimum distance from this point to any segment in the route
            for segment in route {
                let projection = self.project_point_to_segment(point, segment);
                let distance = Haversine.distance(point, projection);
                min_distance = min_distance.min(distance);
            }

            // If the closest segment is still further than our threshold, the route is invalid
            if min_distance > 150.0 {
                // 150m max distance
                return false;
            }
        }

        true // All points are within acceptable distance
    }

    fn find_a_star_path_for_window(
        &self,
        job: &RouteMatchJob,
        start_idx: usize,
        end_idx: usize,
        previous_segment: Option<&WaySegment>,
        constraints: &[(usize, u64)],
    ) -> Result<Vec<WaySegment>> {
        // Get window data
        let window_points = &job.gps_points[start_idx..=end_idx];
        let window_timestamps = &job.timestamps[start_idx..=end_idx];
        let window_candidates = &job.all_candidates.borrow()[start_idx..=end_idx];

        // Validation
        if window_points.is_empty() || window_candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Create a map from point index in the original array to its relative position in the window
        let point_idx_map: HashMap<usize, usize> = (start_idx..=end_idx)
            .enumerate()
            .map(|(window_pos, original_idx)| (original_idx, window_pos))
            .collect();

        // Process constraints: Find candidates that match the constraint segment IDs
        let mut constrained_candidates: HashMap<usize, Vec<&SegmentCandidate>> = HashMap::new();

        for &(point_idx, segment_id) in constraints {
            if point_idx < start_idx || point_idx > end_idx {
                continue; // Skip constraints outside our window
            }

            let window_pos = point_idx_map[&point_idx];

            if window_pos >= window_candidates.len() {
                continue; // Safety check
            }

            // Find candidates that match the segment ID
            let matching_candidates: Vec<&SegmentCandidate> = window_candidates[window_pos]
                .iter()
                .filter(|cand| cand.segment.id == segment_id)
                .collect();

            if !matching_candidates.is_empty() {
                constrained_candidates.insert(window_pos, matching_candidates);
            }
        }

        // If we have constraints but none could be matched, it's a sign the window doesn't align well
        if !constraints.is_empty() && constrained_candidates.is_empty() {
            info!(
                "None of the {} constraints could be matched in this window",
                constraints.len()
            );
            return Ok(Vec::new()); // Signal that we need to use the unconstrained version
        }

        // Get first and last candidates based on constraints
        let first_candidates = if let Some(window_pos) = point_idx_map.get(&start_idx) {
            if let Some(constrained) = constrained_candidates.get(window_pos) {
                constrained.iter().map(|&c| c.clone()).collect::<Vec<_>>()
            } else {
                window_candidates[0].clone()
            }
        } else {
            window_candidates[0].clone()
        };

        let last_candidates = if let Some(window_pos) = point_idx_map.get(&end_idx) {
            if let Some(constrained) = constrained_candidates.get(window_pos) {
                constrained.iter().map(|&c| c.clone()).collect::<Vec<_>>()
            } else {
                window_candidates[window_candidates.len() - 1].clone()
            }
        } else {
            window_candidates[window_candidates.len() - 1].clone()
        };

        if first_candidates.is_empty() || last_candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Track best route
        let mut best_route = Vec::new();
        let mut best_score = f64::MAX;

        // Find paths, honoring constraints
        for first_candidate in &first_candidates {
            for last_candidate in &last_candidates {
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
                    Ok((path, path_cost)) => {
                        // Calculate score with bonus for matching constrained points
                        let mut constraint_bonus = 0.0;

                        // Check how well this path matches constrained points
                        for (window_pos, constrained_candidates) in &constrained_candidates {
                            let point_idx = start_idx + window_pos;
                            if point_idx >= job.gps_points.len() {
                                continue;
                            }

                            let constrained_segment_ids: HashSet<u64> = constrained_candidates
                                .iter()
                                .map(|c| c.segment.id)
                                .collect();

                            // Check if this path includes any of the constrained segments for this point
                            let matched = path
                                .iter()
                                .any(|segment| constrained_segment_ids.contains(&segment.id));

                            if matched {
                                // Significant bonus for matching a constrained point
                                constraint_bonus -= 50.0;
                            } else {
                                // Significant penalty for failing to match a constrained point
                                constraint_bonus += 100.0;
                            }
                        }

                        // Calculate comprehensive score
                        let basic_score = self.calculate_comprehensive_score(
                            &path,
                            window_points,
                            window_timestamps,
                            path_cost,
                            previous_segment,
                        );

                        // Combined score with constraint bonus/penalty
                        let adjusted_score = basic_score + constraint_bonus;

                        if adjusted_score < best_score {
                            best_route = path;
                            best_score = adjusted_score;
                        }
                    }
                    Err(_) => continue,
                }
            }
        }

        // Check if the best route satisfies critical constraints
        let route_validity = self.validate_route_for_window(&best_route, window_points);

        if !route_validity && !constraints.is_empty() {
            // If we have constraints but couldn't find a valid route,
            // signal that we should try without constraints
            return Ok(Vec::new());
        }

        if let Some(exp) = &job.expected_way_sequence {
            if exp.active {
                // Analysis code (unchanged)
                // ...
            }
        }

        Ok(best_route)
    }

    /// Calculate a comprehensive score for a candidate path
    fn calculate_comprehensive_score(
        &self,
        path: &[WaySegment],
        window_points: &[Point<f64>],
        window_timestamps: &[DateTime<Utc>],
        path_cost: f64,
        previous_segment: Option<&WaySegment>,
    ) -> f64 {
        if path.is_empty() || !self.validate_route_for_window(path, window_points) {
            return f64::MAX;
        }

        // Initialize score components
        let mut distance_score = 0.0;
        let mut length_score = 0.0;
        let mut road_class_score = 0.0;
        let mut continuity_score = 0.0;

        // 1. Calculate distance score (average distance from points to path)
        let mut total_distance = 0.0;
        let mut min_distances = Vec::with_capacity(window_points.len());

        for &point in window_points {
            let mut min_distance = f64::MAX;

            for segment in path {
                let projection = self.project_point_to_segment(point, segment);
                let distance = Haversine.distance(point, projection);
                min_distance = min_distance.min(distance);
            }

            total_distance += min_distance;
            min_distances.push(min_distance);
        }

        // Calculate average and standard deviation to penalize inconsistent matches
        let avg_distance = total_distance / window_points.len() as f64;
        let variance = min_distances
            .iter()
            .map(|&d| (d - avg_distance).powi(2))
            .sum::<f64>()
            / min_distances.len() as f64;
        let std_dev = variance.sqrt();

        distance_score = avg_distance + (std_dev * 0.5); // Penalize inconsistent matches

        // 2. Calculate length score (penalize unnecessarily long routes)
        if window_points.len() >= 2 {
            let direct_distance = Haversine.distance(
                *window_points.first().unwrap(),
                *window_points.last().unwrap(),
            );
            let path_length = path.iter().map(|segment| segment.length()).sum::<f64>();

            // Calculate detour ratio (how much longer than direct path)
            let detour_ratio = path_length / direct_distance;

            // Penalize routes that are more than 40% longer than direct path
            if detour_ratio > 1.4 {
                length_score = (detour_ratio - 1.4) * 100.0;
            }
        }

        // 3. Road class score (already implemented, but refactored)
        road_class_score = self.calculate_road_class_score(path);

        // 4. Continuity score (penalize fragmented routes with many transitions)
        let mut transitions = 0;

        for i in 1..path.len() {
            if path[i].osm_way_id != path[i - 1].osm_way_id {
                transitions += 1;
            }
        }

        continuity_score = transitions as f64 * 5.0; // 5 points per transition

        // 5. Add connection penalty/bonus for connection to previous segment
        let connection_score = if let Some(prev_segment) = previous_segment {
            if path.is_empty() {
                20.0 // Penalty for disconnected path
            } else {
                let first_segment = &path[0];
                if prev_segment.connections.contains(&first_segment.id) {
                    -10.0 // Bonus for connected path
                } else {
                    20.0 // Penalty for disconnected path
                }
            }
        } else {
            0.0 // No penalty/bonus if no previous segment
        };

        // Combine all scores with appropriate weights
         // Connection bonus/penalty

        (distance_score * 1.0) +   // Weight for distance to GPS points
            (length_score * 0.2) +     // Weight for route length
            (road_class_score * 0.8) + // Weight for road class
            (continuity_score * 0.5) + // Weight for route continuity
            connection_score
    }

    /// Calculate road class score for a path (lower is better)
    fn calculate_road_class_score(&self, path: &[WaySegment]) -> f64 {
        if path.is_empty() {
            return 0.0;
        }

        let mut score = 0.0;

        // Calculate weighted score based on highway types
        for segment in path {
            let segment_score = match segment.highway_type.as_str() {
                "motorway" | "motorway_link" => 1.0,
                "trunk" | "trunk_link" => 2.0,
                "primary" | "primary_link" => 4.0,
                "secondary" | "secondary_link" => 6.0,
                "tertiary" | "tertiary_link" => 8.0,
                "residential" => 10.0,
                "unclassified" => 12.0,
                "service" => 15.0,
                _ => 20.0,
            };

            // Weight score by segment length
            let length = segment.length();
            score += segment_score * (length / 100.0); // Normalize by 100 meters
        }

        // Normalize by total path length to get fair comparison between paths
        let total_length = path.iter().map(|segment| segment.length()).sum::<f64>();
        if total_length > 0.0 {
            score /= total_length / 100.0;
        }

        score
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

                // Use improved edge cost calculation
                let edge_cost = self.calculate_improved_edge_cost(
                    current_segment,
                    neighbor_segment,
                    segment_length,
                );

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

    /// Calculate improved edge cost for A* search, considering road type, length,
    /// and connection characteristics
    fn calculate_improved_edge_cost(
        &self,
        from_segment: &WaySegment,
        to_segment: &WaySegment,
        distance: f64,
    ) -> f64 {
        // Base cost is the distance
        let mut cost = distance;

        // 1. Apply road class factor - prefer higher class roads
        let road_class_factor = match to_segment.highway_type.as_str() {
            "motorway" => 0.4, // Strongly prefer motorways
            "motorway_link" => 0.5,
            "trunk" => 0.6,
            "trunk_link" => 0.7,
            "primary" => 0.8,
            "primary_link" => 0.9,
            "secondary" => 1.1,
            "secondary_link" => 1.2,
            "tertiary" => 1.4,
            "tertiary_link" => 1.5,
            "residential" => 1.8, // Penalize smaller roads
            "unclassified" => 2.2,
            "service" => 2.5,
            _ => 3.0,
        };

        cost *= road_class_factor;

        // 2. Apply continuity bonus - prefer staying on the same road
        if from_segment.osm_way_id == to_segment.osm_way_id {
            cost *= 0.8; // 20% discount for staying on the same road
        }

        // 3. Apply road name continuity bonus (if roads have the same name)
        if let (Some(from_name), Some(to_name)) = (&from_segment.name, &to_segment.name) {
            if !from_name.is_empty() && from_name == to_name {
                cost *= 0.9; // 10% discount for roads with the same name
            }
        }

        // 4. Apply angle penalty for sharp turns
        let from_end = from_segment.coordinates.last().unwrap();
        let from_before_end =
            from_segment.coordinates[from_segment.coordinates.len().saturating_sub(2)];
        let to_start = to_segment.coordinates.first().unwrap();
        let to_after_start = to_segment.coordinates.get(1).unwrap_or(to_start);

        // Calculate vectors
        let from_vec = (
            from_end.x - from_before_end.x,
            from_end.y - from_before_end.y,
        );
        let to_vec = (to_after_start.x - to_start.x, to_after_start.y - to_start.y);

        // Calculate angle using dot product
        let dot_product = from_vec.0 * to_vec.0 + from_vec.1 * to_vec.1;
        let from_mag = (from_vec.0.powi(2) + from_vec.1.powi(2)).sqrt();
        let to_mag = (to_vec.0.powi(2) + to_vec.1.powi(2)).sqrt();

        if from_mag > 0.0 && to_mag > 0.0 {
            let cos_angle = (dot_product / (from_mag * to_mag)).clamp(-1.0, 1.0);
            let angle = cos_angle.acos().to_degrees();

            // Penalize sharp turns (angles close to 90° or greater)
            if angle > 120.0 {
                let turn_penalty = 1.0 + (angle - 120.0) / 120.0; // Penalty increases with angle
                cost *= turn_penalty;
            }
        }

        cost
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

        // Build comprehensive node-to-segment mapping
        let mut node_to_segments: HashMap<u64, Vec<u64>> = HashMap::new();

        // Track all nodes, not just endpoints
        for segment in segment_map.values() {
            for &node_id in &segment.nodes {
                node_to_segments
                    .entry(node_id)
                    .or_default()
                    .push(segment.id);
            }
        }

        // First add the explicit connections from segment data
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

        // Track connectivity additions for reporting
        let mut added_endpoint_connections = 0;
        let mut added_intermediate_connections = 0;

        // Now check for shared nodes that might have been missed in explicit connections
        for (&node_id, connected_segments) in &node_to_segments {
            if connected_segments.len() > 1 {
                // This node connects multiple segments
                for (i, &segment_id) in connected_segments.iter().enumerate() {
                    for &other_segment_id in connected_segments.iter().skip(i + 1) {
                        // Skip if there's already an edge
                        if graph.contains_edge(segment_id, other_segment_id) {
                            continue;
                        }

                        // Get the segment objects
                        if let (Some(segment), Some(other_segment)) = (
                            segment_map.get(&segment_id),
                            segment_map.get(&other_segment_id),
                        ) {
                            // Check if segments are compatible for connection
                            if !self.are_segments_compatible(segment, other_segment) {
                                continue;
                            }

                            // Check the position of the shared node in each segment
                            let is_segment_endpoint = *segment.nodes.first().unwrap() == node_id
                                || *segment.nodes.last().unwrap() == node_id;

                            let is_other_endpoint = *other_segment.nodes.first().unwrap()
                                == node_id
                                || *other_segment.nodes.last().unwrap() == node_id;

                            // Case 1: Both segments have this node as an endpoint (standard case)
                            if is_segment_endpoint && is_other_endpoint {
                                let cost = calculate_transition_cost(segment, other_segment);
                                graph.add_edge(segment_id, other_segment_id, cost);
                                added_endpoint_connections += 1;
                                continue;
                            }

                            // Case 2: Endpoint connecting to an intermediate node (intersection)
                            if is_segment_endpoint || is_other_endpoint {
                                // For cases where a road endpoint meets another road at an intersection
                                // This is a legitimate connection - create a path from endpoint to midpoint
                                let cost = calculate_transition_cost(segment, other_segment) * 1.2;
                                graph.add_edge(segment_id, other_segment_id, cost);
                                added_intermediate_connections += 1;
                                continue;
                            }

                            // Case 3: Both nodes are intermediate
                            // This is the trickiest case - need to check if this is actually
                            // an intersection rather than just two roads passing over each other

                            // First check if these segments are from the same OSM way
                            // In that case, they definitely should connect
                            if segment.osm_way_id == other_segment.osm_way_id {
                                let cost = calculate_transition_cost(segment, other_segment);
                                graph.add_edge(segment_id, other_segment_id, cost);
                                added_intermediate_connections += 1;
                                continue;
                            }

                            // For different OSM ways, check other factors that suggest connectivity

                            // 1. Compare the road names - if they're the same, likely connected
                            let same_name = match (&segment.name, &other_segment.name) {
                                (Some(name1), Some(name2)) => name1 == name2 && !name1.is_empty(),
                                _ => false,
                            };

                            // 2. Check if they share multiple nodes (stronger evidence of connection)
                            let common_nodes_count = segment
                                .nodes
                                .iter()
                                .filter(|&n| other_segment.nodes.contains(n))
                                .count();

                            // 3. Get highway types and check if they're compatible
                            let compatible_roads = are_road_types_compatible(
                                &segment.highway_type,
                                &other_segment.highway_type,
                            );

                            // 4. Use a topological check (do these roads visibly intersect)
                            let line1 = LineString::from(segment.coordinates.clone());
                            let line2 = LineString::from(other_segment.coordinates.clone());
                            let geometry_intersects = line1.intersects(&line2);

                            // If multiple factors suggest connection, connect the segments
                            let should_connect = (same_name && common_nodes_count > 0)
                                || (common_nodes_count >= 2)
                                || (compatible_roads && geometry_intersects);

                            if should_connect {
                                // Higher cost for this less certain connection
                                let cost = calculate_transition_cost(segment, other_segment) * 1.5;
                                graph.add_edge(segment_id, other_segment_id, cost);
                                added_intermediate_connections += 1;
                            }
                        }
                    }
                }
            }
        }

        // Log connectivity improvements
        debug!(
            "Built road network graph with {} nodes and {} edges",
            graph.node_count(),
            graph.edge_count()
        );

        info!(
            "Added {} connections between endpoints and {} connections via intermediate nodes",
            added_endpoint_connections, added_intermediate_connections
        );

        Ok((graph, segment_map))
    }

    // Helper function to check if two segments should be connected based on properties
    fn are_segments_compatible(&self, segment1: &WaySegment, segment2: &WaySegment) -> bool {
        // 1. Check for layer differences (different z-levels)
        let layer1 = segment1
            .metadata
            .as_ref()
            .and_then(|m| m.get("layer").map(|l| l.parse::<i8>().unwrap_or(0)))
            .unwrap_or(0);

        let layer2 = segment2
            .metadata
            .as_ref()
            .and_then(|m| m.get("layer").map(|l| l.parse::<i8>().unwrap_or(0)))
            .unwrap_or(0);

        // Different layers shouldn't connect (e.g., bridges and tunnels)
        if layer1 != layer2 {
            return false;
        }

        // 2. Check for bridge/tunnel mismatches
        let is_bridge1 = segment1
            .metadata
            .as_ref()
            .and_then(|m| m.get("bridge").map(|v| v == "yes"))
            .unwrap_or(false);

        let is_bridge2 = segment2
            .metadata
            .as_ref()
            .and_then(|m| m.get("bridge").map(|v| v == "yes"))
            .unwrap_or(false);

        if is_bridge1 != is_bridge2 {
            return false;
        }

        let is_tunnel1 = segment1
            .metadata
            .as_ref()
            .and_then(|m| m.get("tunnel").map(|v| v == "yes"))
            .unwrap_or(false);

        let is_tunnel2 = segment2
            .metadata
            .as_ref()
            .and_then(|m| m.get("tunnel").map(|v| v == "yes"))
            .unwrap_or(false);

        if is_tunnel1 != is_tunnel2 {
            return false;
        }

        // 3. Check for road class compatibility
        let compatible_types = crate::osm_preprocessing::are_road_types_compatible(
            &segment1.highway_type,
            &segment2.highway_type,
        );

        if !compatible_types {
            return false;
        }

        // All checks passed
        true
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
                "osm_way_id": segment.osm_way_id, // Include OSM way ID
                "rank": i,
                "score": candidate.score,
                "distance": candidate.distance,
                "highway_type": segment.highway_type,
                "description": format!("Segment ID: {} (OSM: {}), Rank: {}, Score: {:.2}, Distance: {:.2}m", 
                                       segment.id, segment.osm_way_id, i, candidate.score, candidate.distance)
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
                    "osm_way_id": segment.osm_way_id, // Include OSM way ID
                    "description": format!("Projection to segment {} (OSM: {})", segment.id, segment.osm_way_id)
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

    // Update the debug_way_ids method in route_matcher.rs to handle OSM way IDs
    fn debug_way_ids(
        &mut self,
        final_route: &[WaySegment],
        way_ids: &[u64],
        loaded_tiles: &HashSet<String>,
    ) -> Result<()> {
        // First check if any of the specified way IDs are in the final route (now checking osm_way_id)
        let route_osm_way_ids: HashSet<u64> =
            final_route.iter().map(|seg| seg.osm_way_id).collect();

        for &way_id in way_ids {
            if route_osm_way_ids.contains(&way_id) {
                info!(
                    "Debug: OSM Way ID {} is included in the final route",
                    way_id
                );

                // List all segments with this OSM way ID
                let segments: Vec<&WaySegment> = final_route
                    .iter()
                    .filter(|seg| seg.osm_way_id == way_id)
                    .collect();

                for segment in segments {
                    info!(
                        "Debug: OSM Way ID {} corresponds to segment ID {}",
                        way_id, segment.id,
                    );
                }
                continue;
            }

            // Way ID not in final route, investigate why
            info!(
                "Debug: OSM Way ID {} is NOT included in the final route",
                way_id
            );

            // Check if the way ID exists in loaded tiles
            let mut way_exists = false;
            let mut way_segment: Option<WaySegment> = None;

            for tile_id in loaded_tiles {
                let tile = self.tile_loader.load_tile(tile_id)?;
                // Now search by osm_way_id instead of id
                if let Some(segment) = tile.road_segments.iter().find(|s| s.osm_way_id == way_id) {
                    way_exists = true;
                    way_segment = Some(segment.clone());
                    info!(
                        "Debug: OSM Way ID {} exists in tile {} as segment ID {}",
                        way_id, tile_id, segment.id
                    );
                    break;
                }
            }

            if !way_exists {
                info!(
                    "Debug: OSM Way ID {} was not found in any loaded tile",
                    way_id
                );
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
                    "Debug: OSM Way ID {} (segment ID {}) is closest to segment #{} in route (distance: {:.2}m)",
                    way_id, segment.id, closest_point_idx, closest_distance
                );

                // Check connections to nearby segments
                let nearby_segment = &final_route[closest_point_idx];
                let is_connected = segment.connections.contains(&nearby_segment.id)
                    || nearby_segment.connections.contains(&segment.id);

                if is_connected {
                    info!(
                        "Debug: Segment ID {} (OSM Way ID {}) is connected to segment {} (OSM Way ID {}) in the route",
                        segment.id, way_id, nearby_segment.id, nearby_segment.osm_way_id
                    );
                } else {
                    info!(
                        "Debug: Segment ID {} (OSM Way ID {}) is NOT connected to segment {} (OSM Way ID {}) in the route",
                        segment.id, way_id, nearby_segment.id, nearby_segment.osm_way_id
                    );
                }

                // Analyze highway type and other attributes
                info!(
                    "Debug: OSM Way ID {} is type '{}' (route segment is type '{}')",
                    way_id, segment.highway_type, nearby_segment.highway_type
                );

                // Check if it would create a loop
                let mut test_route = final_route.to_vec();
                test_route.insert(closest_point_idx + 1, segment.clone());
            }
        }

        Ok(())
    }

    // Analyze the connectivity for the entire way sequence. As such the ways must be in the expected order.
    pub fn check_way_sequence_connectivity(&self, way_ids: &[u64]) -> Result<Vec<String>> {
        if way_ids.len() < 2 {
            return Err(anyhow!("Need at least 2 way IDs to check connectivity"));
        }

        let mut issues = Vec::new();
        let mut segments_by_way = HashMap::new();

        // Step 1: Find all segments for each way ID
        info!("Finding segments for {} way IDs", way_ids.len());

        // First load all tiles to ensure we have all necessary data
        let mut all_tiles = HashSet::new();

        for &way_id in way_ids {
            // Find all segments with this OSM way ID across all tiles
            let way_segments = self.find_segments_by_osm_way_id(way_id)?;

            if way_segments.is_empty() {
                issues.push(format!(
                    "OSM Way ID {} not found in any loaded tile",
                    way_id
                ));
                continue;
            }

            // Add the tiles containing these segments to our collection
            for (segment, tile_id) in &way_segments {
                all_tiles.insert(tile_id.clone());
            }

            segments_by_way.insert(way_id, way_segments);
        }

        // Step 2: Load all relevant tiles to ensure connectivity info is available
        let mut tile_loader_clone = TileLoader::new(
            self.tile_loader.tile_directory.clone(),
            self.config.max_cached_tiles,
            self.config.tile_config.clone(),
        );

        for tile_id in all_tiles {
            tile_loader_clone.load_tile(&tile_id)?;
        }

        // Step 3: Check connectivity between consecutive way pairs
        for i in 0..way_ids.len() - 1 {
            let way_id1 = way_ids[i];
            let way_id2 = way_ids[i + 1];

            if !segments_by_way.contains_key(&way_id1) || !segments_by_way.contains_key(&way_id2) {
                continue; // Skip if we couldn't find either way
            }

            // Check if there's any connection between segments of these two consecutive ways
            let segments1 = &segments_by_way[&way_id1];
            let segments2 = &segments_by_way[&way_id2];

            let mut connected = false;
            let mut closest_segments = (0, 0, f64::MAX); // (seg1_idx, seg2_idx, distance)
            let mut connection_issues = Vec::new();

            // For each pair of segments between the two ways, check connectivity
            for (idx1, (segment1, _)) in segments1.iter().enumerate() {
                for (idx2, (segment2, _)) in segments2.iter().enumerate() {
                    // Check direct connection
                    if segment1.connections.contains(&segment2.id)
                        || segment2.connections.contains(&segment1.id)
                    {
                        connected = true;

                        // No need to check further if we found a connection
                        break;
                    }

                    // Track closest segments for diagnostics
                    let s1_center = segment1.centroid();
                    let s2_center = segment2.centroid();
                    let distance = Haversine.distance(s1_center, s2_center);

                    if distance < closest_segments.2 {
                        closest_segments = (idx1, idx2, distance);
                    }

                    // Check if they should be connected (segments that intersect or share endpoints)
                    let should_connect = self.should_segments_connect(segment1, segment2);
                    if should_connect.0 && !connected {
                        connection_issues.push(format!(
                            "Segments {} and {} should be connected: {}",
                            segment1.id, segment2.id, should_connect.1
                        ));
                    }
                }

                if connected {
                    break;
                }
            }

            if connected {
                info!(
                    "Ways {} and {} are properly connected (consecutive in path)",
                    way_id1, way_id2
                );
            } else {
                // Get the closest segments for detailed analysis
                let (idx1, idx2, distance) = closest_segments;
                let (segment1, _) = &segments1[idx1];
                let (segment2, _) = &segments2[idx2];

                // Check for geometric issues like near-intersection
                issues.push(format!(
                "No connection found between consecutive ways {} and {} (closest segments: {} and {}, distance: {:.2}m)",
                way_id1, way_id2, segment1.id, segment2.id, distance
            ));

                // Add any specific connection issues we found
                for issue in connection_issues {
                    issues.push(format!("  - {}", issue));
                }

                // Check for layer differences or bridge/tunnel status
                self.check_layer_differences(segment1, segment2, &mut issues);
            }
        }

        if issues.is_empty() {
            issues.push("All consecutive ways in the path are properly connected".to_string());
        }

        Ok(issues)
    }

    // Helper function to find all segments with a specific OSM way ID
    fn find_segments_by_osm_way_id(&self, osm_way_id: u64) -> Result<Vec<(WaySegment, String)>> {
        let mut segments = Vec::new();

        // Search in loaded tiles first for efficiency
        for (tile_id, tile) in &self.tile_loader.loaded_tiles {
            for segment in &tile.road_segments {
                if segment.osm_way_id == osm_way_id {
                    segments.push((segment.clone(), tile_id.clone()));
                }
            }
        }

        // If none found, search all tiles on disk
        if segments.is_empty() {
            debug!("Searching disk for OSM way ID {}", osm_way_id);

            let entries = std::fs::read_dir(&self.config.tile_cache_dir)
                .map_err(|e| anyhow!("Failed to read tile directory: {}", e))?;

            for entry in entries {
                let entry = entry.map_err(|e| anyhow!("Failed to read directory entry: {}", e))?;
                let path = entry.path();

                // Only check bin files
                if path.is_file() && path.extension().is_some_and(|ext| ext == "bin") {
                    let file_name = path.file_stem().unwrap().to_string_lossy();

                    // Load the tile
                    let tile_bytes = std::fs::read(&path)
                        .map_err(|e| anyhow!("Failed to read tile file {}: {}", file_name, e))?;

                    let config = bincode::config::standard();
                    let (tile_index, _): (TileIndex, _) =
                        bincode::serde::decode_from_slice(&tile_bytes, config)
                            .map_err(|e| anyhow!("Invalid tile data in {}: {}", file_name, e))?;

                    // Check segments
                    for segment in &tile_index.road_segments {
                        if segment.osm_way_id == osm_way_id {
                            segments.push((segment.clone(), file_name.to_string()));
                        }
                    }
                }
            }
        }

        Ok(segments)
    }

    // Helper function to determine if two segments should be connected
    fn should_segments_connect(
        &self,
        segment1: &WaySegment,
        segment2: &WaySegment,
    ) -> (bool, String) {
        // First check for shared nodes - this is the most reliable indicator
        let mut shared_nodes = Vec::new();
        for node_id in &segment1.nodes {
            if segment2.nodes.contains(node_id) {
                shared_nodes.push(*node_id);
            }
        }

        if !shared_nodes.is_empty() {
            // Also check if these shared nodes are at endpoints
            let is_endpoint1 = shared_nodes.iter().any(|&node_id| {
                *segment1.nodes.first().unwrap() == node_id
                    || *segment1.nodes.last().unwrap() == node_id
            });

            let is_endpoint2 = shared_nodes.iter().any(|&node_id| {
                *segment2.nodes.first().unwrap() == node_id
                    || *segment2.nodes.last().unwrap() == node_id
            });

            let endpoint_info = if is_endpoint1 && is_endpoint2 {
                " (shared endpoint nodes)"
            } else {
                " (shared intermediate nodes)"
            };

            return (
                true,
                format!(
                    "They share {} node(s): {:?}{}",
                    shared_nodes.len(),
                    shared_nodes,
                    endpoint_info
                ),
            );
        }

        // Check for points that are very close to each other
        const CLOSE_DISTANCE_THRESHOLD: f64 = 1.0; // Increased from 0.5 to 1.0 meters for better matching

        let seg1_start = segment1.coordinates.first().unwrap();
        let seg1_end = segment1.coordinates.last().unwrap();
        let seg2_start = segment2.coordinates.first().unwrap();
        let seg2_end = segment2.coordinates.last().unwrap();

        // Check start-start
        let start_start_point = Point::new(seg1_start.x, seg1_start.y);
        let start_start_dist =
            Haversine.distance(start_start_point, Point::new(seg2_start.x, seg2_start.y));
        if start_start_dist < CLOSE_DISTANCE_THRESHOLD {
            return (
                true,
                format!("Start points are very close ({:.2}m)", start_start_dist),
            );
        }

        // Check start-end
        let start_end_dist =
            Haversine.distance(start_start_point, Point::new(seg2_end.x, seg2_end.y));
        if start_end_dist < CLOSE_DISTANCE_THRESHOLD {
            return (
                true,
                format!(
                    "Start point of #1 is close to end point of #2 ({:.2}m)",
                    start_end_dist
                ),
            );
        }

        // Check end-start
        let end_start_point = Point::new(seg1_end.x, seg1_end.y);
        let end_start_dist =
            Haversine.distance(end_start_point, Point::new(seg2_start.x, seg2_start.y));
        if end_start_dist < CLOSE_DISTANCE_THRESHOLD {
            return (
                true,
                format!(
                    "End point of #1 is close to start point of #2 ({:.2}m)",
                    end_start_dist
                ),
            );
        }

        // Check end-end
        let end_end_dist = Haversine.distance(end_start_point, Point::new(seg2_end.x, seg2_end.y));
        if end_end_dist < CLOSE_DISTANCE_THRESHOLD {
            return (
                true,
                format!("End points are very close ({:.2}m)", end_end_dist),
            );
        }

        // Create LineStrings for intersection check
        let line1 = LineString::from(segment1.coordinates.clone());
        let line2 = LineString::from(segment2.coordinates.clone());

        // Check if lines intersect
        if line1.intersects(&line2) {
            return (true, "Lines geometrically intersect".to_string());
        }

        (false, "No connection criteria met".to_string())
    }

    // Helper to check layer differences
    fn check_layer_differences(
        &self,
        segment1: &WaySegment,
        segment2: &WaySegment,
        issues: &mut Vec<String>,
    ) {
        // Extract layer information
        let layer1 = segment1
            .metadata
            .as_ref()
            .and_then(|m| m.get("layer").map(|l| l.parse::<i8>().unwrap_or(0)))
            .unwrap_or(0);

        let layer2 = segment2
            .metadata
            .as_ref()
            .and_then(|m| m.get("layer").map(|l| l.parse::<i8>().unwrap_or(0)))
            .unwrap_or(0);

        let is_bridge1 = segment1
            .metadata
            .as_ref()
            .and_then(|m| m.get("bridge").map(|v| v == "yes"))
            .unwrap_or(false);

        let is_bridge2 = segment2
            .metadata
            .as_ref()
            .and_then(|m| m.get("bridge").map(|v| v == "yes"))
            .unwrap_or(false);

        let is_tunnel1 = segment1
            .metadata
            .as_ref()
            .and_then(|m| m.get("tunnel").map(|v| v == "yes"))
            .unwrap_or(false);

        let is_tunnel2 = segment2
            .metadata
            .as_ref()
            .and_then(|m| m.get("tunnel").map(|v| v == "yes"))
            .unwrap_or(false);

        // Compare and report differences
        if layer1 != layer2 {
            issues.push(format!(
                "  - Segments are on different layers: {} vs {}",
                layer1, layer2
            ));
        }

        if is_bridge1 != is_bridge2 {
            issues.push(format!(
                "  - Bridge mismatch: {} is {}a bridge, {} is {}a bridge",
                segment1.id,
                if is_bridge1 { "" } else { "not " },
                segment2.id,
                if is_bridge2 { "" } else { "not " }
            ));
        }

        if is_tunnel1 != is_tunnel2 {
            issues.push(format!(
                "  - Tunnel mismatch: {} is {}a tunnel, {} is {}a tunnel",
                segment1.id,
                if is_tunnel1 { "" } else { "not " },
                segment2.id,
                if is_tunnel2 { "" } else { "not " }
            ));
        }
    }

    /// Debug a specific route between two GPS points using a required sequence of OSM way IDs
    ///
    /// This function helps diagnose why a specific sequence of OSM way IDs might not be matched
    /// between two GPS points. It performs a detailed analysis of the connectivity and routing
    /// issues that might prevent the exact sequence from being matched.
    ///
    /// # Arguments
    ///
    /// * `start_point` - The starting GPS point (lon, lat)
    /// * `end_point` - The ending GPS point (lon, lat)
    /// * `way_ids` - The exact sequence of OSM way IDs that should be matched
    ///
    /// # Returns
    ///
    /// A detailed diagnostic report explaining why the route cannot be matched as described,
    /// or confirming that it can be matched.
    pub fn debug_specific_route(
        &mut self,
        start_point: (f64, f64),
        end_point: (f64, f64),
        way_ids: Vec<u64>,
    ) -> Result<String> {
        info!(
            "Debugging specific route between ({}, {}) and ({}, {}) with way IDs: {:?}",
            start_point.0, start_point.1, end_point.0, end_point.1, way_ids
        );

        if way_ids.is_empty() {
            return Err(anyhow!("No way IDs provided for debugging"));
        }

        // Create report structure
        let mut report = DebugReport {
            start_point: geo::Point::new(start_point.0, start_point.1),
            end_point: geo::Point::new(end_point.0, end_point.1),
            way_ids: way_ids.clone(),
            way_segments: HashMap::new(),
            ways_not_found: Vec::new(),
            way_candidates: HashMap::new(),
            segment_connections: HashMap::new(),
            path_analysis: Vec::new(),
            summary: String::new(),
        };

        // Step 1: Load all tiles covering the area between the points with a buffer
        let mut bbox = geo::Rect::new(
            geo::Coord {
                x: start_point.0.min(end_point.0),
                y: start_point.1.min(end_point.1),
            },
            geo::Coord {
                x: start_point.0.max(end_point.0),
                y: start_point.1.max(end_point.1),
            },
        );

        // Add buffer around the bbox (2km converted to approximate degrees)
        let buffer = 0.02; // Roughly 2km at the equator
        bbox = geo::Rect::new(
            geo::Coord {
                x: bbox.min().x - buffer,
                y: bbox.min().y - buffer,
            },
            geo::Coord {
                x: bbox.max().x + buffer,
                y: bbox.max().y + buffer,
            },
        );

        let loaded_tiles =
            self.tile_loader
                .load_tile_range(bbox, buffer, self.config.max_tiles_per_depth)?;

        info!("Loaded {} tiles for debugging", loaded_tiles.len());

        // Step 2: Find all segments for each way ID
        let mut missing_ways = false;
        for &way_id in &way_ids {
            match self.find_segments_by_osm_way_id(way_id) {
                Ok(segments) => {
                    if segments.is_empty() {
                        report.ways_not_found.push(way_id);
                        missing_ways = true;
                    } else {
                        report.way_segments.insert(way_id, segments);
                    }
                }
                Err(_) => {
                    report.ways_not_found.push(way_id);
                    missing_ways = true;
                }
            }
        }

        if missing_ways {
            report.summary = format!(
                "Could not match the route because the following OSM ways were not found: {:?}",
                report.ways_not_found
            );
            return Ok(report.generate());
        }

        // Step 3: Check connectivity between consecutive ways
        let mut connectivity_issues = false;
        for i in 0..way_ids.len() - 1 {
            let way1 = way_ids[i];
            let way2 = way_ids[i + 1];

            let segments1 = &report.way_segments[&way1];
            let segments2 = &report.way_segments[&way2];

            let mut connection_found = false;
            let mut closest_distance = f64::MAX;
            let mut closest_segment_pair = ((0, 0), (0, 0), 0.0); // ((seg1_idx, osm_id1), (seg2_idx, osm_id2), distance)

            // Check if any segment from way1 connects to any segment from way2
            for (s1_idx, (segment1, _)) in segments1.iter().enumerate() {
                report.segment_connections.insert(segment1.id, Vec::new());

                for (s2_idx, (segment2, _)) in segments2.iter().enumerate() {
                    // Check if directly connected
                    let direct_connection = segment1.connections.contains(&segment2.id)
                        || segment2.connections.contains(&segment1.id);

                    if direct_connection {
                        connection_found = true;
                        report
                            .segment_connections
                            .get_mut(&segment1.id)
                            .unwrap()
                            .push(segment2.id);
                    }

                    // Calculate the distance between segments for diagnostic purposes
                    let s1_centroid = segment1.centroid();
                    let s2_centroid = segment2.centroid();
                    let distance = Haversine.distance(s1_centroid, s2_centroid);

                    if distance < closest_distance {
                        closest_distance = distance;
                        closest_segment_pair =
                            ((s1_idx, segment1.id), (s2_idx, segment2.id), distance);
                    }
                }
            }

            // Record the connectivity analysis
            if !connection_found {
                connectivity_issues = true;
                report.path_analysis.push(PathSegmentAnalysis {
                    segment_idx: i,
                    way_id1: way1,
                    way_id2: way2,
                    connected: false,
                    reason: format!(
                        "No connection found between OSM ways {} and {}. Closest segments are {} and {} at {:.2}m apart.",
                        way1, way2, closest_segment_pair.0.1, closest_segment_pair.1.1, closest_segment_pair.2
                    ),
                    details: self.analyze_connectivity_failure(
                        &report.way_segments[&way1][closest_segment_pair.0.0].0,
                        &report.way_segments[&way2][closest_segment_pair.1.0].0
                    )
                });
            } else {
                report.path_analysis.push(PathSegmentAnalysis {
                    segment_idx: i,
                    way_id1: way1,
                    way_id2: way2,
                    connected: true,
                    reason: format!("OSM ways {} and {} are connected", way1, way2),
                    details: Vec::new(),
                });
            }
        }

        if connectivity_issues {
            report.summary =
                "Could not match the route due to connectivity issues between OSM ways."
                    .to_string();
            return Ok(report.generate());
        }

        // Step 4: Verify coverage of the route at start and end points
        let start_point_geo = geo::Point::new(start_point.0, start_point.1);
        let end_point_geo = geo::Point::new(end_point.0, end_point.1);

        let mut start_coverage_ok = false;
        let mut end_coverage_ok = false;
        let mut start_distances = Vec::new();
        let mut end_distances = Vec::new();

        // Check start point proximity to first way
        let first_way_segments = &report.way_segments[&way_ids[0]];
        for (segment, _) in first_way_segments {
            let distance = self.project_point_to_segment_distance(start_point_geo, segment);
            start_distances.push((segment.id, distance));

            if distance <= self.config.max_matching_distance {
                start_coverage_ok = true;
            }
        }

        // Check end point proximity to last way
        let last_way_segments = &report.way_segments[&way_ids[way_ids.len() - 1]];
        for (segment, _) in last_way_segments {
            let distance = self.project_point_to_segment_distance(end_point_geo, segment);
            end_distances.push((segment.id, distance));

            if distance <= self.config.max_matching_distance {
                end_coverage_ok = true;
            }
        }

        // Update report with coverage issues
        if !start_coverage_ok {
            start_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            let closest = start_distances.first().unwrap_or(&(0, f64::MAX));

            report.summary = format!(
                "Start point is too far ({:.2}m) from the first way ID {}. Maximum allowed distance is {:.2}m.",
                closest.1, way_ids[0], self.config.max_matching_distance
            );
            return Ok(report.generate());
        }

        if !end_coverage_ok {
            end_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            let closest = end_distances.first().unwrap_or(&(0, f64::MAX));

            report.summary = format!(
                "End point is too far ({:.2}m) from the last way ID {}. Maximum allowed distance is {:.2}m.",
                closest.1,
                way_ids[way_ids.len() - 1],
                self.config.max_matching_distance
            );
            return Ok(report.generate());
        }

        // Step 5: Try to find a complete path through all segments
        let mut current_way_idx = 0;
        let mut path_ok = true;

        // Start with segments from first way
        let mut current_segments: Vec<u64> = first_way_segments
            .iter()
            .filter(|(seg, _)| {
                let dist = self.project_point_to_segment_distance(start_point_geo, seg);
                dist <= self.config.max_matching_distance
            })
            .map(|(seg, _)| seg.id)
            .collect();

        if current_segments.is_empty() {
            report.summary = format!(
                "Could not find any segments in way {} near the start point",
                way_ids[0]
            );
            return Ok(report.generate());
        }

        // Try to follow the path through all ways
        while current_way_idx < way_ids.len() - 1 {
            let next_way = way_ids[current_way_idx + 1];

            // Find segments in the next way that connect to any of our current segments
            let mut next_segments = Vec::new();
            let next_way_segments = &report.way_segments[&next_way];

            for &current_seg_id in &current_segments {
                for (next_seg, _) in next_way_segments {
                    if self.segments_are_connected(current_seg_id, next_seg.id) {
                        next_segments.push(next_seg.id);
                    }
                }
            }

            if next_segments.is_empty() {
                path_ok = false;
                report.path_analysis[current_way_idx].reason +=
                    " (No valid path found between segments)";
                break;
            }

            // Found valid next segments
            current_segments = next_segments;
            current_way_idx += 1;
        }

        // Final path validation - check if last segments reach the end point
        if path_ok {
            let last_segments = &report.way_segments[&way_ids[way_ids.len() - 1]];
            let mut end_reachable = false;

            for &seg_id in &current_segments {
                if let Some((segment, _)) = last_segments.iter().find(|(s, _)| s.id == seg_id) {
                    let distance = self.project_point_to_segment_distance(end_point_geo, segment);
                    if distance <= self.config.max_matching_distance {
                        end_reachable = true;
                        break;
                    }
                }
            }

            if !end_reachable {
                path_ok = false;
                report.summary = "Found a valid segment path through all ways, but it doesn't reach the end point".to_string();
            }
        }

        if !path_ok && report.summary.is_empty() {
            report.summary =
                "Could not construct a valid path through all required OSM ways".to_string();
        } else if path_ok {
            report.summary =
                "The specified route SHOULD be matchable through the given OSM way sequence."
                    .to_string();

            // Add reasons why it might still not be matching
            report.summary +=
                "\n\nIf the map matcher is still not using this route, potential reasons include:";
            report.summary += "\n1. The algorithm's scoring function prefers other routes based on distance, heading, road type, etc.";
            report.summary += "\n2. The map matcher might be using a different sliding window size that skips some of these connections.";
            report.summary += "\n3. There might be ambiguities at junctions where multiple paths have similar scores.";
            report.summary += "\n4. The GPS points might be too noisy or sparse for the matcher to identify this specific sequence.";
            report.summary += "\n5. The matcher may be imposing constraints on turn angles, road types, or other factors not considered here.";
        }

        Ok(report.generate())
    }

    /// Helper function to check if two segments are connected
    fn segments_are_connected(&mut self, segment1_id: u64, segment2_id: u64) -> bool {
        // Try to load the segments from the segment map
        let segment_map = match self.get_segment_map() {
            Ok(map) => map,
            Err(_) => return false,
        };

        // Check direct connectivity first
        if let Some(segment1) = segment_map.get(&segment1_id) {
            if segment1.connections.contains(&segment2_id) {
                return true;
            }
        }

        if let Some(segment2) = segment_map.get(&segment2_id) {
            if segment2.connections.contains(&segment1_id) {
                return true;
            }
        }

        // If either segment is missing from our map, try to load it directly
        let segment1 = match segment_map.get(&segment1_id) {
            Some(s) => s.clone(),
            None => match self.tile_loader.get_segment(segment1_id) {
                Ok(s) => s,
                Err(_) => return false,
            },
        };

        let segment2 = match segment_map.get(&segment2_id) {
            Some(s) => s.clone(),
            None => match self.tile_loader.get_segment(segment2_id) {
                Ok(s) => s,
                Err(_) => return false,
            },
        };

        // Check if segments should be connected based on more advanced criteria
        let (should_connect, _) = self.should_segments_connect(&segment1, &segment2);
        should_connect
    }

    /// Calculate the distance from a point to a segment after projection
    fn project_point_to_segment_distance(
        &self,
        point: geo::Point<f64>,
        segment: &WaySegment,
    ) -> f64 {
        let projection = self.project_point_to_segment(point, segment);
        Haversine.distance(point, projection)
    }

    /// Get a copy of the current segment map for analysis
    fn get_segment_map(&self) -> Result<HashMap<u64, WaySegment>> {
        let mut segment_map = HashMap::new();

        // Collect all segments from loaded tiles
        for tile in self.tile_loader.loaded_tiles.values() {
            for segment in &tile.road_segments {
                segment_map.insert(segment.id, segment.clone());
            }
        }

        Ok(segment_map)
    }

    /// Analyze why two segments don't connect
    fn analyze_connectivity_failure(
        &self,
        segment1: &WaySegment,
        segment2: &WaySegment,
    ) -> Vec<String> {
        let mut reasons = Vec::new();

        // 1. Check for layer differences
        let layer1 = segment1
            .metadata
            .as_ref()
            .and_then(|m| m.get("layer").map(|l| l.parse::<i8>().unwrap_or(0)))
            .unwrap_or(0);

        let layer2 = segment2
            .metadata
            .as_ref()
            .and_then(|m| m.get("layer").map(|l| l.parse::<i8>().unwrap_or(0)))
            .unwrap_or(0);

        if layer1 != layer2 {
            reasons.push(format!(
                "Segments are on different layers: {} vs {}",
                layer1, layer2
            ));
        }

        // 2. Check bridge/tunnel status
        let is_bridge1 = segment1
            .metadata
            .as_ref()
            .and_then(|m| m.get("bridge").map(|v| v == "yes"))
            .unwrap_or(false);

        let is_bridge2 = segment2
            .metadata
            .as_ref()
            .and_then(|m| m.get("bridge").map(|v| v == "yes"))
            .unwrap_or(false);

        if is_bridge1 != is_bridge2 {
            reasons.push(format!(
                "Bridge mismatch: segment {} is {}a bridge, segment {} is {}a bridge",
                segment1.id,
                if is_bridge1 { "" } else { "not " },
                segment2.id,
                if is_bridge2 { "" } else { "not " }
            ));
        }

        let is_tunnel1 = segment1
            .metadata
            .as_ref()
            .and_then(|m| m.get("tunnel").map(|v| v == "yes"))
            .unwrap_or(false);

        let is_tunnel2 = segment2
            .metadata
            .as_ref()
            .and_then(|m| m.get("tunnel").map(|v| v == "yes"))
            .unwrap_or(false);

        if is_tunnel1 != is_tunnel2 {
            reasons.push(format!(
                "Tunnel mismatch: segment {} is {}a tunnel, segment {} is {}a tunnel",
                segment1.id,
                if is_tunnel1 { "" } else { "not " },
                segment2.id,
                if is_tunnel2 { "" } else { "not " }
            ));
        }

        // 3. Check road types and compatibility
        if !are_road_types_compatible(&segment1.highway_type, &segment2.highway_type) {
            reasons.push(format!(
                "Incompatible road types: {} ({}) and {} ({})",
                segment1.highway_type, segment1.id, segment2.highway_type, segment2.id
            ));
        }

        // 4. Check for shared nodes
        let shared_nodes: Vec<u64> = segment1
            .nodes
            .iter()
            .filter(|n| segment2.nodes.contains(n))
            .cloned()
            .collect();

        if shared_nodes.is_empty() {
            reasons.push("No shared nodes between segments".to_string());

            // If no shared nodes, check the minimum distance between endpoints
            let s1_start = segment1.coordinates.first().unwrap();
            let s1_end = segment1.coordinates.last().unwrap();
            let s2_start = segment2.coordinates.first().unwrap();
            let s2_end = segment2.coordinates.last().unwrap();

            let start_start = Haversine.distance(
                Point::new(s1_start.x, s1_start.y),
                Point::new(s2_start.x, s2_start.y),
            );

            let start_end = Haversine.distance(
                Point::new(s1_start.x, s1_start.y),
                Point::new(s2_end.x, s2_end.y),
            );

            let end_start = Haversine.distance(
                Point::new(s1_end.x, s1_end.y),
                Point::new(s2_start.x, s2_start.y),
            );

            let end_end = Haversine.distance(
                Point::new(s1_end.x, s1_end.y),
                Point::new(s2_end.x, s2_end.y),
            );

            let min_distance = start_start.min(start_end).min(end_start).min(end_end);

            reasons.push(format!(
                "Minimum distance between endpoints: {:.2}m",
                min_distance
            ));

            if min_distance < 1.0 {
                reasons.push(
                    "Endpoints are very close (<1m) but not connected in the road network data"
                        .to_string(),
                );
            }
        } else {
            reasons.push(format!(
                "Segments share {} node(s) but are not marked as connected in the road network",
                shared_nodes.len()
            ));
        }

        // 5. Check oneway restrictions
        if segment1.is_oneway && segment2.is_oneway {
            // Check if the direction allows connection
            let s1_end_node = *segment1.nodes.last().unwrap();
            let s2_start_node = *segment2.nodes.first().unwrap();

            if s1_end_node != s2_start_node {
                reasons.push(
                    "Both segments are one-way and may not connect in the required direction"
                        .to_string(),
                );
            }
        } else if segment1.is_oneway {
            reasons.push(format!(
                "Segment {} is one-way which may restrict connectivity",
                segment1.id
            ));
        } else if segment2.is_oneway {
            reasons.push(format!(
                "Segment {} is one-way which may restrict connectivity",
                segment2.id
            ));
        }

        reasons
    }

    pub fn analyze_window_match(
        &self,
        job: &RouteMatchJob,
        window_start: usize,
        window_end: usize,
        chosen_path: &[WaySegment],
        window_idx: usize,
    ) {
        // Skip if there's no expectation set
        let expectation = match &job.expected_way_sequence {
            Some(exp) if exp.active => exp,
            _ => return,
        };

        // Check if this window contains both the expected start and end points
        let contains_start = window_start <= expectation.start_point_idx
            && expectation.start_point_idx <= window_end;
        let contains_end =
            window_start <= expectation.end_point_idx && expectation.end_point_idx <= window_end;

        if !contains_start || !contains_end {
            // This window doesn't fully contain the expected path points
            return;
        }

        job.add_explanation(format!(
            "Window {} (points {}-{}) contains both expected path endpoints ({} and {})",
            window_idx,
            window_start,
            window_end,
            expectation.start_point_idx,
            expectation.end_point_idx
        ));

        // Extract the way IDs from the chosen path
        let chosen_way_ids: Vec<u64> = chosen_path
            .iter()
            .map(|segment| segment.osm_way_id)
            .collect();

        // First, check if all way IDs were considered as candidates
        let mut all_ways_available = true;
        let mut way_availability = Vec::new();

        for &way_id in &expectation.way_ids {
            let mut way_found = false;
            let mut points_containing_way = Vec::new();

            for point_idx in expectation.start_point_idx..=expectation.end_point_idx {
                if job.is_way_id_candidate(point_idx, way_id) {
                    way_found = true;
                    points_containing_way.push(point_idx);
                }
            }

            if !way_found {
                all_ways_available = false;
                job.add_explanation(format!(
                    "Way ID {} was not considered as a candidate for any relevant point",
                    way_id
                ));
            } else {
                let mut rank_info = Vec::new();
                for &point_idx in &points_containing_way {
                    if let Some((rank, score)) = job.get_way_id_candidate_rank(point_idx, way_id) {
                        rank_info.push(format!(
                            "point {}: rank {} (score: {:.2})",
                            point_idx,
                            rank + 1,
                            score
                        ));
                    }
                }

                way_availability.push(format!(
                    "Way ID {} was a candidate for points {} with rankings: {}",
                    way_id,
                    points_containing_way
                        .iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    rank_info.join(", ")
                ));
            }
        }

        // Add the way availability information
        for info in way_availability {
            job.add_explanation(info);
        }

        if !all_ways_available && !expectation.is_subsequence {
            job.add_explanation(
                "SEQUENCE NOT CONSIDERED: Not all expected way IDs were available as candidates for relevant points".to_string()
            );
            return;
        }

        // Check if the expected sequence exists as a continuous subsequence in the chosen path
        let is_subsequence = self.is_continuous_subsequence(&expectation.way_ids, &chosen_way_ids);

        if is_subsequence {
            job.add_explanation("SUCCESS: The expected way sequence was found as a continuous subsequence in the matched path".to_string());
            return;
        }

        // If we're here, the subsequence wasn't matched despite all ways being available
        job.add_explanation("SEQUENCE NOT MATCHED: The expected way sequence was not found as a continuous subsequence in the matched path".to_string());

        // Check if the connectivity is possible
        let sequence_valid =
            self.check_if_sequence_considered(job, window_start, window_end, &expectation.way_ids);

        if !sequence_valid {
            job.add_explanation(
                "SEQUENCE NOT VIABLE: The expected sequence could not be constructed as a valid path".to_string()
            );

            // Try to find why
            let connectivity_issues = self.find_connectivity_issues(job, &expectation.way_ids);
            for issue in connectivity_issues {
                job.add_explanation(format!("  - {}", issue));
            }
            return;
        }

        job.add_explanation(
            "SEQUENCE WAS VIABLE: The expected way sequence could be constructed as a valid path"
                .to_string(),
        );

        // Find the longest matching subsequence
        let longest_match =
            self.find_longest_matching_subsequence(&expectation.way_ids, &chosen_way_ids);

        if !longest_match.is_empty() {
            job.add_explanation(format!(
                "Longest matching subsequence: {:?} ({} of {} way IDs)",
                longest_match,
                longest_match.len(),
                expectation.way_ids.len()
            ));
        }

        // Find missing and extra ways
        let missing_segments =
            self.identify_missing_segments(&expectation.way_ids, &chosen_way_ids);

        if !missing_segments.is_empty() {
            job.add_explanation(format!(
                "Missing segments in the expected sequence: {:?}",
                missing_segments
            ));
        }

        // Calculate approximate path scores for comparison
        if let Some(expected_score) = self.estimate_path_score_for_relevant_points(
            job,
            &expectation.way_ids,
            expectation.start_point_idx,
            expectation.end_point_idx,
        ) {
            // Calculate score for the chosen path but only for relevant points
            let chosen_points_score = self.calculate_chosen_path_score_for_points(
                job,
                chosen_path,
                expectation.start_point_idx,
                expectation.end_point_idx,
            );

            if let Some(chosen_score) = chosen_points_score {
                job.add_explanation(format!(
                    "SCORING COMPARISON (for points {}-{} only): Chosen path score: {:.2}, Expected path score: {:.2} (lower is better)",
                    expectation.start_point_idx, expectation.end_point_idx, chosen_score, expected_score
                ));

                let percentage_diff = ((expected_score - chosen_score) / chosen_score) * 100.0;

                if percentage_diff > 0.0 {
                    job.add_explanation(format!(
                        "Your expected path scored {:.1}% worse than the chosen path for the relevant points",
                        percentage_diff
                    ));
                } else {
                    job.add_explanation(format!(
                        "Your expected path scored {:.1}% better than the chosen path for the relevant points!",
                        -percentage_diff
                    ));
                    job.add_explanation("This suggests other factors influenced the matcher's decision beyond point-to-path distance.".to_string());
                }

                // Try to explain specifically why the expected path scored differently
                self.analyze_scoring_differences_for_relevant_points(
                    job,
                    &expectation.way_ids,
                    chosen_path,
                    expectation.start_point_idx,
                    expectation.end_point_idx,
                );
            }
        }

        // Analyze why this specific path was chosen
        job.add_explanation("The matcher likely chose a different path because:".to_string());
        job.add_explanation(
            "- The path may have better overall continuity in the larger window context"
                .to_string(),
        );
        job.add_explanation(
            "- There might be connectivity issues within your expected sequence".to_string(),
        );
        job.add_explanation(
            "- The scoring function prefers different road types or fewer transitions".to_string(),
        );
        job.add_explanation(
            "- The sliding window algorithm might not have considered your exact sequence"
                .to_string(),
        );

        // Suggest improvements
        job.add_explanation("\nSuggestions to improve matching:".to_string());
        job.add_explanation(
            "1. Verify the connectivity of your expected way sequence in OSM data".to_string(),
        );
        job.add_explanation("2. Try adjusting the window size in the matcher".to_string());
        job.add_explanation("3. Check for possible ambiguities at junctions where multiple paths have similar scores".to_string());
        job.add_explanation("4. Consider if the expected sequence makes sense for the GPS points (proper road segment choice, direction, etc.)".to_string());
    }

    /// Check if expected sequence is a continuous subsequence of the chosen sequence
    fn is_continuous_subsequence(&self, expected: &[u64], chosen: &[u64]) -> bool {
        if expected.is_empty() || chosen.is_empty() {
            return false;
        }

        // Look for the start of the subsequence
        for i in 0..=chosen.len() - expected.len() {
            let mut match_found = true;

            // Check if the subsequence matches starting at position i
            for j in 0..expected.len() {
                if chosen[i + j] != expected[j] {
                    match_found = false;
                    break;
                }
            }

            if match_found {
                return true;
            }
        }

        false
    }

    /// Find the longest matching subsequence between expected and chosen paths
    fn find_longest_matching_subsequence(&self, expected: &[u64], chosen: &[u64]) -> Vec<u64> {
        if expected.is_empty() || chosen.is_empty() {
            return Vec::new();
        }

        let mut longest = Vec::new();
        let mut current = Vec::new();

        // Find longest continuous subsequence
        for &way_id in expected {
            let mut found = false;

            // See if this continues the current subsequence
            if !current.is_empty() {
                let last_idx = chosen
                    .iter()
                    .position(|&x| x == *current.last().unwrap())
                    .unwrap();
                if last_idx + 1 < chosen.len() && chosen[last_idx + 1] == way_id {
                    current.push(way_id);
                    found = true;
                }
            }

            // If not continuing, see if we can start a new subsequence
            if !found {
                if let Some(idx) = chosen.iter().position(|&x| x == way_id) {
                    // This could be the start of a new subsequence
                    if current.len() > longest.len() {
                        longest = current;
                    }
                    current = vec![way_id];
                    found = true;
                }
            }

            // If not found at all, break the current subsequence
            if !found {
                if current.len() > longest.len() {
                    longest = current;
                }
                current = Vec::new();
            }
        }

        // Check the last subsequence
        if current.len() > longest.len() {
            longest = current;
        }

        longest
    }

    /// Identify missing segments in the expected path vs chosen path
    fn identify_missing_segments(&self, expected: &[u64], chosen: &[u64]) -> Vec<u64> {
        let chosen_set: HashSet<u64> = chosen.iter().cloned().collect();

        expected
            .iter()
            .filter(|&way_id| !chosen_set.contains(way_id))
            .cloned()
            .collect()
    }

    /// Estimate score for a path defined by way IDs, but only for relevant points
    fn estimate_path_score_for_relevant_points(
        &self,
        job: &RouteMatchJob,
        way_ids: &[u64],
        start_point_idx: usize,
        end_point_idx: usize,
    ) -> Option<f64> {
        // Get segment map
        let segment_map = job.segment_map.borrow();

        // Find segments for each way
        let mut way_segments = HashMap::new();
        for &way_id in way_ids {
            let segments: Vec<u64> = segment_map
                .iter()
                .filter(|(_, segment)| segment.osm_way_id == way_id)
                .map(|(id, _)| *id)
                .collect();

            if !segments.is_empty() {
                way_segments.insert(way_id, segments);
            }
        }

        // Try to build a continuous path
        let mut path_segments = Vec::new();
        let mut current_way_idx = 0;
        let mut current_segment = None;

        while current_way_idx < way_ids.len() {
            let way_id = way_ids[current_way_idx];

            if !way_segments.contains_key(&way_id) {
                return None; // Way has no segments
            }

            let segments = &way_segments[&way_id];

            if current_segment.is_none() {
                // Just pick the first segment of the first way
                current_segment = segment_map.get(&segments[0]).cloned();
                path_segments.push(segments[0]);
            } else {
                // Find a segment from the current way that connects to the current segment
                let current_seg = current_segment.as_ref().unwrap();
                let mut found_next = false;

                for &seg_id in segments {
                    if current_seg.connections.contains(&seg_id) {
                        current_segment = segment_map.get(&seg_id).cloned();
                        path_segments.push(seg_id);
                        found_next = true;
                        break;
                    }
                }

                if !found_next {
                    // Just pick the first segment as fallback
                    current_segment = segment_map.get(&segments[0]).cloned();
                    path_segments.push(segments[0]);
                }
            }

            current_way_idx += 1;
        }

        // Calculate path score ONLY for relevant points
        let mut total_score = 0.0;
        let mut point_distances = Vec::new();

        // 1. Calculate distance score (GPS points to segments)
        for point_idx in start_point_idx..=end_point_idx {
            if point_idx >= job.gps_points.len() {
                continue;
            }

            let point = job.gps_points[point_idx];
            let mut min_distance = f64::MAX;
            let mut closest_seg_id = 0;

            for &seg_id in &path_segments {
                if let Some(segment) = segment_map.get(&seg_id) {
                    let projection = self.project_point_to_segment(point, segment);
                    let distance = Haversine.distance(point, projection);
                    if distance < min_distance {
                        min_distance = distance;
                        closest_seg_id = seg_id;
                    }
                }
            }

            point_distances.push((point_idx, min_distance, closest_seg_id));

            // Distance score component
            total_score += min_distance / 20.0; // Normalize distance
        }

        // 2. Road type penalties - only consider segments for relevant points
        let mut used_segments = HashSet::new();
        for (_, _, seg_id) in &point_distances {
            used_segments.insert(*seg_id);
        }

        for &seg_id in &used_segments {
            if let Some(segment) = segment_map.get(&seg_id) {
                // Add penalty based on road type
                let type_penalty = match segment.highway_type.as_str() {
                    "motorway" | "motorway_link" => 0.1,
                    "trunk" | "trunk_link" => 0.2,
                    "primary" | "primary_link" => 0.3,
                    "secondary" | "secondary_link" => 0.5,
                    "tertiary" | "tertiary_link" => 0.7,
                    "residential" => 1.0,
                    "unclassified" => 1.2,
                    _ => 1.5,
                };

                total_score += type_penalty;
            }
        }

        // 3. Transition penalties - just count the ways for simplicity
        total_score += way_ids.len() as f64 * 0.2;

        Some(total_score)
    }

    /// Calculate the score of the chosen path but only for the relevant points
    fn calculate_chosen_path_score_for_points(
        &self,
        job: &RouteMatchJob,
        chosen_path: &[WaySegment],
        start_point_idx: usize,
        end_point_idx: usize,
    ) -> Option<f64> {
        if chosen_path.is_empty() {
            return None;
        }

        let mut total_score = 0.0;
        let mut point_distances = Vec::new();

        // 1. Calculate distance score (relevant GPS points to segments)
        for point_idx in start_point_idx..=end_point_idx {
            if point_idx >= job.gps_points.len() {
                continue;
            }

            let point = job.gps_points[point_idx];
            let mut min_distance = f64::MAX;
            let mut closest_segment_idx = 0;

            for (i, segment) in chosen_path.iter().enumerate() {
                let projection = self.project_point_to_segment(point, segment);
                let distance = Haversine.distance(point, projection);
                if distance < min_distance {
                    min_distance = distance;
                    closest_segment_idx = i;
                }
            }

            point_distances.push((point_idx, min_distance, closest_segment_idx));

            // Distance score component
            total_score += min_distance / 20.0; // Normalize distance
        }

        // 2. Road type penalties - only consider segments for relevant points
        let mut used_segments = HashSet::new();
        for (_, _, seg_idx) in &point_distances {
            used_segments.insert(*seg_idx);
        }

        for &seg_idx in &used_segments {
            let segment = &chosen_path[seg_idx];

            // Add penalty based on road type
            let type_penalty = match segment.highway_type.as_str() {
                "motorway" | "motorway_link" => 0.1,
                "trunk" | "trunk_link" => 0.2,
                "primary" | "primary_link" => 0.3,
                "secondary" | "secondary_link" => 0.5,
                "tertiary" | "tertiary_link" => 0.7,
                "residential" => 1.0,
                "unclassified" => 1.2,
                _ => 1.5,
            };

            total_score += type_penalty;
        }

        // 3. Transition penalties - just count the different ways for simplicity
        let mut way_count = 0;
        let mut last_way_id = 0;

        for &seg_idx in &used_segments {
            let way_id = chosen_path[seg_idx].osm_way_id;
            if way_id != last_way_id {
                way_count += 1;
                last_way_id = way_id;
            }
        }

        total_score += way_count as f64 * 0.2;

        Some(total_score)
    }

    /// Analyze specific differences in scoring between expected and chosen paths
    /// but only for relevant points
    fn analyze_scoring_differences_for_relevant_points(
        &self,
        job: &RouteMatchJob,
        expected_way_ids: &[u64],
        chosen_path: &[WaySegment],
        start_point_idx: usize,
        end_point_idx: usize,
    ) {
        let segment_map = job.segment_map.borrow();

        // Find segments for expected path
        let mut expected_segments = Vec::new();
        for &way_id in expected_way_ids {
            for (_, segment) in segment_map.iter() {
                if segment.osm_way_id == way_id {
                    expected_segments.push(segment);
                }
            }
        }

        if expected_segments.is_empty() {
            return;
        }

        // Analyze distance to relevant GPS points
        let mut expected_distances = Vec::new();
        let mut chosen_distances = Vec::new();
        let mut worse_points = 0;

        for point_idx in start_point_idx..=end_point_idx {
            if point_idx >= job.gps_points.len() {
                continue;
            }

            let point = job.gps_points[point_idx];

            // Find closest expected segment
            let mut expected_min_distance = f64::MAX;
            let mut closest_expected_segment = "";
            for segment in &expected_segments {
                let projection = self.project_point_to_segment(point, segment);
                let distance = Haversine.distance(point, projection);
                if distance < expected_min_distance {
                    expected_min_distance = distance;
                    closest_expected_segment = &segment.highway_type;
                }
            }

            // Find closest chosen segment
            let mut chosen_min_distance = f64::MAX;
            let mut closest_chosen_segment = "";
            for segment in chosen_path {
                let projection = self.project_point_to_segment(point, segment);
                let distance = Haversine.distance(point, projection);
                if distance < chosen_min_distance {
                    chosen_min_distance = distance;
                    closest_chosen_segment = &segment.highway_type;
                }
            }

            expected_distances.push((point_idx, expected_min_distance, closest_expected_segment));
            chosen_distances.push((point_idx, chosen_min_distance, closest_chosen_segment));

            if expected_min_distance > chosen_min_distance {
                worse_points += 1;
            }
        }

        // Create a detailed point-by-point comparison
        job.add_explanation("\nDetailed point-by-point distance comparison:".to_string());

        for ((point_idx, expected_dist, expected_type), (_, chosen_dist, chosen_type)) in
            expected_distances.iter().zip(chosen_distances.iter())
        {
            let comparison = if expected_dist > chosen_dist {
                format!("WORSE by {:.2}m", expected_dist - chosen_dist)
            } else {
                format!("BETTER by {:.2}m", chosen_dist - expected_dist)
            };

            job.add_explanation(format!(
                "Point {}: Expected={:.2}m ({}), Chosen={:.2}m ({}), {}",
                point_idx, expected_dist, expected_type, chosen_dist, chosen_type, comparison
            ));
        }

        // Summarize the distances
        let expected_total: f64 = expected_distances.iter().map(|(_, dist, _)| dist).sum();
        let chosen_total: f64 = chosen_distances.iter().map(|(_, dist, _)| dist).sum();
        let point_count = expected_distances.len() as f64;

        let expected_avg = expected_total / point_count;
        let chosen_avg = chosen_total / point_count;

        job.add_explanation(format!(
            "\nAverage distances: Expected={:.2}m, Chosen={:.2}m",
            expected_avg, chosen_avg
        ));

        job.add_explanation(format!(
            "Your expected path is worse at {}/{} points",
            worse_points,
            expected_distances.len()
        ));

        // Compare road types
        let expected_types: HashMap<_, _> =
            expected_segments
                .iter()
                .map(|s| &s.highway_type)
                .fold(HashMap::new(), |mut acc, t| {
                    *acc.entry(t).or_insert(0) += 1;
                    acc
                });

        let chosen_types: HashMap<_, _> =
            chosen_path
                .iter()
                .map(|s| &s.highway_type)
                .fold(HashMap::new(), |mut acc, t| {
                    *acc.entry(t).or_insert(0) += 1;
                    acc
                });

        job.add_explanation("\nRoad type comparison:".to_string());
        job.add_explanation(
            "Expected path: ".to_string()
                + &expected_types
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v))
                    .collect::<Vec<_>>()
                    .join(", "),
        );

        job.add_explanation(
            "Chosen path: ".to_string()
                + &chosen_types
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v))
                    .collect::<Vec<_>>()
                    .join(", "),
        );
    }

    /// Check if a sequence of way IDs could be considered as a valid path
    fn check_if_sequence_considered(
        &self,
        job: &RouteMatchJob,
        window_start: usize,
        window_end: usize,
        way_ids: &[u64],
    ) -> bool {
        // Early exit for empty sequence
        if way_ids.is_empty() {
            return false;
        }

        // Get segment map
        let segment_map = job.segment_map.borrow();

        // For each way_id, find all possible segments
        let mut way_segments = HashMap::new();
        for &way_id in way_ids {
            let mut segments = Vec::new();

            // Find all segments with this way ID
            for (seg_id, segment) in segment_map.iter() {
                if segment.osm_way_id == way_id {
                    segments.push(*seg_id);
                }
            }

            if segments.is_empty() {
                return false; // Way ID has no segments
            }

            way_segments.insert(way_id, segments);
        }

        // Check if consecutive ways can be connected
        for i in 0..way_ids.len() - 1 {
            let way1 = way_ids[i];
            let way2 = way_ids[i + 1];

            let segments1 = &way_segments[&way1];
            let segments2 = &way_segments[&way2];

            let mut connected = false;

            // Check if any segment from way1 connects to any segment from way2
            for &seg1_id in segments1 {
                if let Some(seg1) = segment_map.get(&seg1_id) {
                    for &seg2_id in segments2 {
                        if seg1.connections.contains(&seg2_id) {
                            connected = true;
                            break;
                        }
                    }
                }

                if connected {
                    break;
                }
            }

            if !connected {
                return false; // Ways cannot be connected
            }
        }

        // All ways have segments and can be connected
        true
    }

    /// Find connectivity issues in the way sequence
    fn find_connectivity_issues(&self, job: &RouteMatchJob, way_ids: &[u64]) -> Vec<String> {
        let mut issues = Vec::new();

        let segment_map = job.segment_map.borrow();

        // Find segments for each way
        let mut way_segments = HashMap::new();
        for &way_id in way_ids {
            let mut segments = Vec::new();

            // Find all segments with this way ID
            for (seg_id, segment) in segment_map.iter() {
                if segment.osm_way_id == way_id {
                    segments.push(*seg_id);
                }
            }

            if segments.is_empty() {
                issues.push(format!(
                    "Way ID {} has no segments in the loaded map data",
                    way_id
                ));
            } else {
                way_segments.insert(way_id, segments);
            }
        }

        // Check connections between consecutive ways
        for i in 0..way_ids.len() - 1 {
            let way1 = way_ids[i];
            let way2 = way_ids[i + 1];

            if !way_segments.contains_key(&way1) || !way_segments.contains_key(&way2) {
                continue; // Already reported missing segments
            }

            let segments1 = &way_segments[&way1];
            let segments2 = &way_segments[&way2];

            let mut connected = false;
            let mut closest_distance = f64::MAX;
            let mut closest_pair = (0, 0);

            // Check connections and find closest segments
            for (i1, &seg1_id) in segments1.iter().enumerate() {
                if let Some(seg1) = segment_map.get(&seg1_id) {
                    for (i2, &seg2_id) in segments2.iter().enumerate() {
                        if let Some(seg2) = segment_map.get(&seg2_id) {
                            if seg1.connections.contains(&seg2_id) {
                                connected = true;
                                break;
                            }

                            // Calculate distance between segment centroids
                            let distance = Haversine.distance(seg1.centroid(), seg2.centroid());
                            if distance < closest_distance {
                                closest_distance = distance;
                                closest_pair = (seg1_id, seg2_id);
                            }
                        }
                    }
                }

                if connected {
                    break;
                }
            }

            if !connected {
                issues.push(format!(
                    "Way {} and {} are not connected in the road network. Closest segments {} and {} are {:.2}m apart",
                    way1, way2, closest_pair.0, closest_pair.1, closest_distance
                ));

                // If they're close but not connected, check why
                if closest_distance < 50.0 {
                    if let (Some(seg1), Some(seg2)) = (
                        segment_map.get(&closest_pair.0),
                        segment_map.get(&closest_pair.1),
                    ) {
                        let (should_connect, reason) = self.should_segments_connect(seg1, seg2);
                        if should_connect {
                            issues.push(format!(
                                "Segments should be connected based on proximity, but aren't: {}",
                                reason
                            ));
                        } else {
                            issues.push(format!(
                                "Segments are close but shouldn't connect: {}",
                                reason
                            ));
                        }
                    }
                }
            }
        }

        issues
    }

    /// Estimate score for a path defined by way IDs
    fn estimate_path_score(
        &self,
        job: &RouteMatchJob,
        way_ids: &[u64],
        window_start: usize,
        window_end: usize,
    ) -> Option<f64> {
        let segment_map = job.segment_map.borrow();

        // Find segments for each way
        let mut way_segments = HashMap::new();
        for &way_id in way_ids {
            let segments: Vec<u64> = segment_map
                .iter()
                .filter(|(_, segment)| segment.osm_way_id == way_id)
                .map(|(id, _)| *id)
                .collect();

            if !segments.is_empty() {
                way_segments.insert(way_id, segments);
            }
        }

        // Try to build a continuous path
        let mut path_segments = Vec::new();
        let mut current_way_idx = 0;
        let mut current_segment = None;

        while current_way_idx < way_ids.len() {
            let way_id = way_ids[current_way_idx];

            if !way_segments.contains_key(&way_id) {
                return None; // Way has no segments
            }

            let segments = &way_segments[&way_id];

            if current_segment.is_none() {
                // Just pick the first segment of the first way
                current_segment = segment_map.get(&segments[0]).cloned();
                path_segments.push(segments[0]);
            } else {
                // Find a segment from the current way that connects to the current segment
                let current_seg = current_segment.as_ref().unwrap();
                let mut found_next = false;

                for &seg_id in segments {
                    if current_seg.connections.contains(&seg_id) {
                        current_segment = segment_map.get(&seg_id).cloned();
                        path_segments.push(seg_id);
                        found_next = true;
                        break;
                    }
                }

                if !found_next {
                    // Just pick the first segment as fallback
                    current_segment = segment_map.get(&segments[0]).cloned();
                    path_segments.push(segments[0]);
                }
            }

            current_way_idx += 1;
        }

        // Calculate path score
        let mut total_score = 0.0;

        // 1. Calculate distance score (GPS points to segments)
        for point_idx in window_start..=window_end {
            if point_idx >= job.gps_points.len() {
                continue;
            }

            let point = job.gps_points[point_idx];
            let mut min_distance = f64::MAX;

            for &seg_id in &path_segments {
                if let Some(segment) = segment_map.get(&seg_id) {
                    let projection = self.project_point_to_segment(point, segment);
                    let distance = Haversine.distance(point, projection);
                    min_distance = min_distance.min(distance);
                }
            }

            // Distance score component
            total_score += min_distance / 20.0; // Normalize distance
        }

        for point_idx in window_start..=window_end {
            if point_idx >= job.gps_points.len() {
                continue;
            }

            let point = job.gps_points[point_idx];
            let mut min_distance = f64::MAX;
            let mut closest_segment_id = 0;

            for &seg_id in &path_segments {
                if let Some(segment) = segment_map.get(&seg_id) {
                    let projection = self.project_point_to_segment(point, segment);
                    let distance = Haversine.distance(point, projection);
                    if distance < min_distance {
                        min_distance = distance;
                        closest_segment_id = seg_id;
                    }
                }
            }

            // Add detailed logging for each point
            job.add_explanation(format!(
                "DEBUG: Point {}: distance to closest segment {} is {:.2}m",
                point_idx, closest_segment_id, min_distance
            ));

            // Distance score component
            total_score += min_distance / 20.0; // Normalize distance
        }

        // 2. Road type penalties
        for &seg_id in &path_segments {
            if let Some(segment) = segment_map.get(&seg_id) {
                // Add penalty based on road type
                let type_penalty = match segment.highway_type.as_str() {
                    "motorway" | "motorway_link" => 0.1,
                    "trunk" | "trunk_link" => 0.2,
                    "primary" | "primary_link" => 0.3,
                    "secondary" | "secondary_link" => 0.5,
                    "tertiary" | "tertiary_link" => 0.7,
                    "residential" => 1.0,
                    "unclassified" => 1.2,
                    _ => 1.5,
                };

                total_score += type_penalty;
            }
        }

        // 3. Transition penalties
        if path_segments.len() > 1 {
            for i in 0..path_segments.len() - 1 {
                let seg1_id = path_segments[i];
                let seg2_id = path_segments[i + 1];

                if let (Some(seg1), Some(seg2)) =
                    (segment_map.get(&seg1_id), segment_map.get(&seg2_id))
                {
                    if seg1.osm_way_id != seg2.osm_way_id {
                        // Penalty for transitioning between different ways
                        total_score += 1.0;
                    }
                }
            }
        }

        Some(total_score)
    }

    /// Analyze specific differences in scoring between expected and chosen paths
    fn analyze_scoring_differences(
        &self,
        job: &RouteMatchJob,
        expected_way_ids: &[u64],
        chosen_path: &[WaySegment],
        window_start: usize,
        window_end: usize,
    ) {
        let segment_map = job.segment_map.borrow();

        // Find segments for expected path
        let mut expected_segments = Vec::new();
        for &way_id in expected_way_ids {
            for (_, segment) in segment_map.iter() {
                if segment.osm_way_id == way_id {
                    expected_segments.push(segment);
                }
            }
        }

        if expected_segments.is_empty() {
            return;
        }

        // Analyze distance to GPS points
        let mut expected_total_distance = 0.0;
        let mut chosen_total_distance = 0.0;
        let mut worse_points = 0;

        for point_idx in window_start..=window_end {
            if point_idx >= job.gps_points.len() {
                continue;
            }

            let point = job.gps_points[point_idx];

            // Find closest expected segment
            let mut expected_min_distance = f64::MAX;
            for segment in &expected_segments {
                let projection = self.project_point_to_segment(point, segment);
                let distance = Haversine.distance(point, projection);
                expected_min_distance = expected_min_distance.min(distance);
            }

            // Find closest chosen segment
            let mut chosen_min_distance = f64::MAX;
            for segment in chosen_path {
                let projection = self.project_point_to_segment(point, segment);
                let distance = Haversine.distance(point, projection);
                chosen_min_distance = chosen_min_distance.min(distance);
            }

            expected_total_distance += expected_min_distance;
            chosen_total_distance += chosen_min_distance;

            if expected_min_distance > chosen_min_distance {
                worse_points += 1;
            }
        }

        let point_count = (window_end - window_start + 1) as f64;
        let expected_avg_distance = expected_total_distance / point_count;
        let chosen_avg_distance = chosen_total_distance / point_count;

        if expected_avg_distance > chosen_avg_distance * 1.2 {
            job.add_explanation(format!(
                "MAIN ISSUE - DISTANCE: Your expected path is further from GPS points (avg {:.2}m vs {:.2}m)",
                expected_avg_distance, chosen_avg_distance
            ));
            job.add_explanation(format!(
                "Your path is worse at {}/{} points in the window",
                worse_points,
                window_end - window_start + 1
            ));
        }

        // Compare road types
        let mut expected_highway_types = HashMap::new();
        let mut chosen_highway_types = HashMap::new();

        for segment in &expected_segments {
            *expected_highway_types
                .entry(segment.highway_type.clone())
                .or_insert(0) += 1;
        }

        for segment in chosen_path {
            *chosen_highway_types
                .entry(segment.highway_type.clone())
                .or_insert(0) += 1;
        }

        // Rate road types (lower is better)
        let rate_road_type = |highway_type: &str| -> f64 {
            match highway_type {
                "motorway" | "motorway_link" => 1.0,
                "trunk" | "trunk_link" => 2.0,
                "primary" | "primary_link" => 3.0,
                "secondary" | "secondary_link" => 4.0,
                "tertiary" | "tertiary_link" => 5.0,
                "residential" => 6.0,
                "unclassified" => 7.0,
                _ => 8.0,
            }
        };

        let mut expected_road_score = 0.0;
        let mut chosen_road_score = 0.0;

        for (highway_type, count) in &expected_highway_types {
            expected_road_score += rate_road_type(highway_type) * (*count as f64);
        }

        for (highway_type, count) in &chosen_highway_types {
            chosen_road_score += rate_road_type(highway_type) * (*count as f64);
        }

        expected_road_score /= expected_segments.len() as f64;
        chosen_road_score /= chosen_path.len() as f64;

        if expected_road_score > chosen_road_score * 1.2 {
            job.add_explanation(format!(
                "ISSUE - ROAD TYPES: Your expected path uses less preferred road types (score {:.2} vs {:.2}, lower is better)",
                expected_road_score, chosen_road_score
            ));

            job.add_explanation(
                "Chosen path road types: ".to_string()
                    + &chosen_highway_types
                        .iter()
                        .map(|(k, v)| format!("{}: {}", k, v))
                        .collect::<Vec<_>>()
                        .join(", "),
            );

            job.add_explanation(
                "Expected path road types: ".to_string()
                    + &expected_highway_types
                        .iter()
                        .map(|(k, v)| format!("{}: {}", k, v))
                        .collect::<Vec<_>>()
                        .join(", "),
            );
        }

        // Check transitions between ways
        let expected_transitions = expected_way_ids.len() - 1;
        let mut chosen_transitions = 0;

        for i in 1..chosen_path.len() {
            if chosen_path[i].osm_way_id != chosen_path[i - 1].osm_way_id {
                chosen_transitions += 1;
            }
        }

        if expected_transitions > chosen_transitions {
            job.add_explanation(format!(
                "ISSUE - TRANSITIONS: Your expected path has more way transitions ({} vs {})",
                expected_transitions, chosen_transitions
            ));
        }
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

/// Structure to hold a debug report for route analysis
struct DebugReport {
    start_point: geo::Point<f64>,
    end_point: geo::Point<f64>,
    way_ids: Vec<u64>,
    way_segments: HashMap<u64, Vec<(WaySegment, String)>>,
    ways_not_found: Vec<u64>,
    way_candidates: HashMap<u64, Vec<u64>>,
    segment_connections: HashMap<u64, Vec<u64>>,
    path_analysis: Vec<PathSegmentAnalysis>,
    summary: String,
}

struct PathSegmentAnalysis {
    segment_idx: usize,
    way_id1: u64,
    way_id2: u64,
    connected: bool,
    reason: String,
    details: Vec<String>,
}

impl DebugReport {
    fn generate(&self) -> String {
        let mut report = String::new();

        // Header
        report.push_str("Route Debug Report\n");
        report.push_str("=================\n\n");

        // Summary
        report.push_str(&format!("SUMMARY: {}\n\n", self.summary));

        // Basic information
        report.push_str(&format!(
            "Start Point: ({}, {})\n",
            self.start_point.x(),
            self.start_point.y()
        ));
        report.push_str(&format!(
            "End Point: ({}, {})\n",
            self.end_point.x(),
            self.end_point.y()
        ));
        report.push_str(&format!("Requested Way IDs: {:?}\n\n", self.way_ids));

        // Way availability
        if !self.ways_not_found.is_empty() {
            report.push_str(&format!("Ways Not Found: {:?}\n", self.ways_not_found));
        }

        // Way segment information
        report.push_str("Way Segments:\n");
        for (&way_id, segments) in &self.way_segments {
            report.push_str(&format!(
                "  Way ID {}: {} segments\n",
                way_id,
                segments.len()
            ));
            for (i, (segment, tile_id)) in segments.iter().enumerate() {
                report.push_str(&format!(
                    "    Segment #{}: ID {}, Tile {}, Highway {}, Oneway: {}, Connections: {}\n",
                    i,
                    segment.id,
                    tile_id,
                    segment.highway_type,
                    segment.is_oneway,
                    segment.connections.len()
                ));
            }
        }
        report.push('\n');

        // Path analysis
        report.push_str("Path Analysis:\n");
        for analysis in &self.path_analysis {
            report.push_str(&format!(
                "  Step {} (Way {} → {}): {}\n",
                analysis.segment_idx,
                analysis.way_id1,
                analysis.way_id2,
                if analysis.connected {
                    "CONNECTED"
                } else {
                    "NOT CONNECTED"
                }
            ));
            report.push_str(&format!("    Reason: {}\n", analysis.reason));

            if !analysis.details.is_empty() {
                report.push_str("    Details:\n");
                for detail in &analysis.details {
                    report.push_str(&format!("      - {}\n", detail));
                }
            }
        }

        report
    }
}
