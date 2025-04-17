use anyhow::{Result, anyhow, bail};
use chrono::{DateTime, Utc};
use core::f64;
use geo::{Closest, ClosestPoint, Haversine, LineString, algorithm::Distance};
use geo_types::Point;
use log::{debug, info, trace, warn};
use ordered_float::OrderedFloat;
use petgraph::prelude::DiGraphMap;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::{HashMap, HashSet, VecDeque},
    path::{Path, PathBuf},
    time::Instant,
};

use crate::{
    osm_preprocessing::{OSMProcessor, WaySegment},
    routing::{calculate_dynamic_transition_cost, check_segment_connectivity},
    tile_loader::TileLoader,
};

pub(crate) const MAX_DISTANCE_BETWEEN_POINT_AND_MATCHED_ROUTE_METER: f64 = 150.0;
pub(crate) const DISTANCE_THRESHOLD_FOR_COST_BIAS_METER: f64 = 10.0;

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
    pub split_windows_on_failure: bool,
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
            split_windows_on_failure: false,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub(crate) struct WindowTrace {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) segments: Vec<MatchedWaySegment>,
    pub(crate) bridge: bool,
    pub(crate) constraints: Vec<PointConstraint>, // Constraints used for this window
    pub(crate) used_constraints: bool,            // Was this window matched with constraints
    pub(crate) constraint_score: Option<f64>,     // Score with constraints
    pub(crate) unconstrained_score: Option<f64>,  // Score without constraints
    pub(crate) attempted_way_ids: Vec<u64>,       // Way IDs that were considered
    pub(crate) debug_notes: Vec<String>,          // Additional debugging notes
}

#[derive(Debug, Clone)]
pub struct PointConstraint {
    pub point_idx: usize,
    pub segment_id: u64,
    pub way_id: u64,   // OSM way ID for easier debugging
    pub distance: f64, // Distance to the point
}

/// Structure to represent a candidate segment for a GPS point
#[derive(Clone, Debug)]
pub(crate) struct SegmentCandidate {
    pub(crate) segment: WaySegment,
    pub(crate) distance: f64,          // Distance from GPS point to segment
    pub(crate) projection: Point<f64>, // Projected point on segment
    pub(crate) closest_node_idx: usize,
    pub(crate) cost: f64, // Overall cost (lower is better)
}

#[derive(Debug, Clone)]
pub(crate) struct PathfindingDebugInfo {
    pub start_point_idx: usize,
    pub end_point_idx: usize,
    pub start_candidates: Vec<SegmentCandidate>,
    pub end_candidates: Vec<SegmentCandidate>,
    pub constraints: Vec<(usize, u64)>,
    pub attempted_pairs: Vec<PathfindingAttempt>,
    pub constrained_candidates: HashMap<usize, Vec<u64>>,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct PathfindingAttempt {
    pub from_segment: u64,
    pub from_osm_way: u64,
    pub to_segment: u64,
    pub to_osm_way: u64,
    pub distance: f64,
    pub result: PathfindingResult,
}

#[derive(Debug, Clone)]
pub enum PathfindingResult {
    Success(Vec<MatchedWaySegment>, f64),
    TooFar(f64, f64),
    NoConnection,
    NoPathFound(String),
}

impl PathfindingResult {
    pub fn is_success(&self) -> bool {
        matches!(self, PathfindingResult::Success(_, _))
    }
}

// Scoring configuration
struct ScoringConfig {
    distance_weight: f64,
    std_dev_weight: f64,
    length_weight: f64,
    road_class_weight: f64,
    continuity_weight: f64,
    constraint_weight: f64,
    check_excessive_distance: bool,
}

#[derive(Debug, Clone)]
pub struct RouteMatchJob {
    // inputs
    pub(crate) gps_points: Vec<Point<f64>>,
    pub(crate) timestamps: Vec<DateTime<Utc>>,
    pub(crate) debug_way_ids: Option<Vec<u64>>,
    // state
    pub(crate) graph: RefCell<Option<DiGraphMap<u64, f64>>>,
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
                return Some((rank, candidate.cost));
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatchedWaySegment {
    /// The underlying way segment
    pub segment: WaySegment,
    /// Index of the starting node/coordinate to use (inclusive)
    pub entry_node: Option<usize>,
    /// Index of the ending node/coordinate to use (inclusive)
    pub exit_node: Option<usize>,
}

impl MatchedWaySegment {
    /// Create a new matched segment from an existing way segment
    pub fn new(
        segment: WaySegment,
        interim_start_idx: Option<usize>,
        interim_end_idx: Option<usize>,
    ) -> Self {
        Self {
            segment,
            entry_node: interim_start_idx,
            exit_node: interim_end_idx,
        }
    }

    /// Create a matched segment using the full way segment
    pub fn from_full_segment(segment: WaySegment) -> Self {
        Self {
            segment,
            entry_node: None,
            exit_node: None,
        }
    }

    /// Get the actual length of this matched segment
    pub fn length(&self) -> f64 {
        let start_idx = self.entry_node.unwrap_or(0);
        let end_idx = self.exit_node.unwrap_or(self.segment.coordinates.len() - 1);

        if start_idx >= end_idx
            || start_idx >= self.segment.coordinates.len()
            || end_idx >= self.segment.coordinates.len()
        {
            return 0.0;
        }

        let mut length = 0.0;
        for i in start_idx..end_idx {
            let p1 = Point::new(self.segment.coordinates[i].x, self.segment.coordinates[i].y);
            let p2 = Point::new(
                self.segment.coordinates[i + 1].x,
                self.segment.coordinates[i + 1].y,
            );
            length += Haversine.distance(p1, p2);
        }
        length
    }

    /// Get the actual coordinates of this matched segment
    pub fn coordinates(&self) -> Vec<geo::Coord<f64>> {
        let entry_idx = self.entry_node.unwrap_or(0);
        let exit_idx = self.exit_node.unwrap_or(self.segment.coordinates.len() - 1);

        if entry_idx >= self.segment.coordinates.len() || exit_idx >= self.segment.coordinates.len()
        {
            panic!(
                "Invalid segment indices: start={}, end={}, len={} for segment {}",
                entry_idx,
                exit_idx,
                self.segment.coordinates.len(),
                self.segment.id
            );
        }

        // For one-way roads, ensure forward traversal
        if self.segment.is_oneway && entry_idx > exit_idx {
            // For one-way roads, we should not traverse backward
            // Return forward path from start to end of segment
            self.segment.coordinates.to_vec()
        } else if entry_idx <= exit_idx {
            // Forward direction
            self.segment.coordinates[entry_idx..=exit_idx].to_vec()
        } else {
            if self.segment.is_oneway {
                panic!(
                    "Invalid entry {}  and exit idx {} for one way segment",
                    entry_idx, exit_idx
                );
            }
            // Reverse direction (only allowed for bidirectional roads)
            let mut reversed = Vec::with_capacity(entry_idx - exit_idx + 1);
            for i in (exit_idx..=entry_idx).rev() {
                reversed.push(self.segment.coordinates[i]);
            }
            reversed
        }
    }

    /// Check if this segment is traversed in the reverse direction
    pub fn is_reversed(&self) -> bool {
        if let (Some(start_idx), Some(end_idx)) = (self.entry_node, self.exit_node) {
            return start_idx > end_idx;
        }
        false
    }

    /// Get actual nodes for this matched segment
    pub fn nodes(&self) -> Vec<u64> {
        let start_idx = self.entry_node.unwrap_or(0);
        let end_idx = self.exit_node.unwrap_or(self.segment.nodes.len() - 1);

        if start_idx >= end_idx
            || start_idx >= self.segment.nodes.len()
            || end_idx >= self.segment.nodes.len()
        {
            return Vec::new();
        }

        self.segment.nodes[start_idx..=end_idx].to_vec()
    }

    /// Get the start node, considering interim start
    pub fn start_node(&self) -> Option<u64> {
        let start_idx = self.entry_node.unwrap_or(0);
        self.segment.nodes.get(start_idx).copied()
    }

    /// Get the end node, considering interim end
    pub fn end_node(&self) -> Option<u64> {
        let end_idx = self
            .exit_node
            .unwrap_or(self.segment.nodes.len().saturating_sub(1));
        self.segment.nodes.get(end_idx).copied()
    }

    /// Returns the centroid of the matched segment
    pub fn centroid(&self) -> Point<f64> {
        let coords = self.coordinates();

        if coords.is_empty() {
            // Fallback
            return self.segment.centroid();
        }

        let sum_x: f64 = coords.iter().map(|c| c.x).sum();
        let sum_y: f64 = coords.iter().map(|c| c.y).sum();
        let count = coords.len() as f64;

        Point::new(sum_x / count, sum_y / count)
    }

    /// Validate and adjust indices for one-way segments if needed
    pub fn validate_direction(&self) -> bool {
        if !self.segment.is_oneway {
            return true; // Bidirectional road, any direction is valid
        }

        // For one-way roads, entry_node must be <= exit_node to ensure forward travel
        if let (Some(entry), Some(exit)) = (self.entry_node, self.exit_node) {
            return entry <= exit; // This is the key check - traversal direction
        }

        true
    }
}

impl From<WaySegment> for MatchedWaySegment {
    fn from(segment: WaySegment) -> Self {
        MatchedWaySegment::from_full_segment(segment)
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
#[derive(Debug)]
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

    fn find_optimal_route(
        &mut self,
        job: &mut RouteMatchJob,
        enforce_constraints: bool,
        honor_timestamps: bool,
        constrained_points: &[(usize, u64)],
        max_matching_distance: f64,
    ) -> Result<Vec<MatchedWaySegment>> {
        let points = &job.gps_points;
        let timestamps = &job.timestamps;

        if points.len() < 2 || (honor_timestamps && points.len() != timestamps.len()) {
            return Err(anyhow!("Invalid points or timestamps"));
        }

        info!(
            "Finding optimal route for {} points with enforce_constraints={}",
            points.len(),
            enforce_constraints
        );
        let start_time = std::time::Instant::now();

        // Step 1: Load tiles covering the entire route area with buffer
        let trace_bbox = self.calculate_trace_bbox(points);
        let buffer = (max_matching_distance * 2.0) / 111_000.0; // Convert meters to approx degrees

        info!("Loading tiles for route area");
        let loaded_tiles = self.tile_loader.load_tile_range(
            trace_bbox,
            buffer,
            self.config.max_tiles_per_depth,
        )?;

        // Step 2: Build road network graph for path finding if not already built
        if job.graph.borrow().is_none() {
            info!("Building road network graph");
            let (graph, segment_map) = self.build_road_network(&loaded_tiles)?;
            job.loaded_tiles.replace(loaded_tiles.clone());
            job.graph.replace(Some(graph));
            job.segment_map.replace(segment_map);
        }

        // Step 3: Find candidate segments for each point
        let mut all_candidates = Vec::with_capacity(points.len());
        let mut nearest_projections = Vec::with_capacity(points.len());

        for (i, &point) in points.iter().enumerate() {
            // For the first and last point, use node-based matching if doing route matching
            let candidates = if enforce_constraints && (i == 0 || i == points.len() - 1) {
                self.find_node_based_candidate_segments(
                    point,
                    &loaded_tiles,
                    max_matching_distance * 1.2, // Slightly larger radius for endpoints
                    self.config.max_candidates_per_point,
                )?
            } else {
                self.find_candidate_segments_for_point(
                    point,
                    &loaded_tiles,
                    max_matching_distance,
                    self.config.max_candidates_per_point,
                    true, // Allow extension
                )?
            };

            if candidates.is_empty() {
                return Err(anyhow!("No candidate segments found for point {}", i));
            }

            // Store best projection for this point
            nearest_projections.push(candidates[0].projection);
            all_candidates.push(candidates);
        }

        // Store candidates in job if needed for route matching
        if enforce_constraints {
            job.all_candidates.replace(all_candidates.clone());
        }

        // Step 4: Process constrained points
        let mut constrained_candidates: HashMap<usize, Vec<u64>> = HashMap::new();

        if enforce_constraints {
            for &(point_idx, segment_id) in constrained_points {
                if point_idx >= points.len() {
                    continue; // Skip invalid points
                }

                // Find candidates that match the segment ID
                let segment_map = job.segment_map.borrow();
                if let Some(segment) = segment_map.get(&segment_id) {
                    let osm_way_id = segment.osm_way_id;

                    // Keep track of all candidates for this point with the same OSM way ID
                    let matching_candidates: Vec<u64> = all_candidates[point_idx]
                        .iter()
                        .filter(|cand| cand.segment.osm_way_id == osm_way_id)
                        .map(|cand| cand.segment.id)
                        .collect();

                    if !matching_candidates.is_empty() {
                        constrained_candidates.insert(point_idx, matching_candidates);
                    }
                }
            }
        }

        // Step 5: Find paths between consecutive points or use sliding window approach
        let matched_segments = if enforce_constraints {
            // Use sliding window approach for route matching
            self.build_route_with_sliding_window(job)?
        } else {
            // Use direct point-to-point approach for route planning
            self.build_direct_route(job, &all_candidates)?
        };

        info!(
            "Optimal route found in {:.2?} with {} segments",
            start_time.elapsed(),
            matched_segments.len()
        );

        Ok(matched_segments)
    }

    /// Build a direct route between points (for route planning)
    fn build_direct_route(
        &mut self,
        job: &RouteMatchJob,
        all_candidates: &[Vec<SegmentCandidate>],
    ) -> Result<Vec<MatchedWaySegment>> {
        let points = &job.gps_points;
        let timestamps = &job.timestamps;
        let honor_timestamps = points.len() == timestamps.len();
        let mut matched_segments: Vec<MatchedWaySegment> = Vec::new();
        let mut used_segments = HashSet::new();

        for i in 0..points.len() - 1 {
            let from_candidates = &all_candidates[i];
            let to_candidates = &all_candidates[i + 1];

            if from_candidates.is_empty() || to_candidates.is_empty() {
                return Err(anyhow!("Missing candidates for points {} or {}", i, i + 1));
            }

            // Try multiple candidates to find a valid path
            let mut path_found = false;

            // Try at most 3 candidates for each endpoint to limit processing time
            for from_candidate in from_candidates.iter().take(3) {
                if path_found {
                    break;
                }

                for to_candidate in to_candidates.iter().take(3) {
                    // Skip if same segment for multi-point routes
                    if from_candidate.segment.id == to_candidate.segment.id && points.len() > 2 {
                        continue;
                    }

                    // Calculate max distance
                    let max_distance = if honor_timestamps && i < timestamps.len() - 1 {
                        // Time-based approach
                        let time_diff = (timestamps[i + 1] - timestamps[i]).num_seconds() as f64;
                        if time_diff > 0.0 {
                            // Using high average speed (40 m/s = 144 km/h)
                            (time_diff * 40.0).max(5000.0)
                        } else {
                            5000.0 // Default 5km
                        }
                    } else {
                        // Distance-based approach
                        let direct_distance = Haversine.distance(points[i], points[i + 1]);
                        (direct_distance * 2.5).max(5000.0) // At least 5km
                    };

                    // Try to find a path - now getting MatchedWaySegment objects directly
                    match self.find_path_with_distance_limit(
                        job,
                        from_candidate.segment.id,
                        from_candidate.closest_node_idx,
                        to_candidate.segment.id,
                        &used_segments,
                        max_distance,
                    ) {
                        Ok((_cost, path_segments)) => {
                            // Add new matched segments to our route
                            for segment in path_segments {
                                if !used_segments.contains(&segment.segment.id) {
                                    used_segments.insert(segment.segment.id);
                                    matched_segments.push(segment);
                                }
                            }
                            path_found = true;
                            break;
                        }
                        Err(_) => {
                            // Try next candidate pair
                            continue;
                        }
                    }
                }
            }

            // If no path found with any candidate pair, try without used_segments restriction
            if !path_found {
                let from_candidate = &from_candidates[0];
                let to_candidate = &to_candidates[0];

                let direct_distance = Haversine.distance(points[i], points[i + 1]);
                let max_distance = (direct_distance * 3.0).max(5000.0); // More generous limit

                match self.find_path_with_distance_limit(
                    job,
                    from_candidate.segment.id,
                    from_candidate.closest_node_idx,
                    to_candidate.segment.id,
                    &HashSet::new(), // Empty set - ignore used segments
                    max_distance,
                ) {
                    Ok((_cost, path_segments)) => {
                        // Add new matched segments to our route
                        for matched in path_segments {
                            if !used_segments.contains(&matched.segment.id) {
                                used_segments.insert(matched.segment.id);
                                matched_segments.push(matched);
                            }
                        }
                    }
                    Err(e) => {
                        // Last attempt failed
                        return Err(anyhow!(
                            "Failed to find path between points {} and {}: {}",
                            i,
                            i + 1,
                            e
                        ));
                    }
                }
            }
        }

        if matched_segments.is_empty() {
            return Err(anyhow!("Failed to build a valid route"));
        }

        Ok(matched_segments)
    }

    /// Modified plan_route function that uses the unified approach
    pub fn plan_route(
        &mut self,
        start_point: Point<f64>,
        end_point: Point<f64>,
        via_points: &[Point<f64>],
    ) -> Result<(serde_json::Value, Vec<MatchedWaySegment>)> {
        info!(
            "Planning route from {:?} to {:?} with {} via points",
            start_point,
            end_point,
            via_points.len()
        );

        // Create all points including start, via, and end
        let mut all_points = vec![start_point];
        all_points.extend_from_slice(via_points);
        all_points.push(end_point);

        // Create a job with complete context for routing
        // We'll use the same setup as match_trace for consistency
        let mut job = RouteMatchJob::new(
            all_points.clone(),
            vec![Utc::now(); all_points.len()], // Dummy timestamps
            None,
        );

        // Turn on tracing for better debugging
        job.activate_tracing();

        // Find route using the sliding window approach (same as match_trace)
        let matched_segments = self.find_optimal_route(
            &mut job,
            true,  // Enforce constraints to ensure we pass through all points
            false, // Don't honor timestamps (we're just planning, not matching historical data)
            &[],   // No initial constraints
            self.config.max_matching_distance * 2.0, // Double the distance for planning
        )?;

        // Generate GeoJSON for the route
        let geojson = self.route_to_geojson(&matched_segments)?;

        info!(
            "Route planning completed with {} segments",
            matched_segments.len()
        );

        Ok((geojson, matched_segments))
    }

    /// Modified match_trace function that uses the unified approach
    pub fn match_trace(&mut self, job: &mut RouteMatchJob) -> Result<Vec<MatchedWaySegment>> {
        if job.gps_points.len() < 2 || job.gps_points.len() != job.timestamps.len() {
            return Err(anyhow!("Invalid GPS points or timestamps"));
        }

        info!(
            "Starting route matching for {} points",
            job.gps_points.len()
        );

        // Find optimal route with constraints and timestamps
        let matched_segments = self.find_optimal_route(
            job,
            true, // Enforce constraints
            true, // Honor timestamps
            &[],  // No initial constraints
            self.config.max_matching_distance,
        )?;

        // Debug information about specified way IDs
        if let Some(way_ids) = &job.debug_way_ids {
            self.debug_way_ids(&matched_segments, way_ids, &job.loaded_tiles.borrow())?;
        }

        Ok(matched_segments)
    }

    // Completely replace the build_road_network method to use tile graphs
    pub(crate) fn build_road_network(
        &mut self,
        loaded_tiles: &HashSet<String>,
    ) -> Result<(DiGraphMap<u64, f64>, HashMap<u64, WaySegment>)> {
        let start_time = Instant::now();

        // Create a new graph that will combine data from all loaded tiles
        let mut graph = DiGraphMap::new();
        let mut segment_map = HashMap::new();

        // First pass: Add all segments and their local graph edges to the graph
        for tile_id in loaded_tiles {
            // Load tile and add its segments to the map
            let tile = self.tile_loader.load_tile(tile_id)?;

            // Convert optimized segments to regular segments and add to segment map
            for opt_segment in &tile.road_segments {
                let segment = opt_segment.to_way_segment(&tile.metadata);
                segment_map.insert(segment.id, segment);
            }

            // Add edges from the tile's local graph to the combined graph
            if let Some(tile_graph) = &tile.tile_graph {
                for &(from, to, cost) in &tile_graph.edges {
                    // Verify that both segments are in our segment map
                    if segment_map.contains_key(&from) && segment_map.contains_key(&to) {
                        graph.add_edge(from, to, cost);
                    }
                }
            }
        }

        // Second pass: Add connections between segments in different tiles
        // These connections would not have been included in any single tile's graph
        for (&segment_id, segment) in &segment_map {
            for &connected_id in &segment.connections {
                // Skip if we've already added this edge (e.g., from a tile's local graph)
                if graph.contains_edge(segment_id, connected_id) {
                    continue;
                }

                // Skip if the connected segment isn't in our loaded tiles
                if !segment_map.contains_key(&connected_id) {
                    continue;
                }

                // This is a cross-tile connection that we need to add
                let connected_segment = &segment_map[&connected_id];

                // Find common nodes to determine connection cost
                let common_nodes: Vec<u64> = segment
                    .nodes
                    .iter()
                    .filter(|&&n| connected_segment.nodes.contains(&n))
                    .cloned()
                    .collect();

                // Only the first common node is considered for now
                let common_node = common_nodes
                    .first()
                    .expect("No common node between connected segments");

                // Check directionality before adding the edge
                let from_segment = &segment;
                let to_segment = connected_segment;

                let from_connection_node_idx =
                    segment.nodes.iter().position(|n| n == common_node).unwrap();
                let to_segment_connection_node_idx = to_segment
                    .nodes
                    .iter()
                    .position(|n| n == common_node)
                    .unwrap();

                OSMProcessor::add_edges(
                    from_segment,
                    from_connection_node_idx,
                    to_segment,
                    to_segment_connection_node_idx,
                    &mut graph,
                );
            }
        }

        debug!(
            "Built combined road network graph with {} nodes and {} edges in {:?}",
            graph.node_count(),
            graph.edge_count(),
            start_time.elapsed()
        );

        Ok((graph, segment_map))
    }

    /// Find all candidate segments within accuracy range for each GPS point
    pub(crate) fn find_all_candidate_segments(
        &mut self,
        job: &RouteMatchJob,
        loaded_tiles: &HashSet<String>,
    ) -> Result<()> {
        // Set base parameters
        let max_distance = self.config.max_matching_distance;
        let max_candidates = self.config.max_candidates_per_point;
        let allow_extension = true; // Allow extended search for points with no candidates

        let mut all_candidates = Vec::with_capacity(job.gps_points.len());

        // Process each GPS point
        for (i, &point) in job.gps_points.iter().enumerate() {
            // Find candidates for this point
            let candidates = self.find_candidate_segments_for_point(
                point,
                loaded_tiles,
                max_distance,
                max_candidates,
                allow_extension,
            )?;

            debug!("Found {} candidates for GPS point {}", candidates.len(), i);
            all_candidates.push(candidates);
        }

        // Store candidates in job
        job.all_candidates.replace(all_candidates);

        // Generate debug visualization if tracing enabled
        if job.tracing {
            for i in 0..job.gps_points.len() {
                let mut pc_geojson = job.point_candidates_geojson.borrow_mut();
                pc_geojson.push(self.generate_point_candidates_geojson(job, i)?);
            }
        }

        Ok(())
    }

    /// New function to find candidate segments based on node proximity rather than
    /// projection to the segment line
    pub fn find_node_based_candidate_segments(
        &mut self,
        point: Point<f64>,
        loaded_tiles: &HashSet<String>,
        max_distance: f64,
        max_candidates: usize,
    ) -> Result<Vec<SegmentCandidate>> {
        let mut node_based_candidates = Vec::new();

        // Collect all available segments from loaded tiles
        for tile_id in loaded_tiles {
            // Load tile and process segments
            let segments = self.tile_loader.get_all_segments_from_tile(tile_id)?;

            for segment in segments {
                if let Some((projection, distance, closest_node_idx)) =
                    self.project_point_to_segment(point, &segment)
                {
                    // Check if the closest node is within max distance
                    if distance <= max_distance {
                        // Calculate cost (lower is better)
                        // Use road type to influence cost - motorways get a bonus
                        let base_cost = distance / (max_distance / 2.0);
                        let road_type_factor = match segment.highway_type.as_str() {
                            "motorway" => 0.5,
                            "motorway_link" => 0.6,
                            "trunk" => 0.7,
                            "trunk_link" => 0.8,
                            "primary" => 0.9,
                            _ => 1.0,
                        };

                        let cost = base_cost * road_type_factor;

                        node_based_candidates.push(SegmentCandidate {
                            segment,
                            distance,
                            projection,
                            closest_node_idx,
                            cost,
                        });
                    }
                }
            }
        }

        // Sort candidates by cost
        node_based_candidates
            .sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(Ordering::Equal));

        // Limit number of candidates
        if node_based_candidates.len() > max_candidates {
            node_based_candidates.truncate(max_candidates);
        }

        Ok(node_based_candidates)
    }

    fn build_route_with_sliding_window(
        &mut self,
        job: &mut RouteMatchJob,
    ) -> Result<Vec<MatchedWaySegment>> {
        let window_size = 7;
        // Process data in overlapping windows
        let step_size = window_size / 2; // More overlap for better correction

        info!(
            "Using window size {} with step size {}",
            window_size, step_size
        );

        let mut complete_route = Vec::new();

        // Track which segments were chosen for each GPS point in previous windows
        let mut previous_point_segments: HashMap<usize, u64> = HashMap::new();

        // Calculate the windows in advance to handle the last window specially
        let mut windows = VecDeque::new();
        let mut window_start = 0;

        while window_start < job.gps_points.len() {
            let mut window_end = (window_start + window_size - 1).min(job.gps_points.len() - 1);

            // Check if this would leave a small last window
            let points_remaining = job.gps_points.len() - window_end - 1;
            if points_remaining > 0 && points_remaining < step_size {
                // This is the second-to-last window, make it larger to include all remaining points
                window_end = job.gps_points.len() - 1;
                windows.push_back((window_start, window_end));
                break; // No more windows needed
            }

            windows.push_back((window_start, window_end));
            window_start += step_size;
        }

        // Special case: If the last window has only one point and it's not already
        // included in the previous window, merge it with the previous window
        if windows.len() >= 2 {
            // Explicitly handle the last window to avoid borrowing issues
            let last_idx = windows.len() - 1;
            let last_start = windows[last_idx].0;
            let last_end = windows[last_idx].1;

            // Check if it's a single-point window
            if last_start == last_end {
                // Remove the last window
                windows.pop_back();

                // Extend the previous window (now the last) to include the single point
                let prev_idx = windows.len() - 1;
                windows[prev_idx].1 = last_start;
            }
        }

        debug!("Created {} windows for processing", windows.len());
        for (i, (start, end)) in windows.iter().enumerate() {
            trace!("Window {}: points {}-{}", i + 1, start, end);
        }

        // Initialize the window_trace array with default values for all windows
        if job.tracing {
            let mut window_traces = job.window_trace.borrow_mut();
            window_traces.resize_with(windows.len(), WindowTrace::default);
        }

        // Track windows with failed constraints for debugging
        let mut failed_windows = Vec::new();
        let mut debug_infos = Vec::new();
        let mut i: i32 = -1;

        // Process each window
        while !windows.is_empty() {
            i += 1;
            let (window_start, window_end) = windows.pop_front().unwrap();

            let window_index = if i < 0 { 0 } else { i as usize };

            info!(
                "Processing window {} from point {} to {} (of {})",
                window_index + 1,
                window_start,
                window_end,
                job.gps_points.len() - 1
            );

            // Create a list of previously matched point constraints for this window
            let mut overlapping_points: Vec<(usize, u64)> = (window_start..=window_end)
                .filter_map(|idx| {
                    previous_point_segments
                        .get(&idx)
                        .map(|&seg_id| (idx, seg_id))
                })
                .collect();
            // Do not include the last matched point of the previous window as a constraint as it
            // has a high(er) chance of being wrong
            overlapping_points.pop();

            // Is this the last window?
            let is_last_window = windows.is_empty();
            // Add the best closest matching segment to the end node as a constraint
            if is_last_window {
                let mut distance = f64::INFINITY;
                let mut best_candidate = None;
                let candidates = job.all_candidates.borrow();
                for candidate in candidates.get(window_end).unwrap() {
                    if candidate.distance < distance {
                        best_candidate = Some(candidate.segment.id);
                        distance = candidate.distance;
                    }
                }
                if best_candidate.is_none() {
                    bail!("No candidate found for window")
                }
                overlapping_points.push((window_end, best_candidate.unwrap()));
            }

            // Record current constraints for debugging purposes
            let mut constraint_details = Vec::new();
            for &(point_idx, segment_id) in &overlapping_points {
                // Find the segment in the segment map to get the OSM way ID
                if let Some(segment) = job.segment_map.borrow().get(&segment_id) {
                    let mut distance = f64::MAX;
                    // Find the actual distance from the point to the segment
                    if point_idx < job.gps_points.len() {
                        let point = job.gps_points[point_idx];
                        let projection = self.project_point_to_segment(point, segment).unwrap();
                        distance = projection.1;
                    }

                    constraint_details.push(PointConstraint {
                        point_idx,
                        segment_id,
                        way_id: segment.osm_way_id,
                        distance,
                    });
                }
            }

            // Find optimal path for this window, passing previous point constraints
            // and indicating if this is the last window
            let window_route = self.find_a_star_path_for_window(
                job,
                window_start,
                window_end,
                &overlapping_points,
            )?;

            // Track attempted way IDs during matching
            let mut attempted_way_ids = HashSet::new();
            for segment in &window_route {
                attempted_way_ids.insert(segment.segment.osm_way_id);
            }

            // Calculate actual scores for constrained route
            let constrained_score = if !window_route.is_empty() {
                Some(self.calculate_comprehensive_score(
                    &window_route,
                    &job.gps_points[window_start..=window_end],
                    &job.timestamps[window_start..=window_end],
                    0.0,             // We don't have the raw path cost here
                    &HashMap::new(), // No need for additional constraints
                    None,
                ))
            } else {
                None
            };

            if window_route.is_empty() {
                // Track this window for detailed debugging
                failed_windows.push((
                    window_index,
                    window_start,
                    window_end,
                    overlapping_points.clone(),
                ));

                // Attempt window splitting before falling back to unconstrained
                let current_window_size = window_end - window_start + 1;
                if self.config.split_windows_on_failure
                    && current_window_size > step_size
                    && current_window_size > 4
                {
                    info!(
                        "Splitting failed window {}-{} into smaller windows",
                        window_start, window_end
                    );

                    // Calculate split point (halving the window)
                    let mid_point = window_start + (current_window_size / 2);

                    // Create new sub-windows
                    let first_half = (window_start, mid_point);
                    let second_half = (mid_point - 1, window_end);

                    // Push new split windows in front
                    windows.push_front(second_half);
                    windows.push_front(first_half);

                    if job.tracing {
                        let mut window_traces = job.window_trace.borrow_mut();
                        window_traces.push(WindowTrace::default());
                    }

                    // Skip further processing of the original failed window
                    i -= 1;
                    continue;
                }

                // Try to find a path without constraints if no valid path with constraints
                info!("No valid path found with constraints. Attempting unconstrained matching...");
                let unconstrained_route =
                    self.find_a_star_path_for_window(job, window_start, window_end, &[])?;

                // Calculate unconstrained score
                let unconstrained_score = if !unconstrained_route.is_empty() {
                    Some(self.calculate_comprehensive_score(
                        &unconstrained_route,
                        &job.gps_points[window_start..=window_end],
                        &job.timestamps[window_start..=window_end],
                        0.0, // We don't have the raw path cost
                        &HashMap::new(),
                        None,
                    ))
                } else {
                    None
                };

                // Perform detailed debugging for this failed window
                if job.tracing {
                    let debug_info = self.debug_constrained_window_failure(
                        job,
                        window_start,
                        window_end,
                        &overlapping_points,
                    );
                    debug_infos.push((window_index, debug_info.clone()));

                    // Store debug information
                    job.add_explanation(format!(
                        "\nWindow {} (points {}-{}) failed with constraints: {}",
                        window_index + 1,
                        window_start,
                        window_end,
                        debug_info.reason
                    ));
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
                            window_index + 1
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
                            // Update the window trace at the current index
                            let mut window_traces = job.window_trace.borrow_mut();
                            window_traces[window_index] = WindowTrace {
                                start: window_start,
                                end: window_end,
                                segments: unconstrained_route.clone(),
                                bridge: true, // Mark as unconstrained bridge
                                constraints: constraint_details,
                                used_constraints: false,
                                constraint_score: constrained_score,
                                unconstrained_score,
                                attempted_way_ids: attempted_way_ids.into_iter().collect(),
                                debug_notes: vec![
                                    format!(
                                        "Window {}: Fell back to unconstrained path due to failed constraints",
                                        window_index + 1
                                    ),
                                    format!(
                                        "Connecting to previous window at overlap point {}",
                                        overlap_gps_idx
                                    ),
                                ],
                            };
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

                        continue;
                    }
                }

                // Use unconstrained route with our window fusion logic
                self.fuse_window_into_route(
                    &mut complete_route,
                    unconstrained_route.clone(),
                    window_start,
                    window_end,
                    is_last_window,
                    &job.gps_points,
                )?;

                // Update the point-to-segment mapping for the new segments
                self.update_point_segment_mapping(
                    &mut previous_point_segments,
                    window_start,
                    window_end,
                    &unconstrained_route,
                    &job.gps_points,
                );

                // Update the window trace for debugging
                if job.tracing {
                    let mut window_traces = job.window_trace.borrow_mut();
                    window_traces[window_index] = WindowTrace {
                        start: window_start,
                        end: window_end,
                        segments: unconstrained_route.clone(),
                        bridge: true,
                        constraints: constraint_details,
                        used_constraints: false,
                        constraint_score: constrained_score,
                        unconstrained_score,
                        attempted_way_ids: attempted_way_ids.into_iter().collect(),
                        debug_notes: vec![
                            format!(
                                "Window {}: Fell back to unconstrained path due to failed constraints",
                                window_index + 1
                            ),
                            format!("Is last window: {}", is_last_window),
                        ],
                    };
                }
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

                // Use our window fusion logic
                self.fuse_window_into_route(
                    &mut complete_route,
                    window_route.clone(),
                    window_start,
                    window_end,
                    is_last_window,
                    &job.gps_points,
                )?;

                // Update the point-to-segment mapping for the new segments
                self.update_point_segment_mapping(
                    &mut previous_point_segments,
                    window_start,
                    window_end,
                    &window_route,
                    &job.gps_points,
                );

                // Update the window trace for debugging
                if job.tracing {
                    let mut window_traces = job.window_trace.borrow_mut();
                    window_traces[window_index] = WindowTrace {
                        start: window_start,
                        end: window_end,
                        segments: window_route.clone(),
                        bridge: false,
                        constraints: constraint_details,
                        used_constraints: true,
                        constraint_score: constrained_score,
                        unconstrained_score: None,
                        attempted_way_ids: attempted_way_ids.into_iter().collect(),
                        debug_notes: vec![
                            format!(
                                "Window {}: Successfully matched with constraints",
                                window_index + 1
                            ),
                            format!("Is last window: {}", is_last_window),
                        ],
                    };
                }
            }
        }

        // After processing all windows, create a comprehensive debug report for failed windows
        if !failed_windows.is_empty() && job.tracing {
            job.add_explanation("\n==== DETAILED PATH FINDING FAILURE ANALYSIS ====".to_string());

            // Sort failed windows by index
            failed_windows.sort_by_key(|(idx, _, _, _)| *idx);

            for (window_idx, window_start, window_end, constraints) in &failed_windows {
                // Find the matching debug info
                if let Some((_, debug_info)) = debug_infos.iter().find(|(idx, _)| idx == window_idx)
                {
                    job.add_explanation(format!(
                        "\nWindow #{} (Points {}-{}): {}",
                        window_idx + 1,
                        window_start,
                        window_end,
                        debug_info.reason
                    ));

                    // Report constraint details
                    if !constraints.is_empty() {
                        job.add_explanation("Constraint points:".to_string());
                        for &(point_idx, segment_id) in constraints {
                            let segment_map = job.segment_map.borrow();
                            if let Some(segment) = segment_map.get(&segment_id) {
                                job.add_explanation(format!(
                                    "  Point {} → Segment {} (OSM way {})",
                                    point_idx, segment_id, segment.osm_way_id
                                ));
                            }
                        }
                    }

                    // Report on start/end candidates
                    job.add_explanation(format!(
                        "Start point {} has {} candidates",
                        window_start,
                        debug_info.start_candidates.len()
                    ));
                    job.add_explanation(format!(
                        "End point {} has {} candidates",
                        window_end,
                        debug_info.end_candidates.len()
                    ));

                    // Report on path attempts
                    let success_count = debug_info
                        .attempted_pairs
                        .iter()
                        .filter(|a| matches!(a.result, PathfindingResult::Success(_, _)))
                        .count();

                    let too_far_count = debug_info
                        .attempted_pairs
                        .iter()
                        .filter(|a| matches!(a.result, PathfindingResult::TooFar(_, _)))
                        .count();

                    let no_conn_count = debug_info
                        .attempted_pairs
                        .iter()
                        .filter(|a| matches!(a.result, PathfindingResult::NoConnection))
                        .count();

                    job.add_explanation(format!(
                        "Pathfinding attempts: {} total ({} successful, {} too far, {} no connection)",
                        debug_info.attempted_pairs.len(),
                        success_count,
                        too_far_count,
                        no_conn_count
                    ));

                    // List a few successful paths if available (for debugging why they weren't used)
                    let successful_paths: Vec<_> = debug_info
                        .attempted_pairs
                        .iter()
                        .filter(|a| matches!(a.result, PathfindingResult::Success(_, _)))
                        .collect();

                    if !successful_paths.is_empty() {
                        job.add_explanation(
                            "Example successful paths that couldn't be used:".to_string(),
                        );
                        for (i, path) in successful_paths.iter().take(2).enumerate() {
                            if let PathfindingResult::Success(segments, cost) = &path.result {
                                job.add_explanation(format!(
                                    "  Path #{}: {} segments, cost: {:.2}",
                                    i + 1,
                                    segments.len(),
                                    cost
                                ));

                                // List the way IDs in this path
                                let way_sequence: Vec<_> =
                                    segments.iter().map(|s| s.segment.osm_way_id).collect();

                                job.add_explanation(format!(
                                    "    Way sequence: {:?}",
                                    way_sequence
                                ));

                                // Check for any issues with these segments vs constraints
                                for &(point_idx, segment_id) in constraints {
                                    let segment_map = job.segment_map.borrow();
                                    if let Some(constraint_segment) = segment_map.get(&segment_id) {
                                        let constraint_way_id = constraint_segment.osm_way_id;

                                        // Check if this way ID appears in the path
                                        if !way_sequence.contains(&constraint_way_id) {
                                            job.add_explanation(format!(
                                                "    Missing constraint: Way {} for point {}",
                                                constraint_way_id, point_idx
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
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
        segments: &[MatchedWaySegment],
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

            for matched_segment in segments {
                let coords = matched_segment.coordinates();
                if coords.is_empty() {
                    continue;
                }

                // Create a temporary line string from actual coords
                let line = LineString::from(coords);
                let closest = line.closest_point(&point);

                let projection = match closest {
                    Closest::SinglePoint(p) => p,
                    _ => matched_segment.centroid(), // Use centroid as fallback
                };

                let distance = Haversine.distance(point, projection);

                if distance < best_distance {
                    best_distance = distance;
                    best_segment_id = matched_segment.segment.id;
                }
            }

            if best_segment_id != 0 && best_distance <= 150.0 {
                mapping.insert(point_idx, best_segment_id);
            }
        }
    }

    /// Find which segment in a route matches a GPS point
    fn find_segment_for_gps_point(
        &self,
        route: &[MatchedWaySegment],
        point: Point<f64>,
    ) -> Option<usize> {
        if route.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_distance = f64::MAX;

        for (i, matched_segment) in route.iter().enumerate() {
            // Get actual coordinates
            let coords = matched_segment.coordinates();
            if coords.is_empty() {
                continue;
            }

            // Create a temporary line string from actual coords
            let line = LineString::from(coords);
            let closest = line.closest_point(&point);

            let projection = match closest {
                Closest::SinglePoint(p) => p,
                _ => point, // Fallback
            };

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

    fn find_a_star_path_for_window(
        &mut self,
        job: &RouteMatchJob,
        window_start: usize,
        window_end: usize,
        constraints: &[(usize, u64)],
    ) -> Result<Vec<MatchedWaySegment>> {
        // Existing implementation remains the same until path finding failure

        // Get window data
        let window_points = &job.gps_points[window_start..=window_end];
        let window_timestamps = &job.timestamps[window_start..=window_end];
        let window_candidates = &job.all_candidates.borrow()[window_start..=window_end];

        // Validation
        if window_points.is_empty() || window_candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Create a map from point index in the original array to its relative position in the window
        let point_idx_map: HashMap<usize, usize> = (window_start..=window_end)
            .enumerate()
            .map(|(window_pos, original_idx)| (original_idx, window_pos))
            .collect();

        // Process constraints: Find candidates that match the constraint segment IDs
        let mut constrained_candidates: HashMap<usize, Vec<&SegmentCandidate>> = HashMap::new();

        for &(point_idx, segment_id) in constraints {
            if point_idx < window_start || point_idx > window_end {
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
            warn!(
                "None of the {} constraints could be matched in this window",
                constraints.len()
            );
            return Ok(Vec::new()); // Signal that we need to use the unconstrained version
        }

        // Get first candidates based on constraints
        let first_candidates = if let Some(window_pos) = point_idx_map.get(&window_start) {
            if let Some(constrained) = constrained_candidates.get(window_pos) {
                constrained.iter().map(|&c| c.clone()).collect::<Vec<_>>()
            } else {
                window_candidates[0].clone()
            }
        } else {
            window_candidates[0].clone()
        };

        // Get last candidates
        let last_candidates = {
            // Regular candidates or constrained candidates
            if let Some(window_pos) = point_idx_map.get(&window_end) {
                if let Some(constrained) = constrained_candidates.get(window_pos) {
                    constrained.iter().map(|&c| c.clone()).collect::<Vec<_>>()
                } else {
                    window_candidates[window_candidates.len() - 1].clone()
                }
            } else {
                window_candidates[window_candidates.len() - 1].clone()
            }
        };

        if first_candidates.is_empty() || last_candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Track best route
        let mut best_route = Vec::new();
        let mut best_score = f64::MAX;
        let mut attempted_way_ids = HashSet::new();

        // Find paths, honoring constraints
        for first_candidate in &first_candidates {
            for last_candidate in &last_candidates {
                // Skip if same segment for multi-point windows
                if first_candidate.segment.id == last_candidate.segment.id
                    && window_points.len() > 1
                {
                    continue;
                }

                // Track attempted way IDs for debugging
                attempted_way_ids.insert(first_candidate.segment.osm_way_id);
                attempted_way_ids.insert(last_candidate.segment.osm_way_id);

                // Find path between candidates using simplified metrics
                // Now we get MatchedWaySegment objects directly
                match self.find_path_with_distance_limit(
                    job,
                    first_candidate.segment.id,
                    first_candidate.closest_node_idx,
                    last_candidate.segment.id,
                    &HashSet::new(),
                    Self::max_route_length_from_points(window_points, window_timestamps),
                ) {
                    Ok((path_cost, matched_path)) => {
                        // Add all way IDs from path to attempted ways
                        for segment in &matched_path {
                            attempted_way_ids.insert(segment.segment.osm_way_id);
                        }

                        // Calculate comprehensive score using matched segments
                        let score = self.calculate_comprehensive_score(
                            &matched_path,
                            window_points,
                            window_timestamps,
                            path_cost,
                            &constrained_candidates,
                            None,
                        );

                        if score < best_score {
                            best_route = matched_path;
                            best_score = score;
                        }
                    }
                    Err(_) => continue,
                }
            }
        }

        if best_route.is_empty() {
            return Ok(best_route);
        }

        // Check if the best route satisfies critical constraints
        let route_validity =
            self.check_for_excessive_point_match_distance(&best_route, window_points);

        if !route_validity.is_empty() && !constraints.is_empty() {
            // Generate detailed debugging info when route not found
            if job.tracing {
                for (point_idx, distance) in route_validity {
                    job.add_explanation(format!(
                        "Segment for point {} is too far away: {}",
                        window_start + point_idx,
                        distance,
                    ));
                }

                let debug_info = self.debug_constrained_window_failure(
                    job,
                    window_start,
                    window_end,
                    constraints,
                );

                // Add detailed explanation to the job
                job.add_explanation(format!(
                    "Window {}-{} failed to find valid route with constraints. Reason: {}",
                    window_start, window_end, debug_info.reason
                ));

                // Log detailed information about attempted paths
                let segment_map = job.segment_map.borrow();
                for (i, attempt) in debug_info.attempted_pairs.iter().enumerate() {
                    let from_seg = segment_map.get(&attempt.from_segment).map_or_else(
                        || format!("Unknown ({})", attempt.from_segment),
                        |s| format!("{} (OSM: {})", s.id, s.osm_way_id),
                    );

                    let to_seg = segment_map.get(&attempt.to_segment).map_or_else(
                        || format!("Unknown ({})", attempt.to_segment),
                        |s| format!("{} (OSM: {})", s.id, s.osm_way_id),
                    );

                    let result_str = match &attempt.result {
                        PathfindingResult::Success(path, cost) => {
                            format!("SUCCESS (cost: {:.2}, {} segments)", cost, path.len())
                        }
                        PathfindingResult::TooFar(max_dist, actual_dist) => format!(
                            "TOO FAR (max: {:.2}m, actual: {:.2}m)",
                            max_dist, actual_dist
                        ),
                        PathfindingResult::NoConnection => "NO CONNECTION".to_string(),
                        PathfindingResult::NoPathFound(reason) => format!("FAILED: {}", reason),
                    };

                    job.add_explanation(format!(
                        "Path attempt #{}: {} → {} ({:.2}m apart): {}",
                        i + 1,
                        from_seg,
                        to_seg,
                        attempt.distance,
                        result_str
                    ));
                }

                // Add information about each constraint
                for &(point_idx, segment_id) in constraints {
                    if let Some(segment) = segment_map.get(&segment_id) {
                        let point = if point_idx < job.gps_points.len() {
                            job.gps_points[point_idx]
                        } else {
                            continue;
                        };

                        // Calculate distance to constraint point
                        let (_, distance, _) =
                            self.project_point_to_segment(point, segment).unwrap();

                        job.add_explanation(format!(
                            "Constraint at point {}: Segment {} (OSM way {}), distance: {:.2}m",
                            point_idx, segment_id, segment.osm_way_id, distance
                        ));

                        // Check if point has any candidates
                        if point_idx >= window_start && point_idx <= window_end {
                            let rel_idx = point_idx - window_start;
                            if rel_idx < window_candidates.len() {
                                let candidates = &window_candidates[rel_idx];
                                job.add_explanation(format!(
                                    "Point {} has {} segment candidates",
                                    point_idx,
                                    candidates.len()
                                ));

                                // Log top 3 candidates
                                for (i, cand) in candidates.iter().take(3).enumerate() {
                                    job.add_explanation(format!(
                                        "  Top candidate #{}: Segment {} (OSM way {}), distance: {:.2}m",
                                        i+1, cand.segment.id, cand.segment.osm_way_id, cand.distance
                                    ));
                                }
                            }
                        }
                    }
                }
            }

            // If we have constraints but couldn't find a valid route,
            // signal that we should try without constraints
            return Ok(Vec::new());
        }

        Ok(best_route)
    }

    /// Calculate a comprehensive score for a candidate path
    fn calculate_comprehensive_score(
        &self,
        path: &[MatchedWaySegment],
        window_points: &[Point<f64>],
        window_timestamps: &[DateTime<Utc>],
        path_cost: f64,
        constrained_candidates: &HashMap<usize, Vec<&SegmentCandidate>>,
        config: Option<ScoringConfig>,
    ) -> f64 {
        // Define default weights
        let config = config.unwrap_or(ScoringConfig {
            distance_weight: 1.0,
            std_dev_weight: 0.5,
            length_weight: 0.2,
            road_class_weight: 0.8,
            continuity_weight: 0.5,
            constraint_weight: 1.0,
            check_excessive_distance: true,
        });

        // Validate path
        if path.is_empty() {
            return f64::MAX;
        }

        // Check excessive distance if enabled
        if config.check_excessive_distance
            && !self
                .check_for_excessive_point_match_distance(path, window_points)
                .is_empty()
        {
            return f64::MAX;
        }

        // 1. Distance score (average distance from points to path + standard deviation)
        let mut total_distance = 0.0;
        let mut min_distances = Vec::with_capacity(window_points.len());

        for &point in window_points {
            let mut min_distance = f64::MAX;

            for matched_segment in path {
                // Get actual coordinates considering interim points
                let coords = matched_segment.coordinates();
                if coords.is_empty() {
                    continue;
                }

                // Project point to segment
                let line = LineString::from(coords);
                let closest = line.closest_point(&point);
                let projection = match closest {
                    Closest::SinglePoint(p) => p,
                    _ => point, // Fallback
                };

                let distance = Haversine.distance(point, projection);
                min_distance = min_distance.min(distance);
            }

            total_distance += min_distance;
            min_distances.push(min_distance);
        }

        // Calculate average and standard deviation
        let avg_distance = total_distance / window_points.len() as f64;
        let variance = min_distances
            .iter()
            .map(|&d| (d - avg_distance).powi(2))
            .sum::<f64>()
            / min_distances.len() as f64;
        let std_dev = variance.sqrt();

        let distance_score = avg_distance + (std_dev * config.std_dev_weight);

        // 2. Length score (penalize unnecessarily long routes)
        let mut length_score = 0.0;
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

        // 3. Road class score (prefer major roads)
        let road_class_score = self.calculate_road_class_score(path);

        // 4. Continuity score (penalize fragmented routes)
        let mut transitions = 0;
        for i in 1..path.len() {
            if path[i].segment.osm_way_id != path[i - 1].segment.osm_way_id {
                transitions += 1;
            }
        }
        let continuity_score = transitions as f64;

        // 5. Constraint score (bonus for matching constrained points)
        let mut constraint_score = 0.0;
        for constrained_candidates in constrained_candidates.values() {
            let constrained_segment_ids: HashSet<u64> = constrained_candidates
                .iter()
                .map(|c| c.segment.id)
                .collect();

            // Check if path includes any constrained segments
            let matched = path.iter().any(|matched_segment| {
                constrained_segment_ids.contains(&matched_segment.segment.id)
            });

            if matched {
                // Bonus for matching a constrained point
                constraint_score -= 50.0;
            } else {
                // Penalty for failing to match a constrained point
                constraint_score += 100.0;
            }
        }

        // Combine all scores with appropriate weights
        (distance_score * config.distance_weight)
            + (length_score * config.length_weight)
            + (road_class_score * config.road_class_weight)
            + (continuity_score * config.continuity_weight)
            + (constraint_score * config.constraint_weight)
    }

    // Calculate road class score
    fn calculate_road_class_score(&self, path: &[MatchedWaySegment]) -> f64 {
        if path.is_empty() {
            return 0.0;
        }

        let mut score = 0.0;
        let mut total_length = 0.0;

        // Calculate weighted score based on highway types and segment lengths
        for matched_segment in path {
            let segment = &matched_segment.segment;
            let segment_length = matched_segment.length().max(1.0); // Avoid division by zero
            total_length += segment_length;

            // Assign score based on road type
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

            // Weight by segment length
            score += segment_score * segment_length;
        }

        // Normalize by total path length
        if total_length > 0.0 {
            score /= total_length;
        }

        score
    }

    // Check if route does not deviate from points too much
    fn check_for_excessive_point_match_distance(
        &self,
        route: &[MatchedWaySegment],
        window_points: &[Point<f64>],
    ) -> Vec<(usize, f64)> {
        let mut distance_errors = Vec::new();

        for (i, point) in window_points.iter().enumerate() {
            let mut min_distance = f64::INFINITY;

            for matched_segment in route {
                let coords = matched_segment.coordinates();
                if coords.is_empty() {
                    continue;
                }

                // Project point to segment
                let line = LineString::from(coords);
                let closest = line.closest_point(point);
                let projection = match closest {
                    Closest::SinglePoint(p) => p,
                    Closest::Intersection(p) => p,
                    Closest::Indeterminate => matched_segment.centroid(),
                };

                let distance = Haversine.distance(*point, projection);
                min_distance = min_distance.min(distance);
            }

            if min_distance > MAX_DISTANCE_BETWEEN_POINT_AND_MATCHED_ROUTE_METER {
                distance_errors.push((i, min_distance));
            }
        }

        distance_errors
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

    // Your existing new_matched_segments_for_route function (preserved as-is)
    // This function already helps avoid duplicating segments between routes
    fn new_matched_segments_for_route(
        &self,
        route: &[MatchedWaySegment],
        matched_segments: Vec<MatchedWaySegment>,
    ) -> Vec<MatchedWaySegment> {
        // If route is empty, just return all segments
        if route.is_empty() {
            return matched_segments;
        }

        // Get last segment in route
        let last_segment_id = route.last().unwrap().segment.id;

        // Skip first segment of new segments if it's the same as the last segment of route
        let start_idx =
            if !matched_segments.is_empty() && matched_segments[0].segment.id == last_segment_id {
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

    /// Process a window and add its segments to the complete route, handling window boundaries correctly
    fn fuse_window_into_route(
        &self,
        complete_route: &mut Vec<MatchedWaySegment>,
        window_route: Vec<MatchedWaySegment>,
        window_start: usize,
        window_end: usize,
        is_last_window: bool,
        gps_points: &[Point<f64>],
    ) -> Result<()> {
        if window_route.is_empty() {
            return Ok(());
        }

        // For the first window, just use its segments directly
        if complete_route.is_empty() {
            // For non-last windows, still truncate at the second-to-last point
            if !is_last_window && window_end > window_start {
                let second_to_last_point = window_end - 1;
                debug!(
                    "First window: truncating at second-to-last point {}",
                    second_to_last_point
                );

                // Find segments that correspond to points only up to the second-to-last point
                let relevant_points = &gps_points[window_start..=second_to_last_point];

                // Choose segments that best match our points
                let mut processed_segments = Vec::new();
                for segment in &window_route {
                    // Check if this segment closely matches any relevant point
                    let mut should_include = false;

                    for point in relevant_points {
                        let coords = segment.coordinates();
                        if coords.is_empty() {
                            continue;
                        }

                        let line = LineString::from(coords.clone());
                        let projection = line.closest_point(point);

                        let distance = match projection {
                            Closest::SinglePoint(p) => Haversine.distance(*point, p),
                            _ => continue,
                        };

                        if distance <= self.config.max_matching_distance {
                            should_include = true;
                            break;
                        }
                    }

                    if should_include {
                        processed_segments.push(segment.clone());
                    }
                }

                debug!(
                    "First window: keeping {} of {} segments",
                    processed_segments.len(),
                    window_route.len()
                );
                complete_route.extend(processed_segments);
            } else {
                // Last window, use all segments
                complete_route.extend(window_route);
            }
            return Ok(());
        }

        // For subsequent windows:
        // 1. Find where this window's first point matches in the existing route
        let overlap_point = &gps_points[window_start];

        // Find which segment in complete_route matches the overlap point (first point of this window)
        let mut best_segment_idx = None;
        let mut min_distance = f64::MAX;

        for (i, segment) in complete_route.iter().enumerate().rev() {
            // Check from the end, as the overlap point is likely near the end of the existing route
            let coords = segment.coordinates();
            if coords.is_empty() {
                continue;
            }

            let line = LineString::from(coords.clone());
            let closest = line.closest_point(overlap_point);

            let distance = match closest {
                Closest::SinglePoint(p) => Haversine.distance(*overlap_point, p),
                _ => continue,
            };

            if distance < min_distance && distance <= self.config.max_matching_distance {
                min_distance = distance;
                best_segment_idx = Some(i);
            }
        }

        // 2. Truncate complete_route up to that segment (inclusive)
        let truncate_idx = match best_segment_idx {
            Some(idx) => idx + 1, // Keep segment that matches overlap point
            None => {
                // No clear overlap found - check if we can find the last segment of complete_route
                // in the new window_route as a fallback
                let last_segment_id = complete_route.last().unwrap().segment.id;
                let overlap_idx = window_route
                    .iter()
                    .position(|seg| seg.segment.id == last_segment_id);

                if let Some(pos) = overlap_idx {
                    // We found the same segment in window_route, start adding segments after it
                    let new_segments: Vec<_> = window_route.into_iter().skip(pos + 1).collect();
                    complete_route.extend(new_segments);
                    return Ok(());
                }

                // Last resort - just use new_matched_segments_for_route to avoid duplication
                let new_segments =
                    self.new_matched_segments_for_route(complete_route, window_route);
                complete_route.extend(new_segments);
                return Ok(());
            }
        };

        debug!(
            "Found overlap at segment {} of complete route",
            truncate_idx - 1
        );

        // Don't truncate beyond the found segment
        if truncate_idx <= complete_route.len() {
            complete_route.truncate(truncate_idx);
        }

        // 3. Find which segment in window_route matches the overlap point
        let mut window_overlap_idx = None;
        min_distance = f64::MAX;

        for (i, segment) in window_route.iter().enumerate() {
            let coords = segment.coordinates();
            if coords.is_empty() {
                continue;
            }

            let line = LineString::from(coords.clone());
            let closest = line.closest_point(overlap_point);

            let distance = match closest {
                Closest::SinglePoint(p) => Haversine.distance(*overlap_point, p),
                _ => continue,
            };

            if distance < min_distance && distance <= self.config.max_matching_distance {
                min_distance = distance;
                window_overlap_idx = Some(i);
            }
        }

        // 4. Add segments from window_route after the overlap segment
        match window_overlap_idx {
            Some(idx) => {
                // For non-last windows, we also need to stop at the second-to-last point
                if !is_last_window && window_end > window_start {
                    let second_to_last_point = &gps_points[window_end - 1];

                    let mut segments_to_add = Vec::new();
                    let mut include_up_to = window_route.len();

                    // Start after the overlap segment
                    for (i, segment) in window_route.iter().enumerate().skip(idx + 1) {
                        // Check if this segment is past the second-to-last point
                        let coords = segment.coordinates();
                        if coords.is_empty() {
                            continue;
                        }

                        let line = LineString::from(coords.clone());
                        let closest = line.closest_point(second_to_last_point);

                        let distance = match closest {
                            Closest::SinglePoint(p) => Haversine.distance(*second_to_last_point, p),
                            _ => continue,
                        };

                        // If we find a segment that matches the second-to-last point,
                        // include it and stop adding more segments
                        if distance <= self.config.max_matching_distance {
                            include_up_to = i + 1; // Include this segment
                            break;
                        }
                    }

                    // Add segments from after overlap up to (and including) the segment for second-to-last point
                    segments_to_add.extend(
                        window_route
                            .into_iter()
                            .skip(idx + 1)
                            .take(include_up_to.saturating_sub(idx + 1)),
                    );

                    debug!(
                        "Adding {} segments from window (truncated at second-to-last point)",
                        segments_to_add.len()
                    );
                    complete_route.extend(segments_to_add);
                } else {
                    // For the last window, add all remaining segments
                    let segments_to_add: Vec<_> = window_route.into_iter().skip(idx + 1).collect();
                    debug!(
                        "Adding {} segments from window (last window)",
                        segments_to_add.len()
                    );
                    complete_route.extend(segments_to_add);
                }
            }
            None => {
                // Couldn't find overlap segment in window_route
                // Fall back to using new_matched_segments_for_route
                let new_segments =
                    self.new_matched_segments_for_route(complete_route, window_route);
                debug!(
                    "Adding {} segments using fallback logic",
                    new_segments.len()
                );
                complete_route.extend(new_segments);
            }
        }

        Ok(())
    }

    /// A* path finding with distance limit and loop avoidance
    /// Returns a vector of MatchedWaySegment with proper entry/exit nodes
    fn find_path_with_distance_limit(
        &self,
        job: &RouteMatchJob,
        from: u64,
        entry_idx: usize,
        to: u64,
        used_segments: &HashSet<u64>,
        max_distance: f64,
    ) -> Result<(f64, Vec<MatchedWaySegment>)> {
        // Trivial case: start and end are the same
        if from == to {
            if let Some(segment) = job.segment_map.borrow().get(&from) {
                let matched = MatchedWaySegment::new(
                    segment.clone(),
                    Some(entry_idx),
                    Some(entry_idx), // Set exit to same as entry for zero-length segment
                );
                return Ok((0.0, vec![matched]));
            }
        }

        // Get segment map for looking up segments
        let segment_map = job.segment_map.borrow();
        let relevant_points: Vec<Point<f64>> = job.gps_points.clone();

        // A* search implementation
        let mut open_set = BinaryHeap::new();
        let mut closed_set = HashSet::new();
        let mut g_scores = HashMap::new();
        let mut came_from = HashMap::new();
        let mut total_distances = HashMap::new();

        // Track entry/exit node indices for each segment in path
        // These are the coordinate indices, not node indices
        let mut segment_entry_idx = HashMap::new();
        let mut segment_exit_idx = HashMap::new();

        // Track the actual node IDs for segment connections
        let mut connection_node_ids = HashMap::new();

        // Initialize for starting segment
        segment_entry_idx.insert(from, entry_idx);

        // Get destination coordinates for heuristic
        let to_segment = segment_map
            .get(&to)
            .ok_or_else(|| anyhow!("Destination segment not found"))?;
        let goal_point = to_segment.centroid();
        let graph = job.graph.borrow();

        // Initialize search
        g_scores.insert(from, 0.0);
        total_distances.insert(from, 0.0);
        open_set.push(OrderedFloat(0.0), from);

        while let Some((_, current)) = open_set.pop() {
            // Check if we've reached destination
            if current == to {
                // Reconstruct path
                let mut matched_path = Vec::new();
                let cost = *g_scores.get(&to).unwrap_or(&0.0);
                let mut current_node = to;

                while current_node != from {
                    if let Some(segment) = segment_map.get(&current_node) {
                        // Get the entry/exit indices for this segment
                        let entry = segment_entry_idx.get(&current_node).copied();
                        let exit = segment_exit_idx.get(&current_node).copied();

                        // For debug purposes
                        if entry.is_none() || exit.is_none() {
                            warn!("Missing entry or exit index for segment {}", current_node);
                        }

                        let matched = MatchedWaySegment::new(segment.clone(), entry, exit);
                        matched_path.push(matched);
                    }

                    if let Some(&prev) = came_from.get(&current_node) {
                        // Get the exit index for the previous segment (where we came from)
                        // This ensures connected segments have matching exit/entry points
                        if let Some(&connection_node) =
                            connection_node_ids.get(&(prev, current_node))
                        {
                            // Find the node index in the previous segment
                            if let Some(prev_segment) = segment_map.get(&prev) {
                                if let Some(exit_pos) = prev_segment
                                    .nodes
                                    .iter()
                                    .position(|&n| n == connection_node)
                                {
                                    segment_exit_idx.insert(prev, exit_pos);
                                }
                            }
                        }

                        current_node = prev;
                    } else {
                        return Err(anyhow!("Path reconstruction failed"));
                    }
                }

                // Add the starting segment
                if let Some(segment) = segment_map.get(&from) {
                    let entry = Some(entry_idx);

                    // For the starting segment, find the exit node index correctly
                    let exit = if let Some(next_node) = matched_path.last().map(|m| m.segment.id) {
                        // If we have connection node information for this segment pair
                        if let Some(&connection_node) = connection_node_ids.get(&(from, next_node))
                        {
                            // Find the position of the connection node in this segment
                            segment
                                .nodes
                                .iter()
                                .position(|&n| n == connection_node)
                                .map(|pos| pos)
                        } else {
                            // Fallback - use existing exit index if set, or calculate based on destination
                            segment_exit_idx.get(&from).copied()
                        }
                    } else {
                        // No next segment, use entry as exit (zero-length traversal)
                        Some(entry_idx)
                    };

                    let matched = MatchedWaySegment::new(segment.clone(), entry, exit);
                    matched_path.push(matched);
                }

                // Reverse to get correct order
                matched_path.reverse();

                return Ok((cost, matched_path));
            }

            // Skip if we've already processed this node
            if closed_set.contains(&current) {
                continue;
            }
            closed_set.insert(current);

            // Get current segment
            let current_segment = match segment_map.get(&current) {
                Some(seg) => seg,
                None => continue, // Skip if segment not found
            };

            // Get the entry index for the current segment
            let current_entry_idx = *segment_entry_idx.get(&current).unwrap_or(&0);

            // Process neighbors
            for &neighbor in &current_segment.connections {
                // Skip if already processed or would create a loop
                if closed_set.contains(&neighbor)
                    || (neighbor != to && used_segments.contains(&neighbor))
                {
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

                // Find common nodes to check direction
                let common_nodes: Vec<u64> = current_segment
                    .nodes
                    .iter()
                    .filter(|&&n| neighbor_segment.nodes.contains(&n))
                    .cloned()
                    .collect();

                if !common_nodes.is_empty() {
                    if common_nodes.len() > 1 {
                        warn!("Currently only the first common node between edges is considered");
                    }

                    let common_node = common_nodes[0];

                    // Get exit position in current segment (node index)
                    let current_exit_idx = current_segment
                        .nodes
                        .iter()
                        .position(|&n| n == common_node)
                        .unwrap();

                    if current_segment.is_oneway && current_exit_idx < current_entry_idx {
                        // We would need to go backwards on the one-way to make this connection, which is invalid
                        continue;
                    }

                    // Get entry position in the neighbor (node index)
                    let neighbor_entry_idx = neighbor_segment
                        .nodes
                        .iter()
                        .position(|&n| n == common_node)
                        .unwrap();

                    if neighbor_segment.is_oneway
                        && neighbor_entry_idx == neighbor_segment.nodes.len() - 1
                    {
                        // We would be entering the one-way neighbor at the end and be stuck, invalid
                        continue;
                    }

                    // Calculate edge cost and distance traveled
                    let (edge_cost, distance) = calculate_dynamic_transition_cost(
                        current_segment,
                        neighbor_segment,
                        current_entry_idx,
                        Some(&relevant_points),
                    );

                    // Calculate distance and check limit
                    let new_total_distance = total_distances[&current] + distance;

                    if new_total_distance > max_distance {
                        continue;
                    }

                    let new_g_score = g_scores[&current] + edge_cost;

                    // Only consider if better path
                    if !g_scores.contains_key(&neighbor) || new_g_score < g_scores[&neighbor] {
                        // Store the common node ID for this segment transition
                        connection_node_ids.insert((current, neighbor), common_node);

                        // Record the exit index for the current segment (node index)
                        segment_exit_idx.insert(current, current_exit_idx);

                        // Update path
                        came_from.insert(neighbor, current);
                        g_scores.insert(neighbor, new_g_score);
                        total_distances.insert(neighbor, new_total_distance);

                        // Store the entry index for this neighbor segment (node index)
                        segment_entry_idx.insert(neighbor, neighbor_entry_idx);

                        // Calculate heuristic
                        let neighbor_point = neighbor_segment.centroid();
                        let h_score = Haversine.distance(neighbor_point, goal_point) / 1000.0; // km
                        let f_score = new_g_score + h_score;

                        // Add to open set
                        open_set.push(OrderedFloat(-f_score), neighbor); // Negative because we want min heap
                    }
                } else {
                    return Err(anyhow!(
                        "Graph is invalid, no common node between neighboring segments {} and {}",
                        current,
                        neighbor
                    ));
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

    /// Generate GeoJSON visualization for point candidates
    fn generate_point_candidates_geojson(
        &self,
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
        final_route: &[MatchedWaySegment],
        way_ids: &[u64],
        loaded_tiles: &HashSet<String>,
    ) -> Result<()> {
        // First check if any of the specified way IDs are in the final route (checking osm_way_id)
        let route_osm_way_ids: HashSet<u64> = final_route
            .iter()
            .map(|seg| seg.segment.osm_way_id)
            .collect();

        for &way_id in way_ids {
            if route_osm_way_ids.contains(&way_id) {
                info!(
                    "Debug: OSM Way ID {} is included in the final route",
                    way_id
                );

                // List all segments with this OSM way ID
                let segments: Vec<&MatchedWaySegment> = final_route
                    .iter()
                    .filter(|seg| seg.segment.osm_way_id == way_id)
                    .collect();

                for matched_segment in segments {
                    let start_idx = matched_segment.entry_node.unwrap_or(0);
                    let end_idx = matched_segment
                        .exit_node
                        .unwrap_or(matched_segment.segment.coordinates.len() - 1);

                    info!(
                        "Debug: OSM Way ID {} corresponds to segment ID {} (using nodes {}-{})",
                        way_id, matched_segment.segment.id, start_idx, end_idx
                    );
                }
                continue;
            }

            // Way ID not in final route, investigate why
            info!(
                "Debug: OSM Way ID {} is NOT included in the final route",
                way_id
            );

            // Rest of debug code as before...
            // Check if the way ID exists in loaded tiles
            let mut way_exists = false;
            let mut way_segment: Option<WaySegment> = None;

            for tile_id in loaded_tiles {
                let segments = self.tile_loader.get_all_segments_from_tile(tile_id)?;
                // Now search by osm_way_id instead of id
                if let Some(segment) = segments.iter().find(|s| s.osm_way_id == way_id) {
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

                for (i, matched_point) in final_route.iter().enumerate() {
                    // Use centroid of segment as reference point
                    let segment_point = segment.centroid();
                    let route_segment_point = matched_point.centroid();

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
                let is_connected = segment.connections.contains(&nearby_segment.segment.id)
                    || nearby_segment.segment.connections.contains(&segment.id);

                if is_connected {
                    info!(
                        "Debug: Segment ID {} (OSM Way ID {}) is connected to segment {} (OSM Way ID {}) in the route",
                        segment.id,
                        way_id,
                        nearby_segment.segment.id,
                        nearby_segment.segment.osm_way_id
                    );
                } else {
                    info!(
                        "Debug: Segment ID {} (OSM Way ID {}) is NOT connected to segment {} (OSM Way ID {}) in the route",
                        segment.id,
                        way_id,
                        nearby_segment.segment.id,
                        nearby_segment.segment.osm_way_id
                    );
                }

                // Analyze highway type and other attributes
                info!(
                    "Debug: OSM Way ID {} is type '{}' (route segment is type '{}')",
                    way_id, segment.highway_type, nearby_segment.segment.highway_type
                );
            }
        }

        Ok(())
    }

    /// Find the connection node between two segments
    /// Returns (node_id, reason) if found, None otherwise
    fn find_connection_between_segments(
        &self,
        segment1: &WaySegment,
        segment2: &WaySegment,
    ) -> Option<(u64, String)> {
        // First check for shared nodes
        let shared_nodes: Vec<u64> = segment1
            .nodes
            .iter()
            .filter(|n| segment2.nodes.contains(n))
            .cloned()
            .collect();

        if !shared_nodes.is_empty() {
            // Prefer endpoints if available
            if let (Some(first1), Some(last1), Some(first2), Some(last2)) = (
                segment1.nodes.first(),
                segment1.nodes.last(),
                segment2.nodes.first(),
                segment2.nodes.last(),
            ) {
                // Check each shared node to see if it's an endpoint
                for &node in &shared_nodes {
                    let is_endpoint1 = node == *first1 || node == *last1;
                    let is_endpoint2 = node == *first2 || node == *last2;

                    // Prefer nodes that are endpoints for both segments
                    if is_endpoint1 && is_endpoint2 {
                        return Some((node, "Shared endpoint node".to_string()));
                    }
                }

                // If no shared endpoints, try finding any endpoint
                for &node in &shared_nodes {
                    if node == *first1 || node == *last1 || node == *first2 || node == *last2 {
                        return Some((node, "Endpoint node for one segment".to_string()));
                    }
                }
            }

            // If no endpoints match, just use the first shared node
            return Some((shared_nodes[0], "Shared intermediate node".to_string()));
        }

        // If no shared nodes, check for explicit connections
        if segment1.connections.contains(&segment2.id) {
            if let Some(last_node) = segment1.nodes.last() {
                return Some((
                    *last_node,
                    "Connection from segment1 to segment2".to_string(),
                ));
            }
        }

        if segment2.connections.contains(&segment1.id) {
            if let Some(first_node) = segment2.nodes.first() {
                return Some((
                    *first_node,
                    "Connection from segment2 to segment1".to_string(),
                ));
            }
        }

        None
    }

    fn find_candidate_segments_for_point(
        &mut self,
        point: Point<f64>,
        loaded_tiles: &HashSet<String>,
        max_distance: f64,
        max_candidates: usize,
        allow_distance_extension: bool,
    ) -> Result<Vec<SegmentCandidate>> {
        let mut candidates = Vec::new();

        // First try with normal distance
        for tile_id in loaded_tiles {
            // Load tile and process segments
            let segments = self.tile_loader.get_all_segments_from_tile(tile_id)?;

            for segment in segments {
                // Use unified projection function with direction validity
                if let Some((projection, distance, closest_node_idx)) =
                    self.project_point_to_segment(point, &segment)
                {
                    // Check if within max distance
                    if distance <= max_distance {
                        // Calculate cost (lower is better)
                        let cost = distance / (max_distance / 2.0);

                        candidates.push(SegmentCandidate {
                            segment,
                            distance,
                            projection,
                            closest_node_idx,
                            cost,
                        });
                    }
                }
            }
        }

        // If no candidates found and extension allowed, try with increased distance
        if candidates.is_empty() && allow_distance_extension {
            debug!("No candidates found within normal range, increasing search distance");

            let extended_max = max_distance * 1.5; // 50% increase

            for tile_id in loaded_tiles {
                let segments = self.tile_loader.get_all_segments_from_tile(tile_id)?;

                for segment in segments {
                    if let Some((projection, distance, closest_node_idx)) =
                        self.project_point_to_segment(point, &segment)
                    {
                        if distance <= extended_max {
                            // Higher cost due to extended range
                            let cost = distance / (max_distance / 2.0) * 1.5;

                            candidates.push(SegmentCandidate {
                                segment,
                                distance,
                                projection,
                                closest_node_idx,
                                cost,
                            });
                        }
                    }
                }
            }
        }

        // Sort candidates by cost
        candidates.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(Ordering::Equal));

        // Limit number of candidates
        if candidates.len() > max_candidates {
            candidates.truncate(max_candidates);
        }

        Ok(candidates)
    }

    /// Project a point to a segment and calculate the distance
    /// Returns (projected_point, distance, closest_index)
    fn project_point_to_segment(
        &self,
        point: Point<f64>,
        segment: &WaySegment,
    ) -> Option<(Point<f64>, f64, usize)> {
        if segment.coordinates.is_empty() {
            return None;
        }

        // Use LineString for projection
        let line = LineString::from(segment.coordinates.clone());

        // Get closest point
        let projection = match line.closest_point(&point) {
            Closest::SinglePoint(p) => p,
            _ => segment.centroid(), // Fallback for rare cases
        };

        // Calculate distance
        let distance = Haversine.distance(point, projection);

        // Find closest coordinate index if needed
        let mut closest_idx = None;
        let mut min_distance = f64::MAX;

        for (i, coord) in segment.coordinates.iter().enumerate() {
            let coord_point = Point::new(coord.x, coord.y);
            let coord_distance = Haversine.distance(projection, coord_point);

            if coord_distance < min_distance {
                min_distance = coord_distance;
                closest_idx = Some(i);
            }
        }

        closest_idx.map(|closest_idx| (projection, distance, closest_idx))
    }

    fn analyze_window_match(
        &self,
        job: &RouteMatchJob,
        window_start: usize,
        window_end: usize,
        chosen_path: &[MatchedWaySegment],
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
            .map(|matched_segment| matched_segment.segment.osm_way_id)
            .collect();

        // Rest of the analysis is the same, just using the segment.osm_way_id instead of direct access
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
                    if let Some((rank, cost)) = job.get_way_id_candidate_rank(point_idx, way_id) {
                        rank_info.push(format!(
                            "point {}: rank {} (cost: {:.2})",
                            point_idx,
                            rank + 1,
                            cost
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

        // Continue with rest of the analysis as before...
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

        // Rest of the function remains unchanged
        // If we're here, the subsequence wasn't matched despite all ways being available
        job.add_explanation("SEQUENCE NOT MATCHED: The expected way sequence was not found as a continuous subsequence in the matched path".to_string());
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

    pub fn analyze_segment_connectivity(
        &self,
        job: &RouteMatchJob,
        window_start: usize,
        window_end: usize,
    ) -> Result<Value> {
        // Get the segment candidates for the start and end points
        let candidates = job.all_candidates.borrow();
        let start_candidates = candidates
            .get(window_start)
            .ok_or_else(|| anyhow!("No candidates for start point"))?;
        let end_candidates = candidates
            .get(window_end)
            .ok_or_else(|| anyhow!("No candidates for end point"))?;

        // Get the segment map
        let segment_map = job.segment_map.borrow();

        // Create debug geojson with connectivity analysis
        let mut features = Vec::new();

        // Add start point
        let start_point = job.gps_points[window_start];
        features.push(json!({
            "type": "Feature",
            "properties": {
                "type": "start_point",
                "index": window_start,
                "description": format!("Start Point #{}", window_start)
            },
            "geometry": {
                "type": "Point",
                "coordinates": [start_point.x(), start_point.y()]
            }
        }));

        // Add end point
        let end_point = job.gps_points[window_end];
        features.push(json!({
            "type": "Feature",
            "properties": {
                "type": "end_point",
                "index": window_end,
                "description": format!("End Point #{}", window_end)
            },
            "geometry": {
                "type": "Point",
                "coordinates": [end_point.x(), end_point.y()]
            }
        }));

        // Track all segments we'll analyze
        let mut analyzed_segments = HashSet::new();

        // Add top start candidates
        for (i, start_candidate) in start_candidates.iter().take(3).enumerate() {
            let segment = &start_candidate.segment;
            let coords: Vec<Vec<f64>> =
                segment.coordinates.iter().map(|c| vec![c.x, c.y]).collect();

            features.push(json!({
                "type": "Feature",
                "properties": {
                    "type": "start_candidate",
                    "segment_id": segment.id,
                    "osm_way_id": segment.osm_way_id,
                    "rank": i,
                    "distance": start_candidate.distance,
                    "highway_type": segment.highway_type,
                    "color": "#00FF00",
                    "opacity": 0.7,
                    "weight": 4,
                    "description": format!("Start Candidate #{}: ID {} (OSM: {}), Distance: {:.2}m",
                                        i, segment.id, segment.osm_way_id, start_candidate.distance)
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                }
            }));

            analyzed_segments.insert(segment.id);
        }

        // Add top end candidates
        for (i, end_candidate) in end_candidates.iter().take(3).enumerate() {
            let segment = &end_candidate.segment;
            let coords: Vec<Vec<f64>> =
                segment.coordinates.iter().map(|c| vec![c.x, c.y]).collect();

            features.push(json!({
                "type": "Feature",
                "properties": {
                    "type": "end_candidate",
                    "segment_id": segment.id,
                    "osm_way_id": segment.osm_way_id,
                    "rank": i,
                    "distance": end_candidate.distance,
                    "highway_type": segment.highway_type,
                    "color": "#FF0000",
                    "opacity": 0.7,
                    "weight": 4,
                    "description": format!("End Candidate #{}: ID {} (OSM: {}), Distance: {:.2}m",
                                       i, segment.id, segment.osm_way_id, end_candidate.distance)
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                }
            }));

            analyzed_segments.insert(segment.id);
        }

        // Now analyze connectivity between segments
        let mut connectivity_features = Vec::new();

        // Analyze if top start candidates can connect to end candidates
        for (i, start_candidate) in start_candidates.iter().take(3).enumerate() {
            let start_seg = &start_candidate.segment;

            for (j, end_candidate) in end_candidates.iter().take(3).enumerate() {
                let end_seg = &end_candidate.segment;

                // Skip self-connections
                if start_seg.id == end_seg.id {
                    continue;
                }

                // Check if directly connected
                let is_directly_connected = start_seg.connections.contains(&end_seg.id)
                    || end_seg.connections.contains(&start_seg.id);

                // Calculate distance between centroids
                let start_centroid = start_seg.centroid();
                let end_centroid = end_seg.centroid();
                let distance = Haversine.distance(start_centroid, end_centroid);

                // Create connection line
                connectivity_features.push(json!({
                    "type": "Feature",
                    "properties": {
                        "type": "segment_connection",
                        "from_segment": start_seg.id,
                        "to_segment": end_seg.id,
                        "from_osm_way": start_seg.osm_way_id,
                        "to_osm_way": end_seg.osm_way_id,
                        "connected": is_directly_connected,
                        "distance": distance,
                        "color": if is_directly_connected { "#00FF00" } else { "#FF0000" },
                        "opacity": 0.5,
                        "weight": 2,
                        "dashArray": if is_directly_connected { "1" } else { "5,5"} ,
                        "description": format!(
                            "Connection: Start #{} → End #{}: {} (Distance: {:.2}m)",
                            i, j,
                            if is_directly_connected { "CONNECTED" } else { "NOT CONNECTED" },
                            distance
                        )
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [start_centroid.x(), start_centroid.y()],
                            [end_centroid.x(), end_centroid.y()]
                        ]
                    }
                }));

                // If not directly connected, try to find a path
                if !is_directly_connected {
                    // Check if there's a graph
                    if let Some(graph) = job.graph.borrow().as_ref() {
                        // Check if both segments are in the graph
                        if graph.contains_node(start_seg.id) && graph.contains_node(end_seg.id) {
                            // Check for the shortest path (using just a few nodes to keep visualization clear)
                            let mut visited = HashSet::new();
                            let mut queue = VecDeque::new();
                            let mut came_from = HashMap::new();

                            visited.insert(start_seg.id);
                            queue.push_back(start_seg.id);

                            const MAX_DEPTH: usize = 5; // Limit search depth
                            let mut current_depth = 0;
                            let mut current_node_count = 1;
                            let mut next_node_count = 0;

                            let mut path_found = false;

                            while let Some(current) = queue.pop_front() {
                                if current == end_seg.id {
                                    path_found = true;
                                    break;
                                }

                                // Get neighbors
                                let neighbors: Vec<_> = graph.neighbors(current).collect();
                                for neighbor in neighbors {
                                    if !visited.contains(&neighbor) {
                                        visited.insert(neighbor);
                                        came_from.insert(neighbor, current);
                                        queue.push_back(neighbor);
                                        next_node_count += 1;
                                    }
                                }

                                // Track depth
                                current_node_count -= 1;
                                if current_node_count == 0 {
                                    current_depth += 1;
                                    if current_depth >= MAX_DEPTH {
                                        break;
                                    }
                                    current_node_count = next_node_count;
                                    next_node_count = 0;
                                }
                            }

                            // If path found, reconstruct and visualize it
                            if path_found {
                                let mut path = Vec::new();
                                let mut current = end_seg.id;

                                while current != start_seg.id {
                                    path.push(current);
                                    current = came_from[&current];
                                }
                                path.push(start_seg.id);
                                path.reverse();

                                // Add segments in path to analyzed_segments
                                for &seg_id in &path {
                                    analyzed_segments.insert(seg_id);
                                }

                                // Create path visualization
                                connectivity_features.push(json!({
                                    "type": "Feature",
                                    "properties": {
                                        "type": "potential_path",
                                        "from_segment": start_seg.id,
                                        "to_segment": end_seg.id,
                                        "hops": path.len() - 1,
                                        "color": "#00FFFF",
                                        "opacity": 0.7,
                                        "weight": 3,
                                        "description": format!(
                                            "Potential path from Start #{} to End #{}: {} hops",
                                            i, j, path.len() - 1
                                        )
                                    },
                                    "geometry": {
                                        "type": "LineString",
                                        "coordinates": path.iter()
                                            .filter_map(|&id| segment_map.get(&id))
                                            .map(|seg| seg.centroid())
                                            .map(|p| vec![p.x(), p.y()])
                                            .collect::<Vec<_>>()
                                    }
                                }));

                                // Add segments in path
                                for &seg_id in &path {
                                    if let Some(segment) = segment_map.get(&seg_id) {
                                        let coords: Vec<Vec<f64>> = segment
                                            .coordinates
                                            .iter()
                                            .map(|c| vec![c.x, c.y])
                                            .collect();

                                        features.push(json!({
                                            "type": "Feature",
                                            "properties": {
                                                "type": "path_segment",
                                                "segment_id": segment.id,
                                                "osm_way_id": segment.osm_way_id,
                                                "highway_type": segment.highway_type,
                                                "color": "#00FFFF",
                                                "opacity": 0.6,
                                                "weight": 3,
                                                "description": format!("Path Segment: ID {} (OSM: {})", 
                                                    segment.id, segment.osm_way_id)
                                            },
                                            "geometry": {
                                                "type": "LineString",
                                                "coordinates": coords
                                            }
                                        }));
                                    }
                                }
                            } else {
                                // No path found, explain why
                                let limit_reached = current_depth >= MAX_DEPTH;

                                connectivity_features.push(json!({
                                    "type": "Feature",
                                    "properties": {
                                        "type": "no_path",
                                        "from_segment": start_seg.id,
                                        "to_segment": end_seg.id,
                                        "reason": if limit_reached {
                                            "Search depth limit reached" }else {
                                            "No path exists"},
                                        "color": "#FF00FF",
                                        "opacity": 0.5,
                                        "weight": 1,
                                        "dashArray": "2,8",
                                        "description": format!(
                                            "No path found: Start #{} to End #{}: {}",
                                            i, j,
                                            if limit_reached {
                                                "Search depth limit reached (> 5 hops)"
                                            } else {
                                                "No path exists"
                                            }
                                        )
                                    },
                                    "geometry": {
                                        "type": "LineString",
                                        "coordinates": [
                                            [start_centroid.x(), start_centroid.y()],
                                            [end_centroid.x(), end_centroid.y()]
                                        ]
                                    }
                                }));

                                // Add visited nodes
                                for &visited_id in &visited {
                                    if let Some(segment) = segment_map.get(&visited_id) {
                                        analyzed_segments.insert(visited_id);
                                    }
                                }
                            }
                        } else {
                            // One or both segments not in graph
                            connectivity_features.push(json!({
                                "type": "Feature",
                                "properties": {
                                    "type": "not_in_graph",
                                    "from_segment": start_seg.id,
                                    "to_segment": end_seg.id,
                                    "from_in_graph": graph.contains_node(start_seg.id),
                                    "to_in_graph": graph.contains_node(end_seg.id),
                                    "color": "#FF00FF",
                                    "opacity": 0.5,
                                    "weight": 1,
                                    "dashArray": "5,10",
                                    "description": format!(
                                        "Connection impossible: {} not in graph",
                                        if !graph.contains_node(start_seg.id) && !graph.contains_node(end_seg.id) {
                                            "Both segments"
                                        } else if !graph.contains_node(start_seg.id) {
                                            "Start segment"
                                        } else {
                                            "End segment"
                                        }
                                    )
                                },
                                "geometry": {
                                    "type": "LineString",
                                    "coordinates": [
                                        [start_centroid.x(), start_centroid.y()],
                                        [end_centroid.x(), end_centroid.y()]
                                    ]
                                }
                            }));
                        }
                    }
                }
            }
        }

        // Add all analyzed segments that haven't been added yet
        for &seg_id in &analyzed_segments {
            if let Some(segment) = segment_map.get(&seg_id) {
                let is_start = start_candidates.iter().any(|c| c.segment.id == seg_id);
                let is_end = end_candidates.iter().any(|c| c.segment.id == seg_id);

                // Skip segments already added as start/end candidates
                if is_start || is_end {
                    continue;
                }

                let coords: Vec<Vec<f64>> =
                    segment.coordinates.iter().map(|c| vec![c.x, c.y]).collect();

                features.push(json!({
                    "type": "Feature",
                    "properties": {
                        "type": "analyzed_segment",
                        "segment_id": segment.id,
                        "osm_way_id": segment.osm_way_id,
                        "highway_type": segment.highway_type,
                        "color": "#888888",
                        "opacity": 0.5,
                        "weight": 2,
                        "description": format!("Analyzed Segment: ID {} (OSM: {})",
                            segment.id, segment.osm_way_id)
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords
                    }
                }));
            }
        }

        // Add connectivity features after segments for proper layering
        features.extend(connectivity_features);

        // Create GeoJSON
        let geojson = json!({
            "type": "FeatureCollection",
            "features": features
        });

        Ok(geojson)
    }

    pub fn debug_constrained_window_failure(
        &mut self,
        job: &RouteMatchJob,
        window_start: usize,
        window_end: usize,
        constraints: &[(usize, u64)],
    ) -> PathfindingDebugInfo {
        // Get window data
        let window_points = &job.gps_points[window_start..=window_end];
        let window_candidates = &job.all_candidates.borrow()[window_start..=window_end];

        // Create a map from point index in the original array to its relative position in the window
        let point_idx_map: HashMap<usize, usize> = (window_start..=window_end)
            .enumerate()
            .map(|(window_pos, original_idx)| (original_idx, window_pos))
            .collect();

        // Process constraints
        let mut constrained_candidates: HashMap<usize, Vec<u64>> = HashMap::new();

        for &(point_idx, segment_id) in constraints {
            if point_idx < window_start || point_idx > window_end {
                continue; // Skip constraints outside our window
            }

            let window_pos = point_idx_map[&point_idx];

            // Find candidates that match the segment ID
            let segment_map = job.segment_map.borrow();
            if let Some(segment) = segment_map.get(&segment_id) {
                let osm_way_id = segment.osm_way_id;

                // Keep track of all candidates for this point with the same OSM way ID
                let matching_candidates: Vec<u64> = window_candidates[window_pos]
                    .iter()
                    .filter(|cand| cand.segment.osm_way_id == osm_way_id)
                    .map(|cand| cand.segment.id)
                    .collect();

                if !matching_candidates.is_empty() {
                    constrained_candidates.insert(window_pos, matching_candidates);
                }
            }
        }

        // Get first and last candidates
        let first_candidates = window_candidates[0].clone();
        let last_candidates = window_candidates[window_candidates.len() - 1].clone();

        // Setup debug info
        let mut debug_info = PathfindingDebugInfo {
            start_point_idx: window_start,
            end_point_idx: window_end,
            start_candidates: first_candidates.clone(),
            end_candidates: last_candidates.clone(),
            constraints: constraints.to_vec(),
            attempted_pairs: Vec::new(),
            constrained_candidates: constrained_candidates.clone(),
            reason: "Unable to find valid path between constrained points".to_string(),
        };

        // Try path finding between each pair of start and end candidates
        for first_candidate in &first_candidates {
            for last_candidate in &last_candidates {
                // Skip if same segment for multi-point windows
                if first_candidate.segment.id == last_candidate.segment.id
                    && window_points.len() > 1
                {
                    continue;
                }

                // Calculate max allowed distance
                let max_distance = Self::max_route_length_from_points(
                    window_points,
                    &job.timestamps[window_start..=window_end],
                );

                // Try to find a path
                match self.find_path_with_distance_limit(
                    job,
                    first_candidate.segment.id,
                    window_start,
                    last_candidate.segment.id,
                    &HashSet::new(),
                    max_distance,
                ) {
                    Ok((path_cost, path)) => {
                        // Path found
                        debug_info.attempted_pairs.push(PathfindingAttempt {
                            from_segment: first_candidate.segment.id,
                            from_osm_way: first_candidate.segment.osm_way_id,
                            to_segment: last_candidate.segment.id,
                            to_osm_way: last_candidate.segment.osm_way_id,
                            distance: Haversine.distance(
                                first_candidate.segment.centroid(),
                                last_candidate.segment.centroid(),
                            ),
                            result: PathfindingResult::Success(path, path_cost),
                        });
                    }
                    Err(e) => {
                        // Path not found, analyze the reason
                        let error_msg = e.to_string();
                        let result = if error_msg.contains("distance limit") {
                            PathfindingResult::TooFar(
                                max_distance,
                                Haversine.distance(
                                    first_candidate.segment.centroid(),
                                    last_candidate.segment.centroid(),
                                ),
                            )
                        } else if error_msg.contains("not connected") {
                            PathfindingResult::NoConnection
                        } else {
                            PathfindingResult::NoPathFound(error_msg)
                        };

                        debug_info.attempted_pairs.push(PathfindingAttempt {
                            from_segment: first_candidate.segment.id,
                            from_osm_way: first_candidate.segment.osm_way_id,
                            to_segment: last_candidate.segment.id,
                            to_osm_way: last_candidate.segment.osm_way_id,
                            distance: Haversine.distance(
                                first_candidate.segment.centroid(),
                                last_candidate.segment.centroid(),
                            ),
                            result,
                        });
                    }
                }
            }
        }

        // Analyze results
        let success_count = debug_info
            .attempted_pairs
            .iter()
            .filter(|attempt| matches!(attempt.result, PathfindingResult::Success(_, _)))
            .count();

        if success_count == 0 {
            // Analyze failure reasons
            let too_far_count = debug_info
                .attempted_pairs
                .iter()
                .filter(|attempt| matches!(attempt.result, PathfindingResult::TooFar(_, _)))
                .count();

            let no_connection_count = debug_info
                .attempted_pairs
                .iter()
                .filter(|attempt| matches!(attempt.result, PathfindingResult::NoConnection))
                .count();

            if debug_info.attempted_pairs.is_empty() {
                debug_info.reason =
                    "No valid candidate pairs were found to attempt pathfinding".to_string();
            } else if too_far_count == debug_info.attempted_pairs.len() {
                debug_info.reason =
                    "All candidate pairs exceed the maximum distance limit".to_string();
            } else if no_connection_count == debug_info.attempted_pairs.len() {
                debug_info.reason =
                    "No connection exists between any start and end candidates".to_string();
            } else {
                debug_info.reason = format!(
                    "Mixed pathfinding failures: {}/{} too far, {}/{} no connection, others failed during A*",
                    too_far_count,
                    debug_info.attempted_pairs.len(),
                    no_connection_count,
                    debug_info.attempted_pairs.len()
                );
            }
        } else {
            debug_info.reason = format!(
                "Found {} successful paths, but constraints may prevent using them",
                success_count
            );
        }

        debug_info
    }

    pub fn debug_direct_segment_routing(
        &mut self,
        job: &RouteMatchJob,
        from_segment_id: u64,
        to_segment_id: u64,
    ) -> PathfindingDebugInfo {
        // Set up debug info
        let mut debug_info = PathfindingDebugInfo {
            start_point_idx: 0, // Using 0 as placeholder
            end_point_idx: 1,   // Using 1 as placeholder
            start_candidates: Vec::new(),
            end_candidates: Vec::new(),
            constraints: Vec::new(),
            attempted_pairs: Vec::new(),
            constrained_candidates: HashMap::new(),
            reason: "Direct segment route analysis".to_string(),
        };

        let segment_map = job.segment_map.borrow();

        // Try to get segments
        let start_segment = match segment_map.get(&from_segment_id) {
            Some(seg) => seg,
            None => {
                debug_info.reason = format!("Start segment {} not found", from_segment_id);
                return debug_info;
            }
        };

        let end_segment = match segment_map.get(&to_segment_id) {
            Some(seg) => seg,
            None => {
                debug_info.reason = format!("End segment {} not found", to_segment_id);
                return debug_info;
            }
        };

        // Create candidate entries for the UI
        let start_projection = start_segment.centroid();
        debug_info.start_candidates.push(SegmentCandidate {
            segment: start_segment.clone(),
            distance: 0.0, // Direct selection, not from a GPS point
            projection: start_projection,
            closest_node_idx: 0,
            cost: 0.0, // Perfect score for direct selection
        });

        let end_projection = end_segment.centroid();
        debug_info.end_candidates.push(SegmentCandidate {
            segment: end_segment.clone(),
            distance: 0.0, // Direct selection, not from a GPS point
            projection: end_projection,
            closest_node_idx: 0,
            cost: 0.0, // Perfect score for direct selection
        });

        // Calculate direct distance between segments
        let direct_distance = Haversine.distance(start_projection, end_projection);

        // Determine max allowed distance (use a multiple of the direct distance)
        let max_distance = (direct_distance * 3.0).max(2000.0); // At least 2km

        // Try to find a path using a clean search
        let attempt = match self.find_path_with_distance_limit(
            job,
            from_segment_id,
            0,
            to_segment_id,
            &HashSet::new(),
            max_distance,
        ) {
            Ok((path_cost, path)) => {
                // Success
                PathfindingAttempt {
                    from_segment: from_segment_id,
                    from_osm_way: start_segment.osm_way_id,
                    to_segment: to_segment_id,
                    to_osm_way: end_segment.osm_way_id,
                    distance: direct_distance,
                    result: PathfindingResult::Success(path, path_cost),
                }
            }
            Err(e) => {
                // Handle different failure cases
                let error_msg = e.to_string();
                let result = if error_msg.contains("distance limit") {
                    PathfindingResult::TooFar(max_distance, direct_distance)
                } else if error_msg.contains("not connected") || error_msg.contains("No connection")
                {
                    PathfindingResult::NoConnection
                } else {
                    PathfindingResult::NoPathFound(error_msg)
                };

                PathfindingAttempt {
                    from_segment: from_segment_id,
                    from_osm_way: start_segment.osm_way_id,
                    to_segment: to_segment_id,
                    to_osm_way: end_segment.osm_way_id,
                    distance: direct_distance,
                    result,
                }
            }
        };

        // Add the attempt to our debug info
        debug_info.attempted_pairs.push(attempt);

        // Detailed connectivity analysis
        // Check for direct connections
        let directly_connected = start_segment.connections.contains(&to_segment_id)
            || end_segment.connections.contains(&from_segment_id);

        if !directly_connected {
            // If not directly connected, try to figure out why
            let (_compatible, should_connect, reason) =
                check_segment_connectivity(start_segment, end_segment);
            if should_connect {
                debug_info.reason = format!(
                    "Segments should be connected based on: {}, but no connection exists in the graph",
                    reason
                );
            } else {
                debug_info.reason = format!("Segments are not connected: {}", reason);
            }

            // The connectivity issues
            let (_, _, issues) = check_segment_connectivity(start_segment, end_segment);
            if !issues.is_empty() {
                debug_info.reason = format!("{}. Issues: {}", debug_info.reason, &issues);
            }
        } else if debug_info.attempted_pairs[0].result.is_success() {
            debug_info.reason =
                "Route found successfully! Segments are properly connected.".to_string();
        } else {
            debug_info.reason =
                "Segments are directly connected but path finding failed.".to_string();
        }

        // Add information about connections from the start segment
        debug_info.reason += format!(
            "\nStart segment has {} outgoing connections, end segment has {} connections",
            start_segment.connections.len(),
            end_segment.connections.len()
        )
        .as_str();

        debug_info
    }

    /// Convert a route to GeoJSON with points
    pub fn route_to_geojson(&self, segments: &[MatchedWaySegment]) -> Result<serde_json::Value> {
        let mut route_coordinates = Vec::new();

        for segment in segments {
            // Get the actual coordinates directy from segment.coordinates()
            let coords = segment.coordinates();

            if coords.is_empty() {
                continue;
            }

            // If this is not the first segment and the first point of this segment
            // is the same as the last point of the previous segment, skip it
            if !route_coordinates.is_empty() {
                let last_point: &[f64; 2] = route_coordinates.last().unwrap();
                let first_point: &[f64; 2] = &[coords[0].x, coords[0].y];

                // Check if points are the same (with a small epsilon)
                let epsilon = 1e-8;
                let points_equal = (last_point[0] - first_point[0]).abs() < epsilon
                    && (last_point[1] - first_point[1]).abs() < epsilon;

                // Add all points except the first if it's a duplicate
                if points_equal {
                    // Add remaining points (skip first)
                    for i in 1..coords.len() {
                        route_coordinates.push([coords[i].x, coords[i].y]);
                    }
                } else {
                    // Add all points
                    for coord in &coords {
                        route_coordinates.push([coord.x, coord.y]);
                    }
                }
            } else {
                // For the first segment, add all points
                for coord in &coords {
                    route_coordinates.push([coord.x, coord.y]);
                }
            }
        }

        // Create GeoJSON
        let geojson = json!({
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {
                    "type": "route",
                    "description": "Route",
                    "color": "#3388ff",
                    "weight": 6
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": route_coordinates
                }
            }]
        });

        Ok(geojson)
    }

    // Helper function to check if two points are the same (with small epsilon)
    fn is_same_point(p1: &geo::Coord<f64>, p2: &geo::Coord<f64>) -> bool {
        const EPSILON: f64 = 1e-8;
        (p1.x - p2.x).abs() < EPSILON && (p1.y - p2.y).abs() < EPSILON
    }

    /// Create a diagnostic GeoJSON to visualize segment entry/exit points
    pub fn diagnose_segment_path(
        &self,
        segments: &[MatchedWaySegment],
    ) -> Result<serde_json::Value> {
        let mut features = Vec::new();

        // Add the main route as the first feature
        let mut route_coordinates = Vec::new();
        let mut last_added_point: Option<geo::Coord<f64>> = None;

        for segment in segments {
            let coords = segment.coordinates();
            if coords.is_empty() {
                continue;
            }

            let start_idx = if let Some(last_point) = last_added_point {
                if !coords.is_empty() && Self::is_same_point(&last_point, &coords[0]) {
                    1
                } else {
                    0
                }
            } else {
                0
            };

            for i in start_idx..coords.len() {
                route_coordinates.push(vec![coords[i].x, coords[i].y]);
                last_added_point = Some(coords[i]);
            }
        }

        features.push(json!({
            "type": "Feature",
            "properties": {
                "type": "route",
                "description": "Route",
                "color": "#3388ff",
                "weight": 6
            },
            "geometry": {
                "type": "LineString",
                "coordinates": route_coordinates
            }
        }));

        // Add diagnostic features showing the entry and exit points for each segment
        for (i, segment) in segments.iter().enumerate() {
            // Get full segment coordinates
            let full_coords = &segment.segment.coordinates;

            // Get actual coordinates based on entry/exit
            let actual_coords = segment.coordinates();

            if actual_coords.is_empty() || full_coords.is_empty() {
                continue;
            }

            // Add the full segment with transparency
            features.push(json!({
                "type": "Feature",
                "properties": {
                    "type": "full_segment",
                    "segment_id": segment.segment.id,
                    "segment_index": i,
                    "description": format!(
                        "Full Segment #{}: {}",
                        i, segment.segment.id
                    ),
                    "color": "#888888",
                    "weight": 2,
                    "opacity": 0.3
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": full_coords.iter().map(|c| vec![c.x, c.y]).collect::<Vec<_>>()
                }
            }));

            // Add entry point marker
            let entry_idx = segment.entry_node.unwrap_or(0);
            if entry_idx < full_coords.len() {
                let entry_point = &full_coords[entry_idx];
                features.push(json!({
                    "type": "Feature",
                    "properties": {
                        "type": "entry_point",
                        "segment_id": segment.segment.id,
                        "index": entry_idx,
                        "description": format!("Segment #{} Entry Point (index: {})", i, entry_idx)
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [entry_point.x, entry_point.y]
                    }
                }));
            }

            // Add exit point marker
            let exit_idx = segment.exit_node.unwrap_or(full_coords.len() - 1);
            if exit_idx < full_coords.len() {
                let exit_point = &full_coords[exit_idx];
                features.push(json!({
                    "type": "Feature",
                    "properties": {
                        "type": "exit_point",
                        "segment_id": segment.segment.id,
                        "index": exit_idx,
                        "description": format!("Segment #{} Exit Point (index: {})", i, exit_idx)
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [exit_point.x, exit_point.y]
                    }
                }));
            }

            // Add the actual route segment with different color
            features.push(json!({
                "type": "Feature",
                "properties": {
                    "type": "actual_segment",
                    "segment_id": segment.segment.id,
                    "segment_index": i,
                    "description": format!(
                        "Actual Segment #{}: {} (entry: {}, exit: {})",
                        i,
                        segment.segment.id,
                        entry_idx,
                        exit_idx
                    ),
                    "color": "#00aa00",
                    "weight": 4,
                    "opacity": 0.7
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": actual_coords.iter().map(|c| vec![c.x, c.y]).collect::<Vec<_>>()
                }
            }));
        }

        Ok(json!({
            "type": "FeatureCollection",
            "features": features
        }))
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

pub fn calculate_heading(from: Point<f64>, to: Point<f64>) -> f64 {
    let dx = to.x() - from.x();
    let dy = to.y() - from.y();
    dy.atan2(dx).to_degrees()
}

struct PathSegmentAnalysis {
    segment_idx: usize,
    way_id1: u64,
    way_id2: u64,
    connected: bool,
    reason: String,
    details: Vec<String>,
}
