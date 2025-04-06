use anyhow::{Result, anyhow, bail};
use chrono::{DateTime, Utc};
use core::f64;
use geo::{Closest, ClosestPoint, Haversine, Intersects, LineString, algorithm::Distance};
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
};

use crate::{
    osm_preprocessing::{OSMProcessor, TileIndex, WaySegment, are_road_types_compatible},
    tile_loader::TileLoader,
};

const MAX_DISTANCE_BETWEEN_POINT_AND_MATCHED_ROUTE_METER: f64 = 150.0;
const DISTANCE_THRESHOLD_FOR_COST_BIAS_METER: f64 = 10.0;

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
    pub(crate) score: f64,             // Overall score (lower is better)
}

#[derive(Debug, Clone)]
pub struct PathfindingDebugInfo {
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
    Success(Vec<WaySegment>, f64),
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

    pub(crate) fn candidate_to_matched(
        &self,
        candidate: &SegmentCandidate,
        projection_point: Point<f64>,
    ) -> MatchedWaySegment {
        let segment = &candidate.segment;
        let mut matched = MatchedWaySegment::from_full_segment(segment.clone());

        // Find closest coordinate index to projection point
        let mut min_distance = f64::MAX;
        let mut closest_idx = 0;

        for (i, coord) in segment.coordinates.iter().enumerate() {
            let point = Point::new(coord.x, coord.y);
            let distance = Haversine.distance(point, projection_point);

            if distance < min_distance {
                min_distance = distance;
                closest_idx = i;
            }
        }

        // For entry point, set as start_idx; for exit point, set as end_idx
        if projection_point == candidate.projection {
            matched.entry_node = Some(closest_idx);
        } else {
            matched.exit_node = Some(closest_idx);
        }

        // Ensure direction constraints for one-way roads
        if segment.is_oneway {
            // Check if both entry and exit are set
            if let (Some(entry), Some(exit)) = (matched.entry_node, matched.exit_node) {
                if entry > exit {
                    // Invalid direction for one-way road, adjust to forward traversal
                    matched.entry_node = Some(entry.min(exit));
                    matched.exit_node = Some(exit.max(entry));
                }
            }
        }

        matched
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
            // This should never happen if validation is properly applied elsewhere
            // But as a safeguard, handle it here too
            if entry_idx <= exit_idx {
                // Forward direction (correct for one-way)
                self.segment.coordinates[entry_idx..=exit_idx].to_vec()
            } else {
                // For one-way roads, we should not traverse backward
                // Return forward path from start to end of segment
                self.segment.coordinates.to_vec()
            }
        } else if entry_idx <= exit_idx {
            // Forward direction
            self.segment.coordinates[entry_idx..=exit_idx].to_vec()
        } else {
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
    pub fn validate_direction(&mut self) -> bool {
        if !self.segment.is_oneway {
            return true; // No need to validate bidirectional roads
        }

        // For one-way roads, entry_node must be <= exit_node to ensure forward travel
        if let (Some(entry), Some(exit)) = (self.entry_node, self.exit_node) {
            if entry > exit {
                return false; // Invalid direction for a one-way road
            }
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
    pub fn match_trace(&mut self, job: &mut RouteMatchJob) -> Result<Vec<MatchedWaySegment>> {
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

        // Step 3: Find candidate segments for each GPS point
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
    pub(crate) fn find_all_candidate_segments(
        &mut self,
        job: &RouteMatchJob,
        loaded_tiles: HashSet<String>,
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
                &loaded_tiles,
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
                        let projection = self.project_point_to_segment(point, segment);
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
            let mut window_route = self.find_a_star_path_for_window(
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

                if is_last_window {
                    self.trim_last_window_route(
                        job,
                        &complete_route,
                        &mut window_route,
                        window_end,
                    )?;
                }

                // If no overlap logic applied, just add non-overlapping segments
                let new_segments = self
                    .new_matched_segments_for_route(&complete_route, unconstrained_route.clone());

                if job.tracing {
                    let mut window_traces = job.window_trace.borrow_mut();
                    window_traces[window_index] = WindowTrace {
                        start: window_start,
                        end: window_end,
                        segments: new_segments.clone(),
                        bridge: false,
                        constraints: constraint_details,
                        used_constraints: true,
                        constraint_score: constrained_score,
                        unconstrained_score: None, // Didn't calculate unconstrained
                        attempted_way_ids: attempted_way_ids.into_iter().collect(),
                        debug_notes: vec![
                            format!(
                                "Window {}: Successfully matched with constraints",
                                window_index + 1
                            ),
                            "No overlap with previous windows found".to_string(),
                        ],
                    };
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
                            let mut window_traces = job.window_trace.borrow_mut();
                            window_traces[window_index] = WindowTrace {
                                start: window_start,
                                end: window_end,
                                segments: window_route.clone(),
                                bridge: false,
                                constraints: constraint_details,
                                used_constraints: true,
                                constraint_score: constrained_score,
                                unconstrained_score: None, // Didn't calculate unconstrained
                                attempted_way_ids: attempted_way_ids.into_iter().collect(),
                                debug_notes: vec![
                                    format!(
                                        "Window {}: Successfully matched with constraints",
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
                        complete_route.extend(window_route.clone());

                        // Update the point-to-segment mapping for the new segments
                        self.update_point_segment_mapping(
                            &mut previous_point_segments,
                            window_start,
                            window_end,
                            &window_route,
                            &job.gps_points,
                        );

                        continue;
                    }
                }

                if is_last_window {
                    self.trim_last_window_route(
                        job,
                        &complete_route,
                        &mut window_route,
                        window_end,
                    )?;
                }

                // If no replacement happened, just add non-overlapping segments
                let new_segments =
                    self.new_matched_segments_for_route(&complete_route, window_route.clone());

                if job.tracing {
                    let mut window_traces = job.window_trace.borrow_mut();
                    window_traces[window_index] = WindowTrace {
                        start: window_start,
                        end: window_end,
                        segments: new_segments.clone(),
                        bridge: false,
                        constraints: constraint_details,
                        used_constraints: true,
                        constraint_score: constrained_score,
                        unconstrained_score: None, // Didn't calculate unconstrained
                        attempted_way_ids: attempted_way_ids.into_iter().collect(),
                        debug_notes: vec![
                            format!(
                                "Window {}: Successfully matched with constraints",
                                window_index + 1
                            ),
                            "No overlap with previous windows found".to_string(),
                        ],
                    };
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
                                    segments.iter().map(|s| s.osm_way_id).collect();

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

    /// Special method to try to match just the last point when regular matching fails
    fn trim_last_window_route(
        &self,
        job: &RouteMatchJob,
        complete_route: &[MatchedWaySegment],
        window_route: &mut [MatchedWaySegment],
        last_point_idx: usize,
    ) -> Result<Vec<MatchedWaySegment>> {
        // Get all candidate segments for the last point
        let last_point = job.gps_points[last_point_idx];
        let last_point_candidates = &job.all_candidates.borrow()[last_point_idx];

        if last_point_candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Find the nearest node to the last point
        let segments: Vec<WaySegment> = last_point_candidates
            .iter()
            .map(|candidate| candidate.segment.clone())
            .collect();

        if let Some((node_id, node_point, distance)) =
            self.find_nearest_node_to_point(last_point, &segments)
        {
            debug!(
                "Last point special matching: found node {} at distance {:.2}m",
                node_id, distance
            );

            let previous_segment = complete_route.last();

            // First, check if we have a previous segment to connect from
            if let Some(prev) = previous_segment {
                // Strategy 1: Check if the previous segment is already connected to a segment containing our node
                let mut direct_connecting_segments = Vec::new();

                for candidate in last_point_candidates {
                    if candidate.segment.nodes.contains(&node_id)
                        && prev.segment.connections.contains(&candidate.segment.id)
                    {
                        // This is an ideal segment - it contains our target node AND is directly connected to previous segment

                        // Create a matched segment that ends at this node
                        let mut matched =
                            MatchedWaySegment::from_full_segment(candidate.segment.clone());

                        // Find the index of this node in the segment
                        if let Some(node_idx) =
                            matched.segment.nodes.iter().position(|&n| n == node_id)
                        {
                            // We want to ensure the segment goes FROM the connection point TO the node
                            if let Some((common_node, _)) = self
                                .find_connection_between_segments(&prev.segment, &matched.segment)
                            {
                                // Find the index of the connection node
                                if let Some(connection_idx) =
                                    matched.segment.nodes.iter().position(|&n| n == common_node)
                                {
                                    // Set entry to the connection point and exit to our target node
                                    matched.entry_node = Some(connection_idx);
                                    matched.exit_node = Some(node_idx);
                                    direct_connecting_segments.push(matched);
                                }
                            }
                        }
                    }
                }

                // If we found any direct connections, prioritize them
                if !direct_connecting_segments.is_empty() {
                    info!(
                        "Found {} segments that directly connect from previous segment to target node",
                        direct_connecting_segments.len()
                    );

                    // Sort by road type preference
                    direct_connecting_segments.sort_by(|a, b| {
                        let a_score = self.get_road_type_score(&a.segment.highway_type);
                        let b_score = self.get_road_type_score(&b.segment.highway_type);
                        a_score
                            .partial_cmp(&b_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                    return Ok(vec![direct_connecting_segments[0].clone()]);
                }
            }

            // Fallback: create segments for all candidates that contain the node
            let mut matching_segments = Vec::new();

            for candidate in last_point_candidates {
                if candidate.segment.nodes.contains(&node_id) {
                    // Create a matched segment that ends at this node
                    let mut matched =
                        MatchedWaySegment::from_full_segment(candidate.segment.clone());

                    // Find the index of this node in the segment
                    if let Some(node_idx) = matched.segment.nodes.iter().position(|&n| n == node_id)
                    {
                        // Set appropriate start/end indices to ensure we go TO the node
                        if node_idx == 0 {
                            // It's the first node
                            matched.entry_node = Some(0);
                            matched.exit_node = Some(0);
                        } else if node_idx == matched.segment.nodes.len() - 1 {
                            // It's the last node
                            matched.entry_node = Some(0);
                            matched.exit_node = Some(matched.segment.nodes.len() - 1);
                        } else {
                            // It's an intermediate node - take shortest path
                            if node_idx < matched.segment.nodes.len() / 2 {
                                matched.entry_node = Some(0);
                                matched.exit_node = Some(node_idx);
                            } else {
                                matched.entry_node = Some(matched.segment.nodes.len() - 1);
                                matched.exit_node = Some(node_idx);
                            }
                        }

                        matching_segments.push(matched);
                    }
                }
            }

            // If we found segments that contain the node
            if !matching_segments.is_empty() {
                // Sort by road type preference
                matching_segments.sort_by(|a, b| {
                    let a_score = self.get_road_type_score(&a.segment.highway_type);
                    let b_score = self.get_road_type_score(&b.segment.highway_type);
                    a_score
                        .partial_cmp(&b_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // If we have a previous segment, prefer connected segments
                if let Some(prev) = previous_segment {
                    // Try to find a segment that's connected to the previous segment
                    for matched in &matching_segments {
                        if prev.segment.connections.contains(&matched.segment.id) {
                            return Ok(vec![matched.clone()]);
                        }
                    }
                }

                // Just take the best segment
                return Ok(vec![matching_segments[0].clone()]);
            }
        }

        // Fallback: just use the best candidate with standard projection to node
        if !last_point_candidates.is_empty() {
            let best_candidate = &last_point_candidates[0];
            let mut matched = job.candidate_to_matched(best_candidate, best_candidate.projection);

            // Try to find the closest node to this projection
            let coords = matched.segment.coordinates.clone();
            let nodes = matched.segment.nodes.clone();

            let mut min_distance = f64::MAX;
            let mut closest_node_idx = None;

            for (i, &node_id) in nodes.iter().enumerate() {
                if i < coords.len() {
                    let coord = &coords[i];
                    let node_point = Point::new(coord.x, coord.y);
                    let distance = Haversine.distance(best_candidate.projection, node_point);

                    if distance < min_distance {
                        min_distance = distance;
                        closest_node_idx = Some(i);
                    }
                }
            }

            // Set the end index to this node
            if let Some(idx) = closest_node_idx {
                matched.exit_node = Some(idx);
            }

            return Ok(vec![matched]);
        }

        Ok(Vec::new())
    }

    /// Helper function to score road types (lower is better)
    fn get_road_type_score(&self, highway_type: &str) -> f64 {
        match highway_type {
            "motorway" | "motorway_link" => 1.0,
            "trunk" | "trunk_link" => 2.0,
            "primary" | "primary_link" => 3.0,
            "secondary" | "secondary_link" => 4.0,
            "tertiary" | "tertiary_link" => 5.0,
            "residential" => 6.0,
            "unclassified" => 7.0,
            "service" => 8.0,
            _ => 9.0,
        }
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

    /// Find the nearest node to a GPS point from a set of road segments
    fn find_nearest_node_to_point(
        &self,
        point: Point<f64>,
        segments: &[WaySegment],
    ) -> Option<(u64, Point<f64>, f64)> {
        let mut nearest_node_id = None;
        let mut nearest_node_point = None;
        let mut min_distance = f64::MAX;

        for segment in segments {
            // Check both first and last nodes (endpoints) of the segment
            if !segment.nodes.is_empty() {
                // Get coordinates for the first node
                let first_node_id = segment.nodes.first().unwrap();
                let first_coord = segment.coordinates.first().unwrap();
                let first_point = Point::new(first_coord.x, first_coord.y);
                let first_distance = Haversine.distance(point, first_point);

                if first_distance < min_distance {
                    min_distance = first_distance;
                    nearest_node_id = Some(*first_node_id);
                    nearest_node_point = Some(first_point);
                }

                // Get coordinates for the last node
                let last_node_id = segment.nodes.last().unwrap();
                let last_coord = segment.coordinates.last().unwrap();
                let last_point = Point::new(last_coord.x, last_coord.y);
                let last_distance = Haversine.distance(point, last_point);

                if last_distance < min_distance {
                    min_distance = last_distance;
                    nearest_node_id = Some(*last_node_id);
                    nearest_node_point = Some(last_point);
                }

                // Also consider intermediate nodes if we want to be thorough
                for (i, node_id) in segment
                    .nodes
                    .iter()
                    .enumerate()
                    .skip(1)
                    .take(segment.nodes.len() - 2)
                {
                    if i < segment.coordinates.len() {
                        let coord = &segment.coordinates[i];
                        let node_point = Point::new(coord.x, coord.y);
                        let distance = Haversine.distance(point, node_point);

                        if distance < min_distance {
                            min_distance = distance;
                            nearest_node_id = Some(*node_id);
                            nearest_node_point = Some(node_point);
                        }
                    }
                }
            }
        }

        if let (Some(id), Some(point)) = (nearest_node_id, nearest_node_point) {
            Some((id, point, min_distance))
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
                match self.find_path_with_distance_limit(
                    job,
                    first_candidate.segment.id,
                    last_candidate.segment.id,
                    &HashSet::new(),
                    Self::max_route_length_from_points(window_points, window_timestamps),
                    false,
                ) {
                    Ok((path_cost, path)) => {
                        // Add all way IDs from path to attempted ways
                        for segment in &path {
                            attempted_way_ids.insert(segment.osm_way_id);
                        }

                        // Convert to MatchedWaySegment with precise entry/exit points
                        let matched_path = self.convert_to_matched_segments(
                            path,
                            first_candidate.projection,
                            last_candidate.projection,
                        );

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
                        let (_, distance, _) = self.project_point_to_segment(point, segment);

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
        if config.check_excessive_distance && !self
                .check_for_excessive_point_match_distance(path, window_points)
                .is_empty() {
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

    /// Return new segments for route, avoiding duplicates
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

    /// A* path finding with distance limit and loop avoidance
    /// Enhanced to better handle split segments
    fn find_path_with_distance_limit(
        &self,
        job: &RouteMatchJob,
        from: u64,
        to: u64,
        used_segments: &HashSet<u64>,
        max_distance: f64,
        debug_mode: bool,
    ) -> Result<(f64, Vec<WaySegment>)> {
        // Trivial case: start and end are the same
        if from == to {
            if let Some(segment) = job.segment_map.borrow().get(&from) {
                return Ok((0.0, vec![segment.clone()]));
            }
        }

        // Get segment map for looking up segments
        let segment_map = job.segment_map.borrow();
        let relevant_points: Vec<Point<f64>> = job.gps_points.clone();

        // Check if segments exist and can be connected
        if let (Some(from_segment), Some(to_segment)) =
            (segment_map.get(&from), segment_map.get(&to))
        {
            // Find common nodes for direct connection
            let common_nodes: Vec<u64> = from_segment
                .nodes
                .iter()
                .filter(|&&n| to_segment.nodes.contains(&n))
                .cloned()
                .collect();

            // Determine entry node (use first common node or None)
            let entry_node = common_nodes.first().cloned();

            // Check direct connection first
            if from_segment.connections.contains(&to) || to_segment.connections.contains(&from) {
                // Directly connected, return immediate path with precise entry node
                let cost = calculate_transition_cost(
                    from_segment,
                    to_segment,
                    entry_node,
                    Some(&relevant_points),
                );
                return Ok((cost, vec![from_segment.clone(), to_segment.clone()]));
            }

            // Debug mode: perform additional checks before A* search
            if debug_mode {
                // Check for missing graph
                if job.graph.borrow().is_none() {
                    return Err(anyhow!("Graph not initialized"));
                }

                // Check if nodes are in graph
                let graph = job.graph.borrow();
                if !graph.as_ref().unwrap().contains_node(from)
                    || !graph.as_ref().unwrap().contains_node(to)
                {
                    return Err(anyhow!("One or both segments not in graph"));
                }

                // Calculate distance between centroids to check feasibility
                let distance = Haversine.distance(from_segment.centroid(), to_segment.centroid());
                if distance > max_distance * 1.5 {
                    return Err(anyhow!(
                        "Direct distance between segments ({:.2}m) exceeds max limit ({:.2}m)",
                        distance,
                        max_distance
                    ));
                }
            }
        } else {
            // At least one segment is missing
            let error_msg = if segment_map.get(&from).is_none() && segment_map.get(&to).is_none() {
                "Both segments not found in segment map"
            } else if segment_map.get(&from).is_none() {
                "Start segment not found in segment map"
            } else {
                "End segment not found in segment map"
            };
            return Err(anyhow!(error_msg));
        }

        // A* search implementation
        let mut open_set = BinaryHeap::new();
        let mut closed_set = HashSet::new();
        let mut g_scores = HashMap::new();
        let mut came_from = HashMap::new();
        let mut total_distances = HashMap::new();
        let mut entry_nodes = HashMap::new();

        // Get destination coordinates for heuristic
        let to_segment = segment_map
            .get(&to)
            .ok_or_else(|| anyhow!("Destination segment not found"))?;
        let goal_point = to_segment.centroid();

        // Initialize search
        g_scores.insert(from, 0.0);
        total_distances.insert(from, 0.0);
        entry_nodes.insert(from, None);
        open_set.push(OrderedFloat(0.0), from);

        while let Some((_, current)) = open_set.pop() {
            // Check if we've reached destination
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

            let graph = job.graph.borrow();

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

                // Calculate distance and check limit
                let segment_length = neighbor_segment.length();
                let new_total_distance = total_distances[&current] + segment_length;

                if new_total_distance > max_distance {
                    continue;
                }

                // Determine the entry node for the neighbor segment
                let entry_node = match entry_nodes.get(&current) {
                    Some(&Some(node)) => {
                        // Find the common node for this transition
                        let common_nodes: Vec<u64> = current_segment
                            .nodes
                            .iter()
                            .filter(|&&n| neighbor_segment.nodes.contains(&n))
                            .cloned()
                            .collect();

                        // Prefer the common node from the previous entry node if possible
                        common_nodes
                            .iter()
                            .find(|&&n| n == node)
                            .cloned()
                            .or_else(|| common_nodes.first().cloned())
                    }
                    _ => None, // No entry node specified, start from the first node
                };

                // Calculate edge cost
                let edge_cost = calculate_transition_cost(
                    current_segment,
                    neighbor_segment,
                    entry_node,
                    Some(&relevant_points),
                );

                let new_g_score = g_scores[&current] + edge_cost;

                // Only consider if better path
                if !g_scores.contains_key(&neighbor) || new_g_score < g_scores[&neighbor] {
                    // Update path
                    came_from.insert(neighbor, current);
                    g_scores.insert(neighbor, new_g_score);
                    total_distances.insert(neighbor, new_total_distance);

                    // Track the entry node for this neighbor
                    entry_nodes.insert(neighbor, entry_node);

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

    /// Build road network graph for path finding
    pub(crate) fn build_road_network(
        &mut self,
        loaded_tiles: &HashSet<String>,
    ) -> Result<(DiGraphMap<u64, f64>, HashMap<u64, WaySegment>)> {
        let mut graph = DiGraphMap::new();
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
            for &connected_segment_id in &segment.connections {
                if !graph.contains_edge(segment.id, connected_segment_id) {
                    // Calculate transition cost
                    let cost = if let Some(conn_segment) = segment_map.get(&connected_segment_id) {
                        // Find a common node for entry
                        let common_nodes: Vec<u64> = segment
                            .nodes
                            .iter()
                            .filter(|&&n| conn_segment.nodes.contains(&n))
                            .cloned()
                            .collect();

                        // Use the first common node or None if no common node
                        let entry_node = common_nodes.first().cloned();

                        calculate_transition_cost(segment, conn_segment, entry_node, None)
                    } else {
                        // Default cost
                        1.0
                    };

                    graph.add_edge(segment.id, connected_segment_id, cost);
                    if !segment.is_oneway {
                        graph.add_edge(connected_segment_id, segment.id, cost);
                    }
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
                            if !self.check_segment_connectivity(segment, other_segment).0 {
                                continue;
                            }

                            // Check the position of the shared node in each segment
                            let is_segment_endpoint = *segment.nodes.first().unwrap() == node_id
                                || *segment.nodes.last().unwrap() == node_id;

                            let is_other_endpoint = *other_segment.nodes.first().unwrap()
                                == node_id
                                || *other_segment.nodes.last().unwrap() == node_id;

                            // Prepare for entry node calculation
                            let entry_node = Some(node_id);

                            // Case 1: Both segments have this node as an endpoint (standard case)
                            if is_segment_endpoint && is_other_endpoint {
                                let cost = calculate_transition_cost(
                                    segment,
                                    other_segment,
                                    entry_node,
                                    None,
                                );
                                graph.add_edge(segment_id, other_segment_id, cost);
                                added_endpoint_connections += 1;
                                continue;
                            }

                            // Case 2: Endpoint connecting to an intermediate node (intersection)
                            if is_segment_endpoint || is_other_endpoint {
                                // For cases where a road endpoint meets another road at an intersection
                                // This is a legitimate connection - create a path from endpoint to midpoint
                                let cost = calculate_transition_cost(
                                    segment,
                                    other_segment,
                                    entry_node,
                                    None,
                                );
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
                                let cost = calculate_transition_cost(
                                    segment,
                                    other_segment,
                                    entry_node,
                                    None,
                                );
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
                                let cost = calculate_transition_cost(
                                    segment,
                                    other_segment,
                                    entry_node,
                                    None,
                                ) * 1.5;
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

    /// Check if two segments should be connected based on their properties and spatial relationship
    /// Returns (is_compatible, should_connect, reason)
    /// - is_compatible: True if segments have compatible properties (layers, road types)
    /// - should_connect: True if segments should be connected in the network
    /// - reason: Detailed explanation of the result
    fn check_segment_connectivity(
        &self,
        segment1: &WaySegment,
        segment2: &WaySegment,
    ) -> (bool, bool, String) {
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
            return (
                false,
                false,
                format!("Segments are on different layers: {} vs {}", layer1, layer2),
            );
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
            return (
                false,
                false,
                format!(
                    "Bridge mismatch: {} is {}a bridge, {} is {}a bridge",
                    segment1.id,
                    if is_bridge1 { "" } else { "not " },
                    segment2.id,
                    if is_bridge2 { "" } else { "not " }
                ),
            );
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
            return (
                false,
                false,
                format!(
                    "Tunnel mismatch: {} is {}a tunnel, {} is {}a tunnel",
                    segment1.id,
                    if is_tunnel1 { "" } else { "not " },
                    segment2.id,
                    if is_tunnel2 { "" } else { "not " }
                ),
            );
        }

        // 3. Check road type compatibility
        let compatible_types =
            are_road_types_compatible(&segment1.highway_type, &segment2.highway_type);

        if !compatible_types {
            return (
                false,
                false,
                format!(
                    "Incompatible road types: {} ({}) and {} ({})",
                    segment1.highway_type, segment1.id, segment2.highway_type, segment2.id
                ),
            );
        }

        // 4. Check traffic direction compatibility
        if segment1.is_oneway && segment2.is_oneway {
            // Additional checks could be performed here for direction compatibility
        }

        // 5. Now check if they should actually connect (spatial relationship)

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
        const CLOSE_DISTANCE_THRESHOLD: f64 = 1.0; // meters

        let seg1_start = segment1.coordinates.first().unwrap();
        let seg1_end = segment1.coordinates.last().unwrap();
        let seg2_start = segment2.coordinates.first().unwrap();
        let seg2_end = segment2.coordinates.last().unwrap();

        // Check all endpoint combinations for close proximity
        let start_start_point = Point::new(seg1_start.x, seg1_start.y);
        let start_start_dist =
            Haversine.distance(start_start_point, Point::new(seg2_start.x, seg2_start.y));
        if start_start_dist < CLOSE_DISTANCE_THRESHOLD {
            return (
                true,
                true,
                format!("Start points are very close ({:.2}m)", start_start_dist),
            );
        }

        let start_end_dist =
            Haversine.distance(start_start_point, Point::new(seg2_end.x, seg2_end.y));
        if start_end_dist < CLOSE_DISTANCE_THRESHOLD {
            return (
                true,
                true,
                format!(
                    "Start point of #1 is close to end point of #2 ({:.2}m)",
                    start_end_dist
                ),
            );
        }

        let end_start_point = Point::new(seg1_end.x, seg1_end.y);
        let end_start_dist =
            Haversine.distance(end_start_point, Point::new(seg2_start.x, seg2_start.y));
        if end_start_dist < CLOSE_DISTANCE_THRESHOLD {
            return (
                true,
                true,
                format!(
                    "End point of #1 is close to start point of #2 ({:.2}m)",
                    end_start_dist
                ),
            );
        }

        let end_end_dist = Haversine.distance(end_start_point, Point::new(seg2_end.x, seg2_end.y));
        if end_end_dist < CLOSE_DISTANCE_THRESHOLD {
            return (
                true,
                true,
                format!("End points are very close ({:.2}m)", end_end_dist),
            );
        }

        // Create LineStrings for intersection check
        let line1 = LineString::from(segment1.coordinates.clone());
        let line2 = LineString::from(segment2.coordinates.clone());

        // Check if lines intersect
        if line1.intersects(&line2) {
            return (true, true, "Lines geometrically intersect".to_string());
        }

        // At this point, segments are compatible but don't appear to connect
        (true, false, "No connection criteria met".to_string())
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
                    let (should_connect, _, _) =
                        self.check_segment_connectivity(segment1, segment2);
                    if should_connect && !connected {
                        connection_issues.push(format!(
                            "Segments {} and {} should be connected: {}",
                            segment1.id, segment2.id, should_connect
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

    /// Convert a path of WaySegments to MatchedWaySegments, respecting connections
    fn convert_to_matched_segments(
        &self,
        segments: Vec<WaySegment>,
        first_projection: Point<f64>,
        last_projection: Point<f64>,
    ) -> Vec<MatchedWaySegment> {
        if segments.is_empty() {
            return Vec::new();
        }

        let mut matched_segments = Vec::with_capacity(segments.len());

        // Process first segment with custom entry point
        if let Some(first_segment) = segments.first() {
            let mut matched_first = MatchedWaySegment::from_full_segment(first_segment.clone());

            // Find entry point (closest to projection)
            let (_, _, closest_idx) =
                self.project_point_to_segment(first_projection, first_segment);
            matched_first.entry_node = closest_idx;

            // If there's a second segment, find connection point for exit
            if segments.len() > 1 {
                if let Some(connection) =
                    self.find_connection_between_segments(first_segment, &segments[1])
                {
                    // Find index of connection node
                    if let Some(node_idx) =
                        first_segment.nodes.iter().position(|&n| n == connection.0)
                    {
                        matched_first.exit_node = Some(node_idx);
                    }
                }
            }

            // Validate direction (especially for one-way roads)
            if !matched_first.validate_direction() && first_segment.is_oneway {
                // For one-way roads, ensure forward traversal
                matched_first.entry_node = Some(0);
                matched_first.exit_node = Some(matched_first.segment.nodes.len() - 1);
            }

            matched_segments.push(matched_first);
        }

        // Process middle segments
        for i in 1..segments.len() - 1 {
            let prev = &segments[i - 1];
            let curr = &segments[i];
            let next = &segments[i + 1];

            let mut matched = MatchedWaySegment::from_full_segment(curr.clone());

            // Find entry node (connection from previous)
            if let Some(prev_conn) = self.find_connection_between_segments(prev, curr) {
                if let Some(entry_idx) = curr.nodes.iter().position(|&n| n == prev_conn.0) {
                    matched.entry_node = Some(entry_idx);
                }
            }

            // Find exit node (connection to next)
            if let Some(next_conn) = self.find_connection_between_segments(curr, next) {
                if let Some(exit_idx) = curr.nodes.iter().position(|&n| n == next_conn.0) {
                    matched.exit_node = Some(exit_idx);
                }
            }

            // Validate direction for one-way roads
            if !matched.validate_direction() && curr.is_oneway {
                matched.entry_node = Some(0);
                matched.exit_node = Some(curr.nodes.len() - 1);
            }

            matched_segments.push(matched);
        }

        // Process last segment with custom exit point
        if segments.len() > 1 {
            if let Some(last_segment) = segments.last() {
                let mut matched_last = MatchedWaySegment::from_full_segment(last_segment.clone());

                // Find exit point (closest to projection)
                let (_, _, closest_idx) =
                    self.project_point_to_segment(last_projection, last_segment);
                matched_last.exit_node = closest_idx;

                // Find entry node (connection from previous segment)
                let prev_segment = &segments[segments.len() - 2];
                if let Some(prev_conn) =
                    self.find_connection_between_segments(prev_segment, last_segment)
                {
                    if let Some(entry_idx) =
                        last_segment.nodes.iter().position(|&n| n == prev_conn.0)
                    {
                        matched_last.entry_node = Some(entry_idx);
                    }
                }

                // Validate direction for one-way roads
                if !matched_last.validate_direction() && last_segment.is_oneway {
                    matched_last.entry_node = Some(0);
                    matched_last.exit_node = Some(last_segment.nodes.len() - 1);
                }

                matched_segments.push(matched_last);
            }
        }

        matched_segments
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
        let (_, should_connect, _) = self.check_segment_connectivity(&segment1, &segment2);
        should_connect
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
            let segments = {
                let tile = self.tile_loader.load_tile(tile_id)?;
                tile.road_segments.clone()
            };

            for segment in segments {
                // Use unified projection function
                let (projection, distance, _) = self.project_point_to_segment(point, &segment);

                // Check if within max distance
                if distance <= max_distance {
                    // Calculate score (lower is better)
                    let score = distance / (max_distance / 2.0);

                    candidates.push(SegmentCandidate {
                        segment,
                        distance,
                        projection,
                        score,
                    });
                }
            }
        }

        // If no candidates found and extension allowed, try with increased distance
        if candidates.is_empty() && allow_distance_extension {
            debug!("No candidates found within normal range, increasing search distance");

            let extended_max = max_distance * 1.5; // 50% increase

            for tile_id in loaded_tiles {
                let segments = {
                    let tile = self.tile_loader.load_tile(tile_id)?;
                    tile.road_segments.clone()
                };

                for segment in segments {
                    let (projection, distance, _) = self.project_point_to_segment(point, &segment);

                    if distance <= extended_max {
                        // Higher score due to extended range
                        let score = distance / (max_distance / 2.0) * 1.5;

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
    ) -> (Point<f64>, f64, Option<usize>) {
        if segment.coordinates.is_empty() {
            return (point, f64::MAX, None);
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

        (projection, distance, closest_idx)
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
                    let (_, distance, _) = self.project_point_to_segment(point, segment);

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
                let (_, distance, _) = self.project_point_to_segment(point, segment);
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
                let (_, distance, _) = self.project_point_to_segment(point, segment);
                if distance < expected_min_distance {
                    expected_min_distance = distance;
                    closest_expected_segment = &segment.highway_type;
                }
            }

            // Find closest chosen segment
            let mut chosen_min_distance = f64::MAX;
            let mut closest_chosen_segment = "";
            for segment in chosen_path {
                let (_, distance, _) = self.project_point_to_segment(point, segment);
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
                        let (_, should_connect, reason) =
                            self.check_segment_connectivity(seg1, seg2);
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
                    let (_, distance, _) = self.project_point_to_segment(point, segment);
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
                    let (_, distance, _) = self.project_point_to_segment(point, segment);
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
                let (_, distance, _) = self.project_point_to_segment(point, segment);
                expected_min_distance = expected_min_distance.min(distance);
            }

            // Find closest chosen segment
            let mut chosen_min_distance = f64::MAX;
            for segment in chosen_path {
                let (_, distance, _) = self.project_point_to_segment(point, segment);
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

    /// Create a segment candidate with full details
    fn create_segment_candidate(
        &self,
        segment: WaySegment,
        point: Point<f64>,
        max_distance: f64,
    ) -> Option<SegmentCandidate> {
        // Project point to segment with all details
        let (projection, distance, _) = self.project_point_to_segment(point, &segment);

        // Check if within maximum allowed distance
        if distance <= max_distance {
            // Calculate score (lower is better)
            let score = distance / (max_distance / 2.0);

            // Create and return candidate
            return Some(SegmentCandidate {
                segment,
                distance,
                projection,
                score,
            });
        }

        None
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
                    last_candidate.segment.id,
                    &HashSet::new(),
                    max_distance,
                    true,
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
            score: 0.0, // Perfect score for direct selection
        });

        let end_projection = end_segment.centroid();
        debug_info.end_candidates.push(SegmentCandidate {
            segment: end_segment.clone(),
            distance: 0.0, // Direct selection, not from a GPS point
            projection: end_projection,
            score: 0.0, // Perfect score for direct selection
        });

        // Calculate direct distance between segments
        let direct_distance = Haversine.distance(start_projection, end_projection);

        // Determine max allowed distance (use a multiple of the direct distance)
        let max_distance = (direct_distance * 3.0).max(2000.0); // At least 2km

        // Try to find a path using a clean search
        let attempt = match self.find_path_with_distance_limit(
            job,
            from_segment_id,
            to_segment_id,
            &HashSet::new(),
            max_distance,
            true,
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
                self.check_segment_connectivity(start_segment, end_segment);
            if should_connect {
                debug_info.reason = format!(
                    "Segments should be connected based on: {}, but no connection exists in the graph",
                    reason
                );
            } else {
                debug_info.reason = format!("Segments are not connected: {}", reason);
            }

            // The connectivity issues
            let (_, _, issues) = self.check_segment_connectivity(start_segment, end_segment);
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

/// Calculate the traversal cost between two road segments
///
/// Parameters:
/// - from_seg: The segment we're coming from
/// - to_seg: The segment we're going to
/// - entry_node: Optional OSM node ID where the segments connect
/// Calculate the traversal cost between two road segments
fn calculate_transition_cost(
    from_seg: &WaySegment,
    to_seg: &WaySegment,
    entry_node: Option<u64>,
    gps_points: Option<&[Point<f64>]>,
) -> f64 {
    // Early validation
    if from_seg.coordinates.is_empty() || to_seg.coordinates.is_empty() {
        return f64::INFINITY;
    }

    // Get the connection node ID
    let connection_node_id = match entry_node {
        Some(node_id) => node_id,
        None => {
            // Find common node IDs between segments
            let shared_node_ids: Vec<u64> = from_seg
                .nodes
                .iter()
                .filter(|n| to_seg.nodes.contains(n))
                .cloned()
                .collect();

            if shared_node_ids.is_empty() {
                // No shared nodes, cannot establish connection
                return f64::INFINITY;
            } else {
                // Prefer endpoint connections if available
                let from_last_id = *from_seg.nodes.last().unwrap_or(&0);
                let to_first_id = *to_seg.nodes.first().unwrap_or(&0);

                if shared_node_ids.contains(&from_last_id) && shared_node_ids.contains(&to_first_id)
                {
                    from_last_id // Optimal end-to-start connection
                } else {
                    // Just use any shared node ID
                    shared_node_ids[0]
                }
            }
        }
    };

    // Find array indices of the connection node ID in both segments
    let entry_idx = match from_seg.nodes.iter().position(|&n| n == connection_node_id) {
        Some(idx) => idx,
        None => return f64::INFINITY, // Node ID not found in from_segment
    };

    let exit_idx = match to_seg.nodes.iter().position(|&n| n == connection_node_id) {
        Some(idx) => idx,
        None => return f64::INFINITY, // Node ID not found in to_segment
    };

    // Determine the entry point for the from_segment

    // In A* path finding, we're typically coming from some previous segment
    // For the start of a path, we might enter at any node
    // For intermediate segments, we typically enter at one node and exit at another

    // We need to handle different cases:
    // 1. Direct endpoint-to-endpoint connection
    // 2. Traversal through a segment from one node to another

    // Calculate distance along from_segment between entry and exit points
    let segment_distance = if entry_idx == exit_idx {
        // No traversal (enter and exit at same point)
        0.1 // Minimal cost
    } else {
        // We need to sum up distances between consecutive coordinates
        let mut distance = 0.0;

        // Make sure indices are properly ordered for iteration
        let (start_idx, end_idx) = if entry_idx <= exit_idx {
            (entry_idx, exit_idx)
        } else {
            (exit_idx, entry_idx)
        };

        // Walk along the segment coordinates and sum distances
        for i in start_idx..end_idx {
            if i + 1 < from_seg.coordinates.len() {
                let p1 = geo::Point::new(from_seg.coordinates[i].x, from_seg.coordinates[i].y);
                let p2 =
                    geo::Point::new(from_seg.coordinates[i + 1].x, from_seg.coordinates[i + 1].y);
                distance += Haversine.distance(p1, p2);
            }
        }

        // Normalize distance to a reasonable cost value
        (distance / 100.0).max(0.1)
    };

    // Road type preference factor
    let road_type_factor = get_road_type_factor(&to_seg.highway_type);

    // Continuity bonus for staying on the same road
    let continuity_factor = if from_seg.osm_way_id == to_seg.osm_way_id {
        0.7 // 30% discount
    } else {
        1.0
    };

    // Calculate base cost combines all factors
    let mut base_cost = (segment_distance * road_type_factor * continuity_factor) + 1.0;

    // Check GPS proximity if points are provided
    if let Some(points) = gps_points {
        // Check proximity to the destination segment (to_seg)
        for point in points {
            // Project the point to the segment to get the closest point
            let line = LineString::from(to_seg.coordinates.clone());
            let closest = line.closest_point(point);

            let projection = match closest {
                Closest::SinglePoint(p) => p,
                _ => continue, // Skip if no clear projection
            };

            let projection_distance = Haversine.distance(*point, projection);

            // If segment is close to a GPS point, apply strong negative cost
            if projection_distance <= DISTANCE_THRESHOLD_FOR_COST_BIAS_METER {
                // Apply a significant cost discount if the transition would bring us near a
                // strong segment candidate
                base_cost *= 0.1;
            }
        }
    }

    // Return base cost if no GPS proximity bonus applied
    base_cost
}

/// Determine the entry and exit indices for traversing a segment
/// Returns (entry_idx, exit_idx) for the from_segment
fn determine_traversal_indices(
    from_seg: &WaySegment,
    to_seg: &WaySegment,
    connection_idx: usize,
) -> (usize, usize) {
    // For one-way roads, direction is more constrained
    if from_seg.is_oneway {
        // For one-way, we typically enter at index 0 and exit at the connection
        return (0, connection_idx);
    }

    // For bidirectional roads, we need to determine the most likely entry point
    // based on the segment's topology

    // Case 1: Connection at start node (index 0)
    if connection_idx == 0 {
        // We're exiting at the start, so we must have entered at the end
        return (from_seg.nodes.len() - 1, connection_idx);
    }

    // Case 2: Connection at end node
    if connection_idx == from_seg.nodes.len() - 1 {
        // We're exiting at the end, so we must have entered at the start
        return (0, connection_idx);
    }

    // Case 3: Connection at intermediate node
    // This is more complex - we need to determine which direction makes more sense

    // If from_seg and to_seg share the same OSM way ID, prefer to continue in same direction
    if from_seg.osm_way_id == to_seg.osm_way_id {
        // Look at the to_segment's node index to determine direction
        let to_conn_idx = to_seg
            .nodes
            .iter()
            .position(|&n| n == from_seg.nodes[connection_idx])
            .unwrap_or(0);

        if to_conn_idx == 0 {
            // Entering to_seg at its start, so we likely came from the start of from_seg
            return (0, connection_idx);
        } else if to_conn_idx == to_seg.nodes.len() - 1 {
            // Entering to_seg at its end, so we likely came from the end of from_seg
            return (from_seg.nodes.len() - 1, connection_idx);
        }
    }

    // Default: assume we entered at the start
    // This is a reasonable fallback for most routing scenarios
    (0, connection_idx)
}

/// Get factor for road type preference
fn get_road_type_factor(highway_type: &str) -> f64 {
    match highway_type {
        "motorway" => 0.5,
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
