use anyhow::{Result, anyhow, bail};
use chrono::{DateTime, Utc};
use geo::{Closest, ClosestPoint, Haversine, LineString, algorithm::Distance};
use geo_types::Point;
use log::{debug, info, trace, warn};
use ordered_float::OrderedFloat;
use petgraph::prelude::{EdgeRef, UnGraphMap};
use serde_json::{Value, json};
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet},
    fs::File,
    io::Write as _,
    path::Path,
}; // Import OrderedFloat for floating-point comparisons

use crate::{
    osm_preprocessing::{OsmProcessor, RoadSegment},
    tile_loader::TileLoader,
};

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

/// Represents a candidate road segment for a GPS point
#[derive(Clone, Debug)]
pub struct Candidate {
    pub segment: RoadSegment,
    pub point_on_edge: Point<f64>,
    pub distance: f64,
    pub heading_diff: f64,
    pub speed: f64,
    pub weights: (f64, f64, f64), // (distance_weight, heading_weight, speed_weight)
    pub road_class_score: f64,    // Score for road class (lower is better for major roads)
}

impl Candidate {
    pub fn total_score(&self) -> f64 {
        let (dist_w, head_w, speed_w) = self.weights;

        // Lower score is better (used for min-heap)
        // Include road class in scoring and increase heading weight impact
        dist_w * self.distance
            + head_w * self.heading_diff * 1.5  // Increased importance of heading
            + speed_w * self.speed
            + 0.15 * self.road_class_score
    }
}

// PathNode for A* search with priority queue
#[derive(Clone, Debug, Eq)]
struct PathNode {
    segment_id: u64,
    cost: OrderedFloat<f64>,
    path: Vec<u64>,
    estimated_total: OrderedFloat<f64>,
}

impl PartialEq for PathNode {
    fn eq(&self, other: &Self) -> bool {
        self.estimated_total.eq(&other.estimated_total)
    }
}

impl Ord for PathNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (lower score is better)
        other.estimated_total.cmp(&self.estimated_total)
    }
}

impl PartialOrd for PathNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Main map matching entry point
pub struct MapMatcher {
    config: MapMatcherConfig,
    tile_loader: TileLoader,
}

impl MapMatcher {
    /// Create a new map matcher with given configuration
    pub fn new(config: MapMatcherConfig) -> Result<Self> {
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

        // Ensure tile cache directory exists
        std::fs::create_dir_all(&self.config.tile_cache_dir)?;

        let processor = OsmProcessor::new(self.config.tile_config.clone());
        processor.process_pbf(&self.config.osm_pbf_path, &self.config.tile_cache_dir)?;

        info!("OSM preprocessing finished");
        Ok(())
    }

    pub fn match_trace(
        &mut self,
        gps_points: &[Point<f64>],
        timestamps: &[DateTime<Utc>],
    ) -> Result<Vec<RoadSegment>> {
        if gps_points.is_empty() {
            return Ok(Vec::new());
        }

        info!("Starting map matching for {} points", gps_points.len());
        let start_time = std::time::Instant::now();

        // 1. Load tiles for entire trace area with buffer
        let trace_bbox = self.calculate_trace_bbox(gps_points);
        let buffer = self.config.max_matching_distance * 2.0 / 111_000.0; // Convert meters to approx degrees
        debug!(
            "Loading tiles for bbox: {:?} with buffer {:.1}m",
            trace_bbox,
            buffer * 111_000.0
        );

        let mut loaded_tiles = self.tile_loader.load_tile_range(
            trace_bbox,
            buffer,
            self.config.max_tiles_per_depth,
        )?;

        info!(
            "Initial tile loading complete. Loaded {} tiles",
            loaded_tiles.len()
        );

        // 2. Find candidates using tile-aware search
        let mut candidates = Vec::new();
        let mut current_path = Vec::new(); // Track current best path for loop prevention

        for (i, point) in gps_points.iter().enumerate() {
            if i % 100 == 0 {
                info!(
                    "Processing point {}/{} ({:.1}%)",
                    i + 1,
                    gps_points.len(),
                    (i + 1) as f64 / gps_points.len() as f64 * 100.0
                );
            }

            let point_start = std::time::Instant::now();
            let point_tiles = self.tile_loader.find_tiles_for_coordinate(*point)?;
            loaded_tiles.extend(point_tiles);

            let time = timestamps
                .get(i)
                .ok_or_else(|| anyhow!("Missing timestamp at index {}", i))?;

            let prev_point = if i > 0 { Some(gps_points[i - 1]) } else { None };
            let prev_timestamp = if i > 0 { Some(timestamps[i - 1]) } else { None };

            // Pass current path to find_candidates_in_tiles for loop prevention
            let point_candidates = self.find_candidates_in_tiles(
                *point,
                prev_point,
                *time,
                prev_timestamp,
                &loaded_tiles,
                &current_path,
            )?;

            debug!(
                "Point {}: Found {} candidates (processed in {:?})",
                i,
                point_candidates.len(),
                point_start.elapsed()
            );

            if point_candidates.is_empty() {
                warn!("No candidates found for point {} at {:?}", i, point);
            } else {
                trace!(
                    "Top candidate for point {}: distance {:.1}m, heading diff {:.1}°",
                    i, point_candidates[0].distance, point_candidates[0].heading_diff
                );
            }

            candidates.push(point_candidates);
        }

        // 3. Build global road network graph for all loaded segments
        info!("Building global road network graph");
        let graph_start = std::time::Instant::now();
        let mut graph = UnGraphMap::new();
        let mut loaded_segments = HashMap::new();

        // First collect all segments and store them
        for tile_id in &loaded_tiles {
            let tile = self.tile_loader.load_tile(tile_id)?;
            for segment in &tile.road_segments {
                loaded_segments.insert(segment.id, segment.clone());
            }
        }

        // Now build the graph (separation helps avoid borrowing conflicts)
        for segment in loaded_segments.values() {
            // Add connections to graph with proper weights
            for &conn_id in &segment.connections {
                if !graph.contains_edge(segment.id, conn_id) {
                    // Calculate transition cost
                    let cost = if let Some(conn_segment) = loaded_segments.get(&conn_id) {
                        // Use a separate function that takes owned values instead of self reference
                        calculate_transition_cost(segment, conn_segment)
                    } else {
                        // Default cost for not-yet-loaded segments
                        1.0
                    };
                    graph.add_edge(segment.id, conn_id, cost);
                }
            }
        }

        debug!(
            "Graph built with {} nodes and {} edges in {:?}",
            graph.node_count(),
            graph.edge_count(),
            graph_start.elapsed()
        );

        // 4. Use Viterbi algorithm to find most likely path
        info!(
            "Starting Viterbi algorithm for {} observations",
            candidates.len()
        );
        let viterbi_start = std::time::Instant::now();

        // Initialize with first layer probabilities (emission only)
        let mut trellis = Vec::new();
        let mut first_layer = Vec::new();

        for cand in &candidates[0] {
            // Initial probability is just emission probability (best score)
            let emission_prob = -cand.total_score(); // Negative because lower score is better
            first_layer.push((cand.segment.id, emission_prob, vec![cand.segment.id]));
        }
        trellis.push(first_layer);

        // Process subsequent layers
        for t in 1..candidates.len() {
            let prev_layer = &trellis[t - 1];
            let mut current_layer = Vec::new();

            for cand in &candidates[t] {
                let mut best_prob = f64::NEG_INFINITY;
                let mut best_path = Vec::new();

                // Find best previous state
                for (prev_id, prev_prob, prev_path) in prev_layer {
                    // Skip if adding this segment would create a loop
                    if self.would_form_loop(cand.segment.id, prev_path, &loaded_segments) {
                        continue;
                    }

                    // Calculate transition probability
                    match self.find_best_path(
                        *prev_id,
                        cand.segment.id,
                        &graph,
                        &mut loaded_segments,
                    ) {
                        Ok((path_cost, path)) => {
                            if path.is_empty() {
                                continue;
                            }

                            // Viterbi step: previous prob + transition + emission
                            let transition_prob = -path_cost; // Negative because lower cost is better
                            let emission_prob = -cand.total_score(); // Negative because lower score is better

                            // Apply a severe loop penalty if there's any loop in the path
                            let loop_penalty = if self.detect_loop_along_path(
                                &path,
                                prev_path,
                                &loaded_segments,
                            ) {
                                debug!("Applying severe loop penalty to path");
                                -self.config.loop_penalty_weight * 1000.0 // Much more severe penalty
                            } else {
                                0.0
                            };

                            // Road continuity bonus (prefer staying on the same road)
                            // let continuity_bonus = if !prev_path.is_empty() && !path.is_empty() {
                            //     let prev_seg_id = *prev_path.last().unwrap();
                            //     let curr_seg_id = path[0];

                            //     if let (Some(prev_seg), Some(curr_seg)) = (
                            //         loaded_segments.get(&prev_seg_id),
                            //         loaded_segments.get(&curr_seg_id),
                            //     ) {
                            //         if prev_seg.name.is_some() && prev_seg.name == curr_seg.name {
                            //             debug!(
                            //                 "Applied continuity bonus for staying on {}",
                            //                 prev_seg.name.as_ref().unwrap()
                            //             );
                            //             self.config.continuity_bonus_weight // Bonus for staying on same named road
                            //         } else if prev_seg.highway_type == curr_seg.highway_type {
                            //             self.config.continuity_bonus_weight * 0.5 // Smaller bonus for staying on same road type
                            //         } else {
                            //             0.0
                            //         }
                            //     } else {
                            //         0.0
                            //     }
                            // } else {
                            //     0.0
                            // };
                            let continuity_bonus = 0.0;

                            let total_prob = prev_prob
                                + transition_prob
                                + emission_prob
                                + loop_penalty
                                + continuity_bonus;

                            if total_prob > best_prob {
                                best_prob = total_prob;
                                // Build new path
                                best_path = prev_path.clone();
                                best_path.extend(path.iter().skip(1)); // Skip first node as it's already in prev_path
                            }
                        }
                        Err(e) => {
                            warn!(
                                "Error finding path from {} to {}: {}. Creating direct connection.",
                                prev_id, cand.segment.id, e
                            );

                            // Create direct connection as fallback
                            let transition_prob = -1000.0; // High cost for fallback
                            let emission_prob = -cand.total_score();
                            let total_prob = prev_prob + transition_prob + emission_prob;

                            if total_prob > best_prob {
                                best_prob = total_prob;
                                best_path = prev_path.clone();
                                best_path.push(cand.segment.id);
                            }
                        }
                    }
                }

                if !best_path.is_empty() {
                    current_layer.push((cand.segment.id, best_prob, best_path));
                }
            }

            if current_layer.is_empty() {
                warn!("No valid transitions at step {}", t);
                // Fall back to just using best candidate
                if let Some(cand) = candidates[t].first() {
                    current_layer.push((
                        cand.segment.id,
                        -cand.total_score(),
                        vec![cand.segment.id],
                    ));
                }
            }

            trellis.push(current_layer.clone());

            // Update current path with best choice so far to inform next iteration
            if let Some(best_state) = current_layer
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            {
                current_path = best_state.2.clone();
            }
        }

        // Get most likely sequence
        let final_layer = trellis.last().ok_or_else(|| anyhow!("Empty trellis"))?;
        let best_sequence = final_layer
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .ok_or_else(|| anyhow!("No path found"))?
            .2
            .clone();

        debug!(
            "Viterbi algorithm completed in {:?}, found path with {} segments",
            viterbi_start.elapsed(),
            best_sequence.len()
        );

        // Apply post-processing to fix any obvious errors or loops
        let cleaned_sequence = self.post_process_path(&best_sequence, &loaded_segments);
        debug!(
            "After post-processing: path has {} segments (removed {} segments)",
            cleaned_sequence.len(),
            best_sequence.len() - cleaned_sequence.len()
        );

        // 5. Reconstruct geometry
        info!(
            "Reconstructing geometry from {} path segments",
            cleaned_sequence.len()
        );
        let result = self.reconstruct_geometry(cleaned_sequence, &loaded_segments)?;

        info!(
            "Map matching completed in {:.2?} total",
            start_time.elapsed()
        );
        debug!("Final path contains {} road segments", result.len());
        debug!(
            "Peak memory usage: {} loaded tiles",
            self.tile_loader.loaded_tiles.len()
        );

        Ok(result)
    }

    fn find_candidates_in_tiles(
        &mut self,
        point: Point<f64>,
        prev_point: Option<Point<f64>>,
        timestamp: DateTime<Utc>,
        prev_timestamp: Option<DateTime<Utc>>,
        tiles: &HashSet<String>,
        current_path: &[u64],
    ) -> Result<Vec<Candidate>> {
        let mut candidates = Vec::new();
        let mut high_priority_candidates = Vec::new();

        // Calculate direction of travel and speed if previous point exists
        let travel_direction = prev_point.map(|pp| calculate_heading(pp, point));
        let current_speed = prev_point
            .zip(prev_timestamp)
            .and_then(|(pp, pt)| calculate_speed(pp, point, timestamp, pt));

        // Search across all loaded tiles
        for tile_id in tiles {
            let segments = {
                let tile = self.tile_loader.load_tile(tile_id)?;
                tile.road_segments.clone() // Clone segments to break borrow
            };

            for seg in &segments {
                // Skip if this segment would likely form a loop
                let loaded_segments = self.get_segment_map(&[seg.id])?;
                if !current_path.is_empty()
                    && self.would_form_loop(seg.id, current_path, &loaded_segments)
                {
                    continue;
                }

                // Check heading direction if we have a direction of travel
                if let (Some(dir), Some(speed)) = (travel_direction, current_speed) {
                    let seg_dir = seg.direction_at_point(point);
                    let heading_diff = angle_difference(dir, seg_dir);

                    // Skip segments that would make us go in the opposite direction
                    // unless we're at a slow speed (indicating a potential turn/u-turn)
                    if heading_diff > self.config.max_turn_angle
                        && speed > self.config.min_turn_speed
                    {
                        continue;
                    }
                }

                // Score the segment
                if let Some(cand) =
                    self.score_segment(seg, point, prev_point, timestamp, prev_timestamp)
                {
                    // Only add if within max distance
                    if cand.distance <= self.config.max_matching_distance {
                        // Prioritize major roads with low scores
                        if cand.road_class_score < 3.0 && cand.total_score() < 10.0 {
                            high_priority_candidates.push(cand);
                        } else {
                            candidates.push(cand);
                        }
                    }
                }
            }
        }

        // Sort both sets of candidates
        high_priority_candidates
            .sort_by(|a, b| a.total_score().partial_cmp(&b.total_score()).unwrap());
        candidates.sort_by(|a, b| a.total_score().partial_cmp(&b.total_score()).unwrap());

        // Combine them, prioritizing major roads
        let mut final_candidates = high_priority_candidates;
        final_candidates.extend(candidates);

        // Limit the total number of candidates
        Ok(final_candidates
            .into_iter()
            .take(self.config.max_candidates_per_point)
            .collect())
    }

    fn would_form_loop(
        &self,
        candidate_id: u64,
        current_path: &[u64],
        loaded_segments: &HashMap<u64, RoadSegment>,
    ) -> bool {
        // Empty path can't form a loop
        if current_path.is_empty() {
            return false;
        }

        // 1. Check for repeated segment ID (most basic loop detection)
        if current_path.contains(&candidate_id) {
            // Only consider it a loop if the segment appears in the recent history
            // (to accommodate legitimate returns to same segment after a long journey)
            let recent_history = std::cmp::min(500, current_path.len());
            let recent_path = &current_path[current_path.len() - recent_history..];

            if recent_path.contains(&candidate_id) {
                debug!(
                    "Prevented loop: segment {} would repeat in recent path",
                    candidate_id
                );
                return true;
            }
        }

        // 2. Check for geographic proximity to create spatial loop detection
        // Only do this check if we have enough segments and the candidate segment exists
        if current_path.len() >= 5 {
            if let Some(candidate_segment) = loaded_segments.get(&candidate_id) {
                let candidate_start = candidate_segment.coordinates.first().unwrap();
                let candidate_end = candidate_segment.coordinates.last().unwrap();

                // Get the beginning part of the path (not the very recent points)
                let path_beginning = if current_path.len() > 10 {
                    &current_path[0..current_path.len() - 5]
                } else {
                    &current_path[0..current_path.len() / 2]
                };

                // Check if candidate would create a geographic loop by connecting back to an earlier segment
                for &segment_id in path_beginning {
                    if let Some(earlier_segment) = loaded_segments.get(&segment_id) {
                        for coord in &earlier_segment.coordinates {
                            // Calculate distance to candidate segment endpoints
                            let distance_to_start = Haversine.distance(
                                Point::new(candidate_start.x, candidate_start.y),
                                Point::new(coord.x, coord.y),
                            );

                            let distance_to_end = Haversine.distance(
                                Point::new(candidate_end.x, candidate_end.y),
                                Point::new(coord.x, coord.y),
                            );

                            // If either endpoint is very close to an earlier segment, likely forming a loop
                            if distance_to_start < self.config.loop_distance_threshold
                                || distance_to_end < self.config.loop_distance_threshold
                            {
                                debug!(
                                    "Prevented spatial loop: segment {} is too close to earlier segment {}",
                                    candidate_id, segment_id
                                );
                                return true;
                            }
                        }
                    }
                }
            }
        }

        false
    }

    fn find_best_path(
        &mut self,
        from: u64,
        to: u64,
        graph: &UnGraphMap<u64, f64>,
        loaded_segments: &mut HashMap<u64, RoadSegment>,
    ) -> Result<(f64, Vec<u64>)> {
        if from == to {
            return Ok((0.0, vec![from]));
        }

        // Use A* search
        let mut open_set = BinaryHeap::new();
        let mut closed_set = HashSet::new();
        let mut costs = HashMap::new();

        // Get destination coordinates for heuristic
        let to_segment = match self.get_segment(to, loaded_segments) {
            Ok(segment) => segment,
            Err(e) => {
                warn!(
                    "Destination segment {} not found: {}. Using fallback.",
                    to, e
                );
                // Return direct path if we can't find the segment
                return Ok((100.0, vec![from, to]));
            }
        };

        let goal_point = to_segment.centroid();

        // Initialize with start node
        costs.insert(from, OrderedFloat(0.0));
        open_set.push(PathNode {
            segment_id: from,
            cost: OrderedFloat(0.0),
            path: vec![from],
            estimated_total: OrderedFloat(0.0),
        });

        // Process nodes in order of lowest estimated cost
        while let Some(current) = open_set.pop() {
            if current.segment_id == to {
                return Ok((*current.cost, current.path));
            }

            if closed_set.contains(&current.segment_id) {
                continue;
            }
            closed_set.insert(current.segment_id);

            // Get current node info
            let current_segment = match self.get_segment(current.segment_id, loaded_segments) {
                Ok(segment) => segment,
                Err(e) => {
                    warn!("Segment {} not found: {}. Skipping.", current.segment_id, e);
                    continue;
                }
            };

            // Process all neighbors
            for edge in graph.edges(current.segment_id) {
                let neighbor_id = edge.target();
                if closed_set.contains(&neighbor_id) {
                    continue;
                }

                // Skip if adding the neighbor would create a loop in the path
                if current.path.contains(&neighbor_id) {
                    continue;
                }

                // Get neighbor segment
                let neighbor_segment = match self.get_segment(neighbor_id, loaded_segments) {
                    Ok(segment) => segment,
                    Err(e) => {
                        warn!(
                            "Neighbor segment {} not found: {}. Skipping.",
                            neighbor_id, e
                        );
                        continue;
                    }
                };

                // Check if oneway restriction applies
                if current_segment.is_oneway && !current_segment.connections.contains(&neighbor_id)
                {
                    continue;
                }

                // Calculate actual cost to neighbor
                let edge_cost = *edge.weight();
                let new_cost = current.cost.0 + edge_cost;

                // Only consider if this is a better path
                if !costs.contains_key(&neighbor_id)
                    || OrderedFloat(new_cost) < *costs.get(&neighbor_id).unwrap()
                {
                    costs.insert(neighbor_id, OrderedFloat(new_cost));

                    // Calculate heuristic (Haversine distance to goal)
                    let neighbor_point = neighbor_segment.centroid();
                    let heuristic = Haversine.distance(neighbor_point, goal_point) / 1000.0; // Convert to km for better scale

                    // Create new path
                    let mut new_path = current.path.clone();
                    new_path.push(neighbor_id);

                    // Add to queue
                    open_set.push(PathNode {
                        segment_id: neighbor_id,
                        cost: OrderedFloat(new_cost),
                        path: new_path,
                        estimated_total: OrderedFloat(new_cost + heuristic),
                    });
                }
            }
        }

        // If no path found, return a direct connection as fallback
        warn!(
            "No path found between {} and {}. Using direct connection as fallback.",
            from, to
        );
        Ok((1000.0, vec![from, to]))
    }

    fn reconstruct_geometry(
        &mut self,
        path: Vec<u64>,
        loaded_segments: &HashMap<u64, RoadSegment>,
    ) -> Result<Vec<RoadSegment>> {
        let mut result = Vec::new();

        for &seg_id in &path {
            if let Some(segment) = loaded_segments.get(&seg_id) {
                result.push(segment.clone());
            } else {
                // Segment not in cache, try to load it
                let segment = self.tile_loader.get_segment(seg_id)?;
                result.push(segment);
            }
        }

        Ok(result)
    }

    fn calculate_trace_bbox(&self, points: &[Point<f64>]) -> geo::Rect<f64> {
        // Handle empty input
        if points.is_empty() {
            return geo::Rect::new(
                geo::Coord {
                    x: -180.0,
                    y: -90.0,
                },
                geo::Coord { x: 180.0, y: 90.0 },
            );
        }

        // Initialize with first point
        let mut min_x = points[0].x();
        let mut max_x = points[0].x();
        let mut min_y = points[0].y();
        let mut max_y = points[0].y();

        // Calculate bounds
        for point in points {
            min_x = min_x.min(point.x());
            max_x = max_x.max(point.x());
            min_y = min_y.min(point.y());
            max_y = max_y.max(point.y());
        }

        // Convert buffer from meters to degrees (approximate)
        let buffer_degrees = self.config.max_matching_distance / 111_000.0;

        // Expand bounds with buffer
        geo::Rect::new(
            geo::Coord {
                x: min_x - buffer_degrees,
                y: min_y - buffer_degrees,
            },
            geo::Coord {
                x: max_x + buffer_degrees,
                y: max_y + buffer_degrees,
            },
        )
    }

    fn score_segment(
        &mut self,
        seg: &RoadSegment,
        point: Point<f64>,
        prev_point: Option<Point<f64>>,
        timestamp: DateTime<Utc>,
        prev_timestamp: Option<DateTime<Utc>>,
    ) -> Option<Candidate> {
        // Calculate projection details
        let (projection, distance) = self.project_to_segment(point, seg);

        if distance > self.config.max_matching_distance {
            return None;
        }

        // Calculate heading similarity - with increased importance
        let seg_heading = seg.direction_at_point(projection);
        let user_heading = prev_point.map(|pp| calculate_heading(pp, point));
        let heading_diff = user_heading
            .map(|uh| angle_difference(uh, seg_heading))
            .unwrap_or(0.0);

        // Give higher penalty to large heading differences
        let adjusted_heading_diff = if heading_diff > 90.0 {
            // Exponentially increase penalty for heading differences over 90 degrees
            heading_diff * 1.5
        } else {
            heading_diff
        };

        // Calculate speed consistency
        let speed = prev_point
            .zip(prev_timestamp)
            .and_then(|(pp, pt)| calculate_speed(pp, point, timestamp, pt));

        let speed_diff = speed
            .map(|s| {
                let max_speed = seg.max_speed.unwrap_or(50.0); // Default 50 km/h if not specified
                (s - max_speed).abs() / max_speed
            })
            .unwrap_or(0.0);

        // Calculate road class score (prioritize major roads)
        let road_class_score = calculate_road_class_score(&seg.highway_type);

        Some(Candidate {
            segment: seg.clone(),
            point_on_edge: projection,
            distance,
            heading_diff: adjusted_heading_diff,
            speed: speed_diff,
            weights: (
                self.config.distance_weight,
                self.config.heading_weight,
                self.config.speed_weight,
            ),
            road_class_score,
        })
    }

    // Enhanced loop detection along path
    fn detect_loop_along_path(
        &self,
        new_segments: &[u64],
        existing_path: &[u64],
        loaded_segments: &HashMap<u64, RoadSegment>,
    ) -> bool {
        // Combine paths for checking
        let mut full_path = existing_path.to_vec();
        full_path.extend_from_slice(new_segments);

        // Basic loop detection: segment ID repetition
        let mut seen_segments = HashSet::new();
        for &seg_id in &full_path {
            if !seen_segments.insert(seg_id) {
                return true;
            }
        }

        // Advanced spatial loop detection
        // Only do this check if we have enough segments
        if full_path.len() >= 8 {
            // Get coordinates for all segments
            let mut segment_coords = Vec::new();

            for &seg_id in &full_path {
                if let Some(segment) = loaded_segments.get(&seg_id) {
                    // Add segment start and end points
                    if let Some(start) = segment.coordinates.first() {
                        segment_coords.push((start.x, start.y));
                    }
                    if let Some(end) = segment.coordinates.last() {
                        segment_coords.push((end.x, end.y));
                    }
                }
            }

            // Check for any point that's very close to an earlier point (potential loop)
            let threshold = self.config.loop_distance_threshold / 111000.0; // Convert meters to approximate degrees

            for i in 0..segment_coords.len() {
                let (x1, y1) = segment_coords[i];

                // Only check against points that are far enough away in the sequence
                // to avoid detecting legitimate curves as loops
                for j in 0..i.saturating_sub(5) {
                    let (x2, y2) = segment_coords[j];

                    // Simple Euclidean distance check (approximation of Haversine for small distances)
                    let dist_squared = (x1 - x2).powi(2) + (y1 - y2).powi(2);
                    if dist_squared < threshold.powi(2) {
                        debug!(
                            "Spatial loop detected: points ({}, {}) and ({}, {}) are very close",
                            x1, y1, x2, y2
                        );
                        return true;
                    }
                }
            }
        }

        false
    }

    fn project_to_segment(&self, point: Point<f64>, seg: &RoadSegment) -> (Point<f64>, f64) {
        let line = LineString::from(seg.coordinates.clone());
        match line.closest_point(&point) {
            Closest::SinglePoint(projected) => {
                // Use Haversine for spherical distance
                let distance = Haversine.distance(point, projected);
                (projected, distance)
            }
            _ => (point, f64::MAX),
        }
    }

    fn get_segment(
        &mut self,
        segment_id: u64,
        loaded_segments: &mut HashMap<u64, RoadSegment>,
    ) -> Result<RoadSegment> {
        if let Some(segment) = loaded_segments.get(&segment_id) {
            return Ok(segment.clone());
        }

        // Not in memory, load from tile with robust error handling
        match self.tile_loader.get_segment(segment_id) {
            Ok(segment) => {
                // Cache the segment for future use
                loaded_segments.insert(segment_id, segment.clone());
                Ok(segment)
            }
            Err(e) => {
                warn!(
                    "Failed to get segment {}: {}. Will try to search in all tiles.",
                    segment_id, e
                );

                // Try to find the segment in any tile using the extended search
                let tile_id = self.tile_loader.find_segment_tile(segment_id)?;
                let tile = self.tile_loader.load_tile(&tile_id)?;

                // Now find segment in the loaded tile
                if let Some(segment) = tile.road_segments.iter().find(|s| s.id == segment_id) {
                    // Cache the segment for future use
                    loaded_segments.insert(segment_id, segment.clone());
                    return Ok(segment.clone());
                }

                Err(anyhow!(
                    "Segment {} not found even after extended search",
                    segment_id
                ))
            }
        }
    }

    // Helper method to get segment map for a list of segment IDs
    fn get_segment_map(&mut self, segment_ids: &[u64]) -> Result<HashMap<u64, RoadSegment>> {
        let mut segments = HashMap::new();

        for &seg_id in segment_ids {
            // We need to use a mutable reference for the get_segment method
            // but we want to avoid modifying the original segments map
            // This is a bit of a hack, but it works
            let segment = self.tile_loader.get_segment(seg_id)?;
            segments.insert(seg_id, segment);
        }

        Ok(segments)
    }

    // Post-process the matched path to remove or fix obvious errors
    fn post_process_path(
        &self,
        path: &[u64],
        loaded_segments: &HashMap<u64, RoadSegment>,
    ) -> Vec<u64> {
        if path.is_empty() {
            return Vec::new();
        }

        let mut cleaned_path = Vec::new();
        let mut i = 0;

        // Process each segment
        while i < path.len() {
            let current_id = path[i];
            cleaned_path.push(current_id);

            // Look ahead for loops and local detours
            if i + 3 < path.len() {
                let mut loop_detected = false;
                let current_seg = loaded_segments.get(&current_id);

                // Check if we're on a major road
                if let Some(seg) = current_seg {
                    let road_class = calculate_road_class_score(&seg.highway_type);

                    // Aggressive loop detection for minor roads
                    let detection_threshold = if road_class > 3.0 { 5 } else { 15 };

                    // For minor roads, we're more aggressive about detecting loops
                    if road_class > 3.0 {
                        // Look ahead for quick return to same segment on minor roads
                        let local_path =
                            &path[i..std::cmp::min(i + detection_threshold, path.len())];
                        if local_path.iter().filter(|&&id| id == current_id).count() > 1 {
                            debug!("Minor road loop detected in short span");
                            loop_detected = true;
                        }
                    }

                    if road_class < 3.0 {
                        // If we're on a major road
                        // Look ahead for small local detours
                        let mut j = i + 1;
                        let mut detour_ids = Vec::new();
                        let mut detour_class_sum = 0.0;
                        let mut detour_length = 0;

                        // Collect potential detour segments
                        while j < path.len() && detour_length < detection_threshold {
                            let seg_id = path[j];
                            if seg_id == current_id {
                                // We've returned to the same segment - definitely a loop
                                loop_detected = true;
                                break;
                            }

                            if let Some(seg) = loaded_segments.get(&seg_id) {
                                let seg_class = calculate_road_class_score(&seg.highway_type);

                                // If we hit another major road, it might not be a detour
                                if seg_class < 3.0 && seg.name != seg.name {
                                    break;
                                }

                                detour_ids.push(seg_id);
                                detour_class_sum += seg_class;
                                detour_length += 1;
                            }

                            j += 1;
                        }

                        // Determine if it's a detour to a lower road class
                        if loop_detected
                            || (detour_length > 1
                                && detour_class_sum / detour_length as f64 > road_class + 3.0)
                        {
                            debug!(
                                "Detected detour/loop of {} segments from major road",
                                detour_length
                            );

                            // Skip the detour
                            i = j;
                            continue;
                        }
                    }

                    // Check for spatial loops by looking for segments that nearly touch earlier segments
                    let current_coords = seg.coordinates.clone();
                    let mut spatial_loop = false;

                    // Look ahead for segments that come back near this segment
                    let look_ahead = std::cmp::min(15, path.len() - i - 1);
                    if look_ahead > 4 {
                        for j in 2..look_ahead {
                            let ahead_id = path[i + j];
                            if let Some(ahead_seg) = loaded_segments.get(&ahead_id) {
                                // Calculate minimum distance between segment endpoints
                                let min_distance = self.calculate_min_segment_distance(
                                    &current_coords,
                                    &ahead_seg.coordinates,
                                );

                                if min_distance < self.config.loop_distance_threshold {
                                    debug!(
                                        "Spatial loop detected: segment {} is very close to segment {}",
                                        ahead_id, current_id
                                    );
                                    spatial_loop = true;
                                    break;
                                }
                            }
                        }
                    }

                    if spatial_loop {
                        // Skip ahead to the next segment
                        i += 1;
                        continue;
                    }
                }
            }

            i += 1;
        }

        // Additional post-processing: remove repeated segments that aren't directly adjacent
        let mut final_path = Vec::new();
        let mut recent_segments = HashSet::new();

        for &seg_id in &cleaned_path {
            recent_segments.insert(seg_id);
            final_path.push(seg_id);

            // Keep the set limited to recent segments to allow reuse of same segment after a while
            if recent_segments.len() > 50 {
                if let Some(oldest) = final_path.get(final_path.len().saturating_sub(50)) {
                    recent_segments.remove(oldest);
                }
            }
        }

        final_path
    }

    // Helper method to calculate minimum distance between two sets of coordinates
    fn calculate_min_segment_distance(
        &self,
        coords1: &[geo::Coord<f64>],
        coords2: &[geo::Coord<f64>],
    ) -> f64 {
        let mut min_distance = f64::MAX;

        for c1 in coords1 {
            let p1 = Point::new(c1.x, c1.y);
            for c2 in coords2 {
                let p2 = Point::new(c2.x, c2.y);
                let distance = Haversine.distance(p1, p2);
                min_distance = min_distance.min(distance);
            }
        }

        min_distance
    }

    /// Generates a GeoJSON file for debugging purposes
    /// Shows GPS points and candidate segments with unique colors for each point
    pub fn generate_debug_geojson(
        &mut self,
        gps_points: &[Point<f64>],
        timestamps: &[DateTime<Utc>],
        output_path: &Path,
    ) -> Result<()> {
        info!("Generating debug GeoJSON with candidate segments");

        // Define a set of colors for different candidate groups
        let colors = [
            "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf",
            "#999999", "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
            "#b3b3b3", "#8dd3c7", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5",
        ];

        let mut features = Vec::new();
        let mut candidate_groups = Vec::new();

        // First collect all candidate segments for each GPS point
        for (i, point) in gps_points.iter().enumerate() {
            if i % 10 == 0 {
                info!(
                    "Processing debug data for point {}/{}",
                    i + 1,
                    gps_points.len()
                );
            }

            let time = timestamps
                .get(i)
                .ok_or_else(|| anyhow::anyhow!("Missing timestamp"))?;

            let prev_point = if i > 0 { Some(gps_points[i - 1]) } else { None };
            let prev_timestamp = if i > 0 { Some(timestamps[i - 1]) } else { None };

            // Load tiles for this point
            let point_tiles = self.tile_loader.find_tiles_for_coordinate(*point)?;

            // Find candidates without loop prevention (to show all possible candidates)
            let candidates = self.find_debug_candidates_in_tiles(
                *point,
                prev_point,
                *time,
                prev_timestamp,
                &point_tiles,
            )?;

            candidate_groups.push(candidates);
        }

        // Add GPS points as GeoJSON points
        for (i, point) in gps_points.iter().enumerate() {
            let color_idx = i % colors.len();

            // Add the GPS point as a feature
            features.push(json!({
                "type": "Feature",
                "properties": {
                    "type": "gps_point",
                    "point_index": i,
                    "color": colors[color_idx],
                    "radius": 5,
                    "description": format!("GPS Point {}", i)
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [point.x(), point.y()]
                }
            }));
        }

        // Add candidates as GeoJSON LineStrings
        for (i, candidates) in candidate_groups.iter().enumerate() {
            let color_idx = i % colors.len();

            for (j, candidate) in candidates.iter().enumerate() {
                // Create a feature for each candidate segment
                let coordinates: Vec<Vec<f64>> = candidate
                    .segment
                    .coordinates
                    .iter()
                    .map(|coord| vec![coord.x, coord.y])
                    .collect();

                features.push(json!({
                    "type": "Feature",
                    "properties": {
                        "type": "candidate",
                        "point_index": i,
                        "candidate_index": j,
                        "color": colors[color_idx],
                        "segment_id": candidate.segment.id,
                        "score": candidate.total_score(),
                        "distance": candidate.distance,
                        "heading_diff": candidate.heading_diff,
                        "road_type": candidate.segment.highway_type,
                        "road_name": candidate.segment.name,
                        "description": format!(
                            "Candidate {} for point {} (score: {:.2}, dist: {:.2}m, heading_diff: {:.2}°)",
                            j, i, candidate.total_score(), candidate.distance, candidate.heading_diff
                        )
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coordinates
                    }
                }));

                // Add a marker for the point on the edge
                features.push(json!({
                    "type": "Feature",
                    "properties": {
                        "type": "projection",
                        "point_index": i,
                        "candidate_index": j,
                        "color": colors[color_idx],
                        "radius": 3,
                        "description": format!("Projection for point {} on candidate {}", i, j)
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [candidate.point_on_edge.x(), candidate.point_on_edge.y()]
                    }
                }));
            }
        }

        // Create the final GeoJSON object
        let geojson = json!({
            "type": "FeatureCollection",
            "features": features
        });

        // Write to file
        let mut file = File::create(output_path)?;
        file.write_all(serde_json::to_string_pretty(&geojson)?.as_bytes())?;

        info!(
            "Debug GeoJSON with {} features written to {:?}",
            features.len(),
            output_path
        );

        Ok(())
    }

    // Special version of find_candidates that doesn't apply loop prevention
    // so we can see all potential candidates for each point
    fn find_debug_candidates_in_tiles(
        &mut self,
        point: Point<f64>,
        prev_point: Option<Point<f64>>,
        timestamp: DateTime<Utc>,
        prev_timestamp: Option<DateTime<Utc>>,
        tiles: &std::collections::HashSet<String>,
    ) -> Result<Vec<Candidate>> {
        let mut candidates = Vec::new();

        // Search across all loaded tiles
        for tile_id in tiles {
            let segments = {
                let tile = self.tile_loader.load_tile(tile_id)?;
                tile.road_segments.clone() // Clone segments to break borrow
            };

            for seg in &segments {
                // Score the segment
                if let Some(cand) =
                    self.score_segment(seg, point, prev_point, timestamp, prev_timestamp)
                {
                    // Only add if within max distance
                    if cand.distance <= self.config.max_matching_distance {
                        candidates.push(cand);
                    }
                }
            }
        }

        // Sort candidates
        candidates.sort_by(|a, b| a.total_score().partial_cmp(&b.total_score()).unwrap());

        // Return all candidates for debugging (don't limit)
        Ok(candidates)
    }
}

// Helper function moved outside of MapMatcher impl to avoid borrow checker issues
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

    // Combine factors with increased turn weighting
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

fn calculate_speed(
    from: Point<f64>,
    to: Point<f64>,
    timestamp: DateTime<Utc>,
    prev_time: DateTime<Utc>,
) -> Option<f64> {
    let distance = Haversine.distance(from, to); // Meters
    let duration = (timestamp - prev_time).num_seconds() as f64;

    if duration > 0.0 {
        Some((distance / duration) * 3.6) // Convert m/s to km/h
    } else {
        None
    }
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

pub fn matched_route_to_geojson(
    gps_points: &[Point<f64>],
    matched_segments: &[RoadSegment],
    _candidates_per_point: &[Vec<Candidate>], // Prefix with underscore to indicate intentionally unused
) -> Value {
    let mut features = Vec::new();

    // Add matched route as a LineString
    let mut route_coordinates = Vec::new();
    for segment in matched_segments {
        for coord in &segment.coordinates {
            route_coordinates.push(vec![coord.x, coord.y]);
        }
    }

    features.push(json!({
        "type": "Feature",
        "properties": {
            "type": "matched_route",
            "color": "#ff0000",
            "weight": 4,
            "description": "Matched Route"
        },
        "geometry": {
            "type": "LineString",
            "coordinates": route_coordinates
        }
    }));

    // Add GPS points
    for (i, point) in gps_points.iter().enumerate() {
        features.push(json!({
            "type": "Feature",
            "properties": {
                "type": "gps_point",
                "point_index": i,
                "color": "#000000",
                "radius": 5,
                "description": format!("GPS Point {}", i)
            },
            "geometry": {
                "type": "Point",
                "coordinates": [point.x(), point.y()]
            }
        }));
    }

    // Note: We're not using the candidates_per_point parameter here as it would require
    // more complex tracking of which candidates were selected for the final route

    json!({
        "type": "FeatureCollection",
        "features": features
    })
}
