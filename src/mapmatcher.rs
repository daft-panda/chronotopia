use anyhow::{Result, anyhow, bail};
use chrono::{DateTime, Utc};
use geo::{Closest, ClosestPoint, Haversine, LineString, algorithm::Distance};
use geo_types::Point;
use log::{debug, info, trace, warn};
use ordered_float::OrderedFloat;
use petgraph::prelude::{EdgeRef, UnGraphMap};
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet},
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
        // Include road class in scoring
        dist_w * self.distance
            + head_w * self.heading_diff
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

            let point_candidates = self.find_candidates_in_tiles(
                *point,
                prev_point,
                *time,
                prev_timestamp,
                &loaded_tiles,
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

                            // Check for loops in the path
                            let loop_penalty = if self.detect_loop(&best_path, &loaded_segments) {
                                debug!("Applying loop penalty to path");
                                -self.config.loop_penalty_weight // Heavy penalty for paths that create loops
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

            trellis.push(current_layer);
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
    ) -> Result<Vec<Candidate>> {
        let mut candidates = Vec::new();
        let mut high_priority_candidates = Vec::new();

        // Search across all loaded tiles
        for tile_id in tiles {
            let segments = {
                let tile = self.tile_loader.load_tile(tile_id)?;
                tile.road_segments.clone() // Clone segments to break borrow
            };

            for seg in &segments {
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

        // Calculate heading similarity
        let seg_heading = seg.direction_at_point(projection);
        let user_heading = prev_point.map(|pp| calculate_heading(pp, point));
        let heading_diff = user_heading
            .map(|uh| angle_difference(uh, seg_heading))
            .unwrap_or(0.0);

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
            heading_diff,
            speed: speed_diff,
            weights: (
                self.config.distance_weight,
                self.config.heading_weight,
                self.config.speed_weight,
            ),
            road_class_score,
        })
    }

    fn detect_loop(&self, path: &[u64], loaded_segments: &HashMap<u64, RoadSegment>) -> bool {
        // Need at least 4 segments to form a loop
        if path.len() < 4 {
            return false;
        }

        // 1. Check for direct segment ID repetition (simplest loop detection)
        let recent_window = std::cmp::min(1000, path.len());
        let recent_segments = &path[path.len() - recent_window..];

        // Build a segment ID set and check for duplicates
        let mut seen_segments = HashSet::new();
        for &seg_id in recent_segments {
            if !seen_segments.insert(seg_id) {
                // This segment ID has already been seen in our window
                debug!(
                    "Detected loop: segment {} appears twice in recent path",
                    seg_id
                );
                return true;
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

            // Look ahead for loops
            if i + 3 < path.len() {
                let mut loop_detected = false;
                let current_seg = loaded_segments.get(&current_id);

                // Check if we're on a major road
                if let Some(seg) = current_seg {
                    let road_class = calculate_road_class_score(&seg.highway_type);

                    if road_class < 3.0 {
                        // If we're on a major road
                        // Look ahead for small local detours
                        let mut j = i + 1;
                        let mut detour_ids = Vec::new();
                        let mut detour_class_sum = 0.0;
                        let mut detour_length = 0;

                        // Collect potential detour segments
                        while j < path.len() && detour_length < 10 {
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
                }
            }

            i += 1;
        }

        cleaned_path
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

    // Calculate heading change penalty
    let from_heading = from_seg.direction_at_point(from_point);
    let to_heading = to_seg.direction_at_point(to_point);
    let heading_diff = angle_difference(from_heading, to_heading);

    // Penalize sharp turns
    let turn_penalty = if heading_diff > 90.0 {
        3.0 // Major turn penalty
    } else if heading_diff > 45.0 {
        1.5 // Minor turn penalty
    } else {
        1.0 // No penalty for slight turns
    };

    // Consider road type compatibility
    let type_penalty = if from_seg.highway_type == to_seg.highway_type {
        1.0 // Same road type is preferred
    } else {
        // Changing road types has a small penalty
        1.2
    };

    // Combine factors
    (distance / 100.0) * turn_penalty * type_penalty
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
