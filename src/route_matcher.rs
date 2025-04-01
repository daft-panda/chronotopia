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
    mapmatcher::{TileConfig, calculate_heading},
    osm_preprocessing::{OsmProcessor, WaySegment},
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
    pub(crate) segments: Vec<WaySegment>,
    pub(crate) bridge: bool,
}

pub struct RouteMatchJob {
    // inputs
    pub(crate) gps_points: Vec<Point<f64>>,
    pub(crate) timestamps: Vec<DateTime<Utc>>,
    debug_way_ids: Option<Vec<u64>>,
    // state
    graph: RefCell<Option<UnGraphMap<u64, f64>>>,
    segment_map: RefCell<HashMap<u64, WaySegment>>,
    loaded_tiles: RefCell<HashSet<String>>,
    // trackers
    all_candidates: RefCell<Vec<Vec<SegmentCandidate>>>,
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
        if job.tracing {
            for i in 0..job.gps_points.len() {
                let mut pc_geojson = job.point_candidates_geojson.borrow_mut();
                pc_geojson.push(self.debug_point_candidates(job, i)?);
            }
        }

        Ok(())
    }

    /// Build route using sliding window approach with improved loop handling
    /// and strict connectivity requirements
    fn build_route_with_sliding_window(&mut self, job: &RouteMatchJob) -> Result<Vec<WaySegment>> {
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
        let mut last_end_segment: Option<WaySegment> = None;

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

            // Match this window using the improved loop-aware method with connectivity enforcement
            let window_route = self.match_window_avoiding_loops(
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

                    // Use the loop-aware matching for the smaller window too
                    let smaller_route = self.match_window_avoiding_loops(
                        job,
                        window_start,
                        smaller_end,
                        last_end_segment.as_ref(),
                    )?;

                    if !smaller_route.is_empty() {
                        // Verify connectivity
                        let is_connected = if let Some(ref last_seg) = last_end_segment {
                            Self::is_connected_to_previous(&smaller_route, Some(last_seg))
                        } else {
                            true
                        };

                        if is_connected {
                            // Add to complete route
                            let new_segments =
                                self.new_segments_for_route(&complete_route, smaller_route);
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
                        } else {
                            debug!("Smaller window route lacks connectivity - skipping");
                        }
                    }
                }

                // If we can't find any route, we skip this GPS point
                window_start += 1;
                continue;
            }

            // Verify connectivity to previous segment if we have one
            if let Some(ref last_seg) = last_end_segment {
                if !Self::is_connected_to_previous(&window_route, Some(last_seg)) {
                    debug!(
                        "Window route lacks connectivity to previous segment - skipping this window"
                    );
                    window_start += 1;
                    continue;
                }
            }

            // Check if adding this window would create a loop
            let potential_new_segments =
                self.new_segments_for_route(&complete_route, window_route.clone());

            if !potential_new_segments.is_empty() {
                let combined_route = [&complete_route[..], &potential_new_segments[..]].concat();

                if self.detect_loop(&combined_route).is_some() {
                    debug!("Adding window would create a loop - skipping this window");
                    window_start += 1;
                    continue;
                }

                // Add window route to complete route
                if let Some(ref mut window_traces) = window_tracing {
                    window_traces.push(WindowTrace {
                        start: window_start,
                        end: window_end,
                        segments: potential_new_segments.clone(),
                        ..Default::default()
                    });
                }
                complete_route.extend(potential_new_segments);

                // Update last segment for next window
                last_end_segment = complete_route.last().cloned();

                // Move window forward
                window_start += step_size;
            } else {
                // No new segments added (likely duplicate with last window)
                window_start += 1;
            }
        }

        // Final check for loops - should not be needed with our approach
        if let Some(loop_info) = self.detect_loop(&complete_route) {
            // This should rarely happen with our improved algorithm
            warn!(
                "Final check found a loop at positions {} and {} (segment ID: {})",
                loop_info.first_pos, loop_info.second_pos, loop_info.segment_id
            );

            // Remove the loop by keeping everything before the loop's second occurrence
            let mut clean_route = complete_route[..loop_info.second_pos].to_vec();

            // Try to bridge the gap if possible, maintaining connectivity
            if loop_info.second_pos < complete_route.len() - 1 {
                let after_loop = &complete_route[loop_info.second_pos + 1..];
                let last_clean = clean_route.last().cloned();

                if let Some(last) = last_clean {
                    if !after_loop.is_empty() {
                        let first_after = &after_loop[0];

                        // Check if we can connect directly
                        if last.connections.contains(&first_after.id)
                            || first_after.connections.contains(&last.id)
                        {
                            clean_route.extend_from_slice(after_loop);
                        }
                    }
                }
            }

            debug!(
                "Final route has {} segments after loop removal",
                clean_route.len()
            );
            return Ok(clean_route);
        }

        if complete_route.is_empty() {
            return Err(anyhow!("Failed to build valid route"));
        }

        Ok(complete_route)
    }

    // Add this code to the implementation of RouteMatcher

    /// Handle loop detection and alternative path selection during window matching
    /// Returns the best route without loops for the window that maintains connectivity
    fn match_window_avoiding_loops(
        &mut self,
        job: &RouteMatchJob,
        start_idx: usize,
        end_idx: usize,
        previous_segment: Option<&WaySegment>,
    ) -> Result<Vec<WaySegment>> {
        // First attempt at matching the window
        let initial_route =
            self.match_window_with_context(job, start_idx, end_idx, previous_segment)?;

        // If we have no previous segment, or the route is empty, return the initial result
        if previous_segment.is_none() || initial_route.is_empty() {
            return Ok(initial_route);
        }

        // Check if the initial route contains loops
        if let Some(loop_info) = self.detect_loop(&initial_route) {
            debug!(
                "Loop detected in window {}-{}: segment {} appears at positions {} and {}",
                start_idx, end_idx, loop_info.segment_id, loop_info.first_pos, loop_info.second_pos
            );

            // Verify the first segment is connected to the previous segment
            if !Self::is_connected_to_previous(&initial_route, previous_segment) {
                debug!(
                    "Initial route is not connected to previous segment - will enforce connectivity"
                );
            }

            // Try to fix the loop by blacklisting the problematic segment at the second position
            // BUT preserve connectivity with the previous segment
            let blacklisted_segments = vec![(loop_info.second_pos, loop_info.segment_id)];

            // Re-match with blacklisted segments and strict connectivity requirement
            let alternative_route = self.match_window_with_connectivity(
                job,
                start_idx,
                end_idx,
                previous_segment,
                &blacklisted_segments,
            )?;

            // Check if we still have loops
            if self.detect_loop(&alternative_route).is_none() {
                debug!(
                    "Successfully resolved loop in window {}-{} with connectivity preserved",
                    start_idx, end_idx
                );
                return Ok(alternative_route);
            }

            // If multiple loops or couldn't fix, try a more aggressive approach
            // but still maintain connectivity
            debug!("Multiple loops detected, trying aggressive fix with connectivity preservation");
            let all_segments_in_loop = self.get_all_loop_segments(&initial_route);

            // Blacklist all segments in loops for the second occurrence points
            let mut comprehensive_blacklist = Vec::new();
            for &seg_id in &all_segments_in_loop {
                if let Some(positions) = self.find_segment_positions(&initial_route, seg_id) {
                    if positions.len() > 1 {
                        // Blacklist from second occurrence onwards
                        for &pos in &positions[1..] {
                            comprehensive_blacklist.push((pos, seg_id));
                        }
                    }
                }
            }

            // Try matching with comprehensive blacklist and enforced connectivity
            let comprehensive_fix = self.match_window_with_connectivity(
                job,
                start_idx,
                end_idx,
                previous_segment,
                &comprehensive_blacklist,
            )?;

            if !comprehensive_fix.is_empty() && self.detect_loop(&comprehensive_fix).is_none() {
                debug!(
                    "Successfully resolved all loops with comprehensive blacklist while preserving connectivity"
                );
                return Ok(comprehensive_fix);
            }

            // As a last resort, try a minimal window with just the first GPS point
            // but strict connectivity to the previous segment
            if start_idx + 1 <= end_idx {
                debug!(
                    "Attempting minimal window with first GPS point only to maintain connectivity"
                );
                let minimal_route =
                    self.match_minimal_window_with_connectivity(job, start_idx, previous_segment)?;

                if !minimal_route.is_empty() && self.detect_loop(&minimal_route).is_none() {
                    debug!("Successfully found minimal connected route for first point only");
                    return Ok(minimal_route);
                }
            }

            // If all else fails but we still need connectivity, try to find a minimal
            // directly-connected segment from previous
            if !initial_route.is_empty() {
                debug!("Using fallback: selecting a directly connected segment from previous");
                let fallback_route = self.select_fallback_segment(job, previous_segment)?;
                if !fallback_route.is_empty() {
                    return Ok(fallback_route);
                }
            }
        }

        // Check if the initial route has connectivity issues
        if !initial_route.is_empty()
            && previous_segment.is_some()
            && !Self::is_connected_to_previous(&initial_route, previous_segment)
        {
            debug!("Initial route lacks connectivity to previous segment");

            // Try to find a route with proper connectivity
            let connected_route = self.match_window_with_connectivity(
                job,
                start_idx,
                end_idx,
                previous_segment,
                &[], // No blacklist
            )?;

            if !connected_route.is_empty() {
                debug!("Found alternative route with proper connectivity");
                return Ok(connected_route);
            }

            // If still no connectivity, try a minimal window
            if start_idx + 1 <= end_idx {
                let minimal_route =
                    self.match_minimal_window_with_connectivity(job, start_idx, previous_segment)?;

                if !minimal_route.is_empty() {
                    debug!("Found minimal connected route for first point only");
                    return Ok(minimal_route);
                }
            }

            // Last resort: try to find any directly connected segment from previous
            debug!("Using fallback: selecting a directly connected segment from previous");
            return self.select_fallback_segment(job, previous_segment);
        }

        // If no loops and connectivity is good (or no previous segment), return the initial route
        Ok(initial_route)
    }

    /// Match a window with enforced connectivity to the previous segment
    fn match_window_with_connectivity(
        &mut self,
        job: &RouteMatchJob,
        start_idx: usize,
        end_idx: usize,
        previous_segment: Option<&WaySegment>,
        blacklisted_segments: &[(usize, u64)], // (relative position in window, segment_id)
    ) -> Result<Vec<WaySegment>> {
        // Early return if no previous segment (connectivity not enforceable)
        if previous_segment.is_none() {
            return self.match_window_with_context(job, start_idx, end_idx, previous_segment);
        }

        // Get directly connected segments to the previous segment
        let prev_seg = previous_segment.unwrap();
        let segment_map = job.segment_map.borrow();
        let mut connected_segments = Vec::new();

        for &conn_id in &prev_seg.connections {
            if let Some(segment) = segment_map.get(&conn_id) {
                connected_segments.push(segment.clone());
            }
        }

        if connected_segments.is_empty() {
            debug!("Previous segment has no connected segments in the map!");
            return self.match_window_with_context(job, start_idx, end_idx, previous_segment);
        }

        // Get candidates for first point in window
        let window_candidates = job.all_candidates.borrow()[start_idx..=end_idx].to_vec();
        let first_point_candidates = &window_candidates[0];

        // Find candidates that are directly connected to previous segment
        let mut directly_connected_candidates = Vec::new();

        for candidate in first_point_candidates {
            if prev_seg.connections.contains(&candidate.segment.id)
                || candidate.segment.connections.contains(&prev_seg.id)
            {
                directly_connected_candidates.push(candidate.clone());
            }
        }

        // If no directly connected candidates, check distance to connected segments
        // and add them as potential candidates if they're close enough
        if directly_connected_candidates.is_empty() {
            debug!("No directly connected candidates for first point, checking proximities");

            for segment in &connected_segments {
                // Check distance from GPS point to this segment
                let point = job.gps_points[start_idx];
                let projection = self.project_point_to_segment(point, segment);
                let distance = Haversine.distance(point, projection);

                // If within reasonable distance (75m), add as candidate
                if distance <= 75.0 {
                    let score = distance / 75.0; // Score based on distance
                    let new_candidate = SegmentCandidate {
                        segment: segment.clone(),
                        distance,
                        projection,
                        score,
                    };

                    directly_connected_candidates.push(new_candidate);
                }
            }
        }

        // If we still have no connected candidates, allow a small bridge/gap
        // For example, we could be at an intersection where OSM segments don't perfectly connect
        if directly_connected_candidates.is_empty() {
            debug!("No connected candidates found, will allow a short bridge");

            // Try to find segments close to both the previous segment end and the GPS point
            let prev_end = match prev_seg.coordinates.last() {
                Some(coord) => geo::Point::new(coord.x, coord.y),
                None => return Err(anyhow!("Previous segment has no coordinates")),
            };

            let current_point = job.gps_points[start_idx];

            for candidate in first_point_candidates {
                let candidate_start = match candidate.segment.coordinates.first() {
                    Some(coord) => geo::Point::new(coord.x, coord.y),
                    None => continue,
                };

                // Calculate distance from previous segment end to candidate start
                let gap_distance = Haversine.distance(prev_end, candidate_start);

                // Only allow short bridges (less than 30m) and non-parallel paths
                if gap_distance <= 30.0 {
                    // Check if the bridge would be parallel to the previous segment
                    // by comparing headings
                    let prev_heading = prev_seg.direction_at_point(prev_end);
                    let candidate_heading = candidate.segment.direction_at_point(candidate_start);
                    let heading_diff = angle_difference(prev_heading, candidate_heading).abs();

                    // If headings are similar (within 30 degrees), segments might be parallel
                    // which we want to avoid
                    if heading_diff > 30.0 {
                        directly_connected_candidates.push(candidate.clone());
                    }
                }
            }
        }

        // If we still have no candidates, return empty route (we'll skip this point)
        if directly_connected_candidates.is_empty() {
            debug!("Cannot find any suitable connected candidates, returning empty route");
            return Ok(Vec::new());
        }

        // Sort connected candidates by score
        directly_connected_candidates.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Construct a new list of candidates with connectivity enforcement
        let mut modified_candidates = window_candidates.clone();
        modified_candidates[0] = directly_connected_candidates;

        // Apply blacklist to other points in window
        for &(rel_pos, segment_id) in blacklisted_segments {
            if rel_pos > 0 && rel_pos < modified_candidates.len() {
                modified_candidates[rel_pos].retain(|c| c.segment.id != segment_id);

                // If we emptied a candidate list, just put back the original but deprioritize the blacklisted
                if modified_candidates[rel_pos].is_empty() {
                    let mut adjusted = window_candidates[rel_pos].clone();
                    for candidate in &mut adjusted {
                        if candidate.segment.id == segment_id {
                            candidate.score *= 2.0; // Penalize score
                        }
                    }
                    modified_candidates[rel_pos] = adjusted;
                }
            }
        }

        // Find best route with the modified candidates
        let window_points = &job.gps_points[start_idx..=end_idx];
        let window_timestamps = &job.timestamps[start_idx..=end_idx];

        // Try each combination of first and last candidates
        let first_candidates = &modified_candidates[0];
        let last_candidates = &modified_candidates[modified_candidates.len() - 1];

        let mut best_route = Vec::new();
        let mut best_score = f64::MAX;

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
                    &modified_candidates,
                );

                if let Ok((route, score)) = route_result {
                    // Check if route contains any loops
                    if self.detect_loop(&route).is_none() {
                        // Verify first segment is connected to previous
                        let connectivity_score =
                            if Self::is_connected_to_previous(&route, previous_segment) {
                                1.0 // Good connectivity
                            } else {
                                1000.0 // Heavy penalty for disconnected routes
                            };

                        let final_score = score * connectivity_score;

                        if final_score < best_score {
                            best_route = route;
                            best_score = final_score;
                        }
                    }
                }
            }
        }

        Ok(best_route)
    }

    /// Match just the first point in a window with strict connectivity to previous
    fn match_minimal_window_with_connectivity(
        &mut self,
        job: &RouteMatchJob,
        point_idx: usize,
        previous_segment: Option<&WaySegment>,
    ) -> Result<Vec<WaySegment>> {
        // Early return if no previous segment
        if previous_segment.is_none() {
            return Ok(Vec::new());
        }

        let prev_seg = previous_segment.unwrap();
        let point_candidates = &job.all_candidates.borrow()[point_idx];

        // Find candidates directly connected to previous segment
        let mut connected_candidates = Vec::new();

        for candidate in point_candidates {
            if prev_seg.connections.contains(&candidate.segment.id)
                || candidate.segment.connections.contains(&prev_seg.id)
            {
                connected_candidates.push(candidate.clone());
            }
        }

        // If no directly connected candidates, try very nearby segments
        if connected_candidates.is_empty() {
            let segment_map = job.segment_map.borrow();

            for &conn_id in &prev_seg.connections {
                if let Some(segment) = segment_map.get(&conn_id) {
                    // Check distance from GPS point to this segment
                    let point = job.gps_points[point_idx];
                    let projection = self.project_point_to_segment(point, segment);
                    let distance = Haversine.distance(point, projection);

                    // Use a stricter distance threshold for minimal window
                    if distance <= 50.0 {
                        connected_candidates.push(SegmentCandidate {
                            segment: segment.clone(),
                            distance,
                            projection,
                            score: distance / 50.0,
                        });
                    }
                }
            }
        }

        // If we found connected candidates, return the best one
        if !connected_candidates.is_empty() {
            // Sort by score
            connected_candidates.sort_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Return just the best segment
            return Ok(vec![connected_candidates[0].segment.clone()]);
        }

        // If no connected candidates found, return empty
        Ok(Vec::new())
    }

    /// Select a fallback segment directly connected to the previous segment
    fn select_fallback_segment(
        &self,
        job: &RouteMatchJob,
        previous_segment: Option<&WaySegment>,
    ) -> Result<Vec<WaySegment>> {
        if previous_segment.is_none() {
            return Ok(Vec::new());
        }

        let prev_seg = previous_segment.unwrap();
        let segment_map = job.segment_map.borrow();

        // Find the connected segment that best matches the direction of travel
        let mut best_segment = None;
        let mut best_score = f64::MAX;

        // Find the heading of the previous segment at its end
        let prev_coords = &prev_seg.coordinates;
        let prev_end =
            geo::Point::new(prev_coords.last().unwrap().x, prev_coords.last().unwrap().y);

        // If we have at least two points in previous segment, we can get a heading
        let prev_heading = if prev_coords.len() >= 2 {
            let prev_second_to_last = geo::Point::new(
                prev_coords[prev_coords.len() - 2].x,
                prev_coords[prev_coords.len() - 2].y,
            );

            calculate_heading(prev_second_to_last, prev_end)
        } else {
            0.0 // Default if we can't calculate
        };

        // Find the connected segment that best maintains the heading
        for &conn_id in &prev_seg.connections {
            if let Some(segment) = segment_map.get(&conn_id) {
                // Skip if segment is too long (we want a minimal fallback)
                if segment.length() > 100.0 {
                    continue;
                }

                // Get heading of connected segment at its start
                let conn_coords = &segment.coordinates;
                if conn_coords.len() < 2 {
                    continue;
                }

                let conn_start = geo::Point::new(
                    conn_coords.first().unwrap().x,
                    conn_coords.first().unwrap().y,
                );

                let conn_second = geo::Point::new(conn_coords[1].x, conn_coords[1].y);

                let conn_heading = calculate_heading(conn_start, conn_second);

                // Calculate heading difference (smaller is better)
                let heading_diff = angle_difference(prev_heading, conn_heading);

                // Score based on heading difference and segment length
                // We prefer segments that maintain heading but are short
                let score = heading_diff + (segment.length() / 20.0);

                if score < best_score {
                    best_score = score;
                    best_segment = Some(segment.clone());
                }
            }
        }

        if let Some(segment) = best_segment {
            Ok(vec![segment])
        } else {
            Ok(Vec::new())
        }
    }

    /// Detect a loop in a route (if any)
    fn detect_loop(&self, route: &[WaySegment]) -> Option<LoopInfo> {
        let mut seen_ids = HashMap::new();

        for (pos, segment) in route.iter().enumerate() {
            if let Some(first_pos) = seen_ids.get(&segment.id) {
                return Some(LoopInfo {
                    segment_id: segment.id,
                    first_pos: *first_pos,
                    second_pos: pos,
                });
            } else {
                seen_ids.insert(segment.id, pos);
            }
        }

        None
    }

    /// Check if route's first segment is connected to the previous segment
    fn is_connected_to_previous(route: &[WaySegment], previous: Option<&WaySegment>) -> bool {
        if previous.is_none() || route.is_empty() {
            return true; // No connectivity constraint if no previous segment
        }

        let prev = previous.unwrap();
        let first = &route[0];

        // Check direct connection
        if prev.connections.contains(&first.id) || first.connections.contains(&prev.id) {
            return true;
        }

        // If no direct connection, check distance between endpoints
        // Allowing a small gap (e.g., at junctions)
        if let (Some(prev_end), Some(first_start)) =
            (prev.coordinates.last(), first.coordinates.first())
        {
            let prev_point = geo::Point::new(prev_end.x, prev_end.y);
            let first_point = geo::Point::new(first_start.x, first_start.y);

            let gap = Haversine.distance(prev_point, first_point);

            // Allow small gaps (< 20m) only at junction points
            if gap < 20.0 {
                // Check if either point is a junction
                let is_junction = prev.connections.len() > 1 || first.connections.len() > 1;
                return is_junction;
            }
        }

        false
    }

    /// Find all positions of a segment in a route
    fn find_segment_positions(&self, route: &[WaySegment], segment_id: u64) -> Option<Vec<usize>> {
        let positions: Vec<usize> = route
            .iter()
            .enumerate()
            .filter_map(|(i, s)| if s.id == segment_id { Some(i) } else { None })
            .collect();

        if positions.is_empty() {
            None
        } else {
            Some(positions)
        }
    }

    /// Get all segment IDs involved in loops
    fn get_all_loop_segments(&self, route: &[WaySegment]) -> HashSet<u64> {
        let mut seen_ids = HashMap::new();
        let mut loop_segments = HashSet::new();

        for (pos, segment) in route.iter().enumerate() {
            if let Some(first_pos) = seen_ids.get(&segment.id) {
                // Found a loop, add all segments in the loop
                for i in *first_pos..=pos {
                    loop_segments.insert(route[i].id);
                }
            } else {
                seen_ids.insert(segment.id, pos);
            }
        }

        loop_segments
    }

    /// Match a window of GPS points, considering context from previous window
    fn match_window_with_context(
        &mut self,
        job: &RouteMatchJob,
        start_idx: usize,
        end_idx: usize,
        previous_segment: Option<&WaySegment>,
    ) -> Result<Vec<WaySegment>> {
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
    ) -> Result<Vec<WaySegment>> {
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
    ) -> Result<Vec<WaySegment>> {
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
        from_segment: &WaySegment,
        to_segment: &WaySegment,
        window_points: &[Point<f64>],
        window_timestamps: &[DateTime<Utc>],
        window_candidates: &[Vec<SegmentCandidate>],
    ) -> Result<(Vec<WaySegment>, f64)> {
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

    /// Find a route through specified waypoint segments
    fn find_path_via_segment_waypoints(
        &self,
        job: &RouteMatchJob,
        from_id: u64,
        to_id: u64,
        waypoint_ids: &HashSet<u64>,
        max_distance: f64,
    ) -> Result<(Vec<WaySegment>, f64)> {
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
fn calculate_transition_cost(from_seg: &WaySegment, to_seg: &WaySegment) -> f64 {
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
    segment: WaySegment,
    distance: f64,          // Distance from GPS point to segment
    projection: Point<f64>, // Projected point on segment
    score: f64,             // Overall score (lower is better)
}

/// Helper struct to describe a detected loop
struct LoopInfo {
    segment_id: u64,
    first_pos: usize,
    second_pos: usize,
}
