use anyhow::{Result, anyhow};
use geo::line_intersection::line_intersection;
use geo::{Coord, LineString, Point, algorithm::Distance};
use geo::{Haversine, LineIntersection};
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, info, trace, warn};
use ordered_float::OrderedFloat;
use osmpbf::{Element, ElementReader};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self};
use std::sync::{Arc, Mutex};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
    time::Instant,
};

use crate::route_matcher::{TileConfig, calculate_heading};

/// Represents a road segment in the processed network
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaySegment {
    pub id: u64,
    pub nodes: Vec<u64>,
    pub coordinates: Vec<Coord<f64>>,
    pub is_oneway: bool,
    pub highway_type: String,
    pub max_speed: Option<f64>,
    pub connections: Vec<u64>,
    pub name: Option<String>,
    pub metadata: Option<BTreeMap<String, String>>, // Store additional OSM tags
    pub original_id: Option<u64>,                   // Track original ID for split segments
    pub split_id: Option<u64>,                      // Track split node ID
}

// Default implementation for backward compatibility
impl Default for WaySegment {
    fn default() -> Self {
        Self {
            id: 0,
            nodes: Vec::new(),
            coordinates: Vec::new(),
            is_oneway: false,
            highway_type: String::new(),
            max_speed: None,
            connections: Vec::new(),
            name: None,
            metadata: None,
            original_id: None,
            split_id: None,
        }
    }
}

impl WaySegment {
    pub fn centroid(&self) -> Point<f64> {
        let start = self.coordinates.first().unwrap();
        let end = self.coordinates.last().unwrap();
        Point::new((start.x + end.x) / 2.0, (start.y + end.y) / 2.0)
    }

    pub fn bounding_box(&self) -> geo::Rect<f64> {
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;

        for coord in &self.coordinates {
            min_x = min_x.min(coord.x);
            min_y = min_y.min(coord.y);
            max_x = max_x.max(coord.x);
            max_y = max_y.max(coord.y);
        }

        geo::Rect::new(
            geo::Coord { x: min_x, y: min_y },
            geo::Coord { x: max_x, y: max_y },
        )
    }

    pub fn direction_at_point(&self, point: Point<f64>) -> f64 {
        let line = LineString::from(self.coordinates.clone());
        let nearest_segment = line
            .lines()
            .min_by(|a, b| {
                Haversine
                    .distance(point, a.start_point())
                    .partial_cmp(&Haversine.distance(point, b.start_point()))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|| {
                // Fallback if min_by fails (should be rare)
                line.lines().next().unwrap()
            });

        calculate_heading(
            Point::new(nearest_segment.start.x, nearest_segment.start.y),
            Point::new(nearest_segment.end.x, nearest_segment.end.y),
        )
    }

    pub fn length(&self) -> f64 {
        let mut total = 0.0;
        let line = LineString::from(self.coordinates.clone());

        for segment in line.lines() {
            let start = Point::new(segment.start.x, segment.start.y);
            let end = Point::new(segment.end.x, segment.end.y);
            total += Haversine.distance(start, end);
        }

        total // Length in meters
    }
}

// Tile metadata and index
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TileIndex {
    pub tile_id: String,
    pub bbox: geo_types::Rect<f64>,
    pub road_segments: Vec<WaySegment>,
    pub segment_index: HashMap<u64, String>,
}

/// Memory-efficient OSM processor that uses a streaming approach
pub struct StreamingOsmProcessor {
    config: TileConfig,
    temp_dir: String,
    node_batch_size: usize,
    way_batch_size: usize,
    parallel_processing: bool,
}

impl StreamingOsmProcessor {
    pub fn new(config: TileConfig, temp_dir: String) -> Self {
        Self {
            config,
            temp_dir,
            node_batch_size: 5_000_000, // Process 5M nodes at a time
            way_batch_size: 500_000,    // Process 500K ways at a time
            parallel_processing: true,  // Use parallel processing by default
        }
    }

    /// Process entire OSM PBF file and generate tiles with low memory usage
    pub fn process_pbf(&self, pbf_path: &str, output_dir: &str) -> Result<()> {
        let start_time = Instant::now();

        // Create temp directory if it doesn't exist
        fs::create_dir_all(&self.temp_dir)?;

        info!("Starting memory-efficient OSM processing");

        // Step 1: Extract and process nodes in batches, saving to temporary files
        info!("Processing nodes in batches");
        self.process_nodes_in_batches(pbf_path)?;

        // Step 2: Process ways in batches, loading required nodes
        info!("Processing ways in batches");
        let road_segments = self.process_ways_in_batches(pbf_path)?;

        // Step 3: Process segment intersections (can also be done in batches if needed)
        info!("Processing segment intersections");
        let road_segments = self.process_segment_intersections(road_segments)?;

        // Step 4: Build connectivity
        info!("Building road connectivity graph");
        let road_segments = self.build_connectivity(road_segments)?;

        // Step 5: Generate tiles (can be done in parallel)
        info!("Generating adaptive tiles");
        self.generate_tiles(road_segments, output_dir)?;

        // Clean up temporary files
        self.cleanup_temp_files()?;

        info!("OSM processing completed in {:?}", start_time.elapsed());

        Ok(())
    }

    /// Process nodes in batches to avoid memory issues
    fn process_nodes_in_batches(&self, pbf_path: &str) -> Result<()> {
        let reader = ElementReader::from_path(pbf_path)?;

        let mut batch_num = 0;
        let mut current_nodes = HashMap::with_capacity(self.node_batch_size);

        // First count total nodes to show progress
        info!("Counting nodes for progress tracking");
        let total_nodes = self.count_elements(pbf_path, true, false, false)?;

        let progress_bar = ProgressBar::new(total_nodes);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {percent}% {msg}",
                )?
                .progress_chars("##-"),
        );

        reader.for_each(|element| {
            match element {
                Element::Node(node) => {
                    current_nodes.insert(
                        node.id() as u64,
                        Coord {
                            x: node.lon(),
                            y: node.lat(),
                        },
                    );
                    progress_bar.inc(1);
                }
                Element::DenseNode(node) => {
                    current_nodes.insert(
                        node.id() as u64,
                        Coord {
                            x: node.lon(),
                            y: node.lat(),
                        },
                    );
                    progress_bar.inc(1);
                }
                _ => {}
            }

            // If we've reached batch size, save to disk and clear memory
            if current_nodes.len() >= self.node_batch_size {
                self.save_node_batch_to_disk(&current_nodes, batch_num)
                    .expect("Failed to save node batch");

                batch_num += 1;
                current_nodes.clear();
                progress_bar.set_message(format!("Batch {} saved", batch_num));
            }
        })?;

        // Save any remaining nodes
        if !current_nodes.is_empty() {
            self.save_node_batch_to_disk(&current_nodes, batch_num)?;
            progress_bar.set_message(format!("Final batch {} saved", batch_num));
        }

        progress_bar.finish_with_message(format!(
            "Processed and saved {} node batches",
            batch_num + 1
        ));
        info!(
            "Node processing complete, {} batches saved to disk",
            batch_num + 1
        );

        Ok(())
    }

    /// Save a batch of nodes to disk for later retrieval
    fn save_node_batch_to_disk(
        &self,
        nodes: &HashMap<u64, Coord<f64>>,
        batch_num: usize,
    ) -> Result<()> {
        let file_path = format!("{}/node_batch_{}.bin", self.temp_dir, batch_num);
        let config = bincode::config::standard();
        let serialized_tile = bincode::serde::encode_to_vec(nodes, config)?;

        std::fs::write(&file_path, serialized_tile)?;

        trace!("Saved {} nodes to {}", nodes.len(), file_path);
        Ok(())
    }

    /// Process ways in batches, loading only needed nodes
    fn process_ways_in_batches(&self, pbf_path: &str) -> Result<Vec<WaySegment>> {
        let reader = ElementReader::from_path(pbf_path)?;

        // Count total ways for progress tracking
        let total_ways = self.count_elements(pbf_path, false, true, false)?;

        let progress_bar = ProgressBar::new(total_ways);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[{elapsed_precise}] {bar:40.green/blue} {pos:>7}/{len:7} {percent}% {msg}",
                )?
                .progress_chars("##-"),
        );

        // Important highway types to filter for relevant roads
        let important_highway_types: HashSet<&str> = [
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "motorway_link",
            "trunk_link",
            "primary_link",
            "secondary_link",
            "tertiary_link",
            "residential",
            "unclassified",
            "service",
        ]
        .iter()
        .cloned()
        .collect();

        // List of important tags to preserve
        let important_tags: HashSet<&str> = [
            "highway", "oneway", "maxspeed", "name", "layer", "bridge", "tunnel", "junction",
            "access", "surface", "lanes", "ref", "footway", "cycleway",
        ]
        .iter()
        .cloned()
        .collect();

        let mut all_road_segments = Vec::new();
        let mut batch_num = 0;
        let mut ways_batch = Vec::with_capacity(self.way_batch_size);

        // First pass: collect all relevant ways
        reader.for_each(|element| {
            if let Element::Way(way) = element {
                // Filter for highway ways
                let highway_type = way
                    .tags()
                    .find(|(key, _)| *key == "highway")
                    .map(|(_, value)| value.to_string());

                if let Some(highway) = highway_type {
                    // Skip insignificant highways
                    if !important_highway_types.contains(highway.as_str()) {
                        progress_bar.inc(1);
                        return;
                    }

                    // Collect basic way information
                    let way_id = way.id() as u64;
                    let node_refs: Vec<u64> = way.refs().map(|id| id as u64).collect();

                    // Skip ways with insufficient nodes
                    if node_refs.len() < 2 {
                        progress_bar.inc(1);
                        return;
                    }

                    let is_oneway = way.tags().any(|(key, value)| {
                        (key == "oneway" && (value == "yes" || value == "1"))
                            || (key == "highway" && value == "motorway")
                    });

                    // Extract max speed if available
                    let max_speed =
                        way.tags()
                            .find(|(key, _)| *key == "maxspeed")
                            .and_then(|(_, value)| {
                                let v = value.trim();
                                if v.ends_with(" mph") {
                                    v.trim_end_matches(" mph")
                                        .parse::<f64>()
                                        .ok()
                                        .map(|mph| mph * 1.60934)
                                } else if v.ends_with(" km/h") {
                                    v.trim_end_matches(" km/h").parse::<f64>().ok()
                                } else {
                                    v.parse::<f64>().ok()
                                }
                            });

                    // Extract road name if available
                    let name = way
                        .tags()
                        .find(|(key, _)| *key == "name")
                        .map(|(_, value)| value.to_string());

                    // Collect all important tags into metadata
                    let mut metadata = BTreeMap::new();
                    for (key, value) in way.tags() {
                        if important_tags.contains(key) {
                            metadata.insert(key.to_string(), value.to_string());
                        }
                    }

                    // Store way data for batch processing
                    ways_batch.push((
                        way_id,
                        node_refs,
                        highway.to_string(),
                        is_oneway,
                        max_speed,
                        name,
                        metadata,
                    ));

                    // Process batch if it's full
                    if ways_batch.len() >= self.way_batch_size {
                        let mut segments =
                            self.process_way_batch(&ways_batch, &progress_bar).unwrap();
                        all_road_segments.append(&mut segments);

                        ways_batch.clear();
                        batch_num += 1;

                        progress_bar.set_message(format!("Way batch {} processed", batch_num));
                    }
                }

                progress_bar.inc(1);
            }
        })?;

        // Process any remaining ways in the last batch
        if !ways_batch.is_empty() {
            let mut segments = self.process_way_batch(&ways_batch, &progress_bar)?;
            all_road_segments.append(&mut segments);

            progress_bar.set_message(format!("Final way batch {} processed", batch_num + 1));
        }

        progress_bar.finish_with_message(format!(
            "Processed {} road segments",
            all_road_segments.len()
        ));

        Ok(all_road_segments)
    }

    /// Process a batch of ways, loading only the required nodes
    fn process_way_batch(
        &self,
        ways_batch: &[(
            u64,
            Vec<u64>,
            String,
            bool,
            Option<f64>,
            Option<String>,
            BTreeMap<String, String>,
        )],
        progress_bar: &ProgressBar,
    ) -> Result<Vec<WaySegment>> {
        // Collect all node IDs needed for this batch
        let mut needed_nodes = HashSet::new();
        for (_, node_refs, _, _, _, _, _) in ways_batch {
            needed_nodes.extend(node_refs);
        }

        progress_bar.set_message(format!(
            "Loading {} unique nodes for way batch",
            needed_nodes.len()
        ));

        // Load only the needed nodes from disk
        let node_locations = self.load_required_nodes(&needed_nodes)?;

        // Create road segments
        let mut road_segments = Vec::new();

        for (way_id, node_refs, highway_type, is_oneway, max_speed, name, metadata) in ways_batch {
            let coordinates: Vec<Coord<f64>> = node_refs
                .iter()
                .filter_map(|&node_id| node_locations.get(&node_id).cloned())
                .collect();

            // Skip if we couldn't find enough coordinates
            if coordinates.len() < 2 {
                continue;
            }

            let segment = WaySegment {
                id: *way_id,
                nodes: node_refs.clone(),
                coordinates,
                is_oneway: *is_oneway,
                highway_type: highway_type.clone(),
                max_speed: *max_speed,
                connections: Vec::new(),
                name: name.clone(),
                metadata: Some(metadata.clone()),
                original_id: None,
                split_id: None,
            };

            road_segments.push(segment);
        }

        Ok(road_segments)
    }

    /// Load only required nodes for the current batch from disk
    fn load_required_nodes(&self, needed_nodes: &HashSet<u64>) -> Result<HashMap<u64, Coord<f64>>> {
        let mut node_locations = HashMap::new();

        // Get list of all node batch files
        let paths = fs::read_dir(&self.temp_dir)?
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .path()
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.starts_with("node_batch_") && name.ends_with(".bin"))
                    .unwrap_or(false)
            })
            .map(|entry| entry.path())
            .collect::<Vec<_>>();

        // Load nodes from each batch file
        for path in paths {
            let tile_bytes = std::fs::read(&path)
                .map_err(|e| anyhow!("Failed to read path {:?}: {}", path, e))?;
            let config = bincode::config::standard();
            let (batch_nodes, _): (HashMap<u64, Coord<f64>>, _) =
                bincode::serde::decode_from_slice(&tile_bytes, config)?;

            // Add only needed nodes from this batch
            for (&node_id, &coord) in batch_nodes.iter() {
                if needed_nodes.contains(&node_id) {
                    node_locations.insert(node_id, coord);
                }
            }

            // If we've found all needed nodes, we can stop
            if node_locations.len() >= needed_nodes.len() {
                break;
            }
        }

        Ok(node_locations)
    }

    /// Process segment intersections in a memory-efficient way
    fn process_segment_intersections(
        &self,
        mut road_segments: Vec<WaySegment>,
    ) -> Result<Vec<WaySegment>> {
        let start_time = Instant::now();
        info!("Detecting and processing segment intersections");

        // Create a spatial index for faster intersection checking
        // We'll use a simple grid-based approach with a reasonable cell size
        let cell_size = 0.001; // ~111 meters at the equator
        let mut grid_index: HashMap<(i32, i32), Vec<usize>> = HashMap::new();

        // Extract layer information from each segment
        let segment_layers: Vec<_> = road_segments
            .iter()
            .map(|segment| {
                let layer = segment
                    .metadata
                    .as_ref()
                    .and_then(|metadata| {
                        let layer = metadata
                            .get("layer")
                            .map_or(0, |v| v.parse::<i8>().unwrap_or(0));
                        let is_bridge = metadata.get("bridge").map_or(false, |v| v == "yes");
                        let is_tunnel = metadata.get("tunnel").map_or(false, |v| v == "yes");

                        Some((layer, is_bridge, is_tunnel))
                    })
                    .unwrap_or((0, false, false));

                layer
            })
            .collect();

        // Progress bar for intersection processing
        let total_segments = road_segments.len();
        let progress_bar = ProgressBar::new(total_segments as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {percent}% {msg}",
                )?
                .progress_chars("##-"),
        );

        // Index segments by grid cells they intersect
        for (idx, segment) in road_segments.iter().enumerate() {
            let bbox = segment.bounding_box();

            // Calculate grid cells this segment might intersect
            let min_x_cell = (bbox.min().x / cell_size).floor() as i32;
            let max_x_cell = (bbox.max().x / cell_size).ceil() as i32;
            let min_y_cell = (bbox.min().y / cell_size).floor() as i32;
            let max_y_cell = (bbox.max().y / cell_size).ceil() as i32;

            // Add segment to all relevant cells
            for x in min_x_cell..=max_x_cell {
                for y in min_y_cell..=max_y_cell {
                    grid_index.entry((x, y)).or_default().push(idx);
                }
            }
        }

        // Track new segments to be added after processing
        let new_segments = Arc::new(Mutex::new(Vec::new()));
        // Track segments to be removed
        let segments_to_remove = Arc::new(Mutex::new(HashSet::new()));
        // Track already processed pairs to avoid duplicates
        let processed_pairs = Arc::new(Mutex::new(HashSet::new()));
        // Track generated intersection nodes for connectivity
        let intersection_nodes = Arc::new(Mutex::new(HashMap::new()));
        // Track next available node id (for creating new intersection nodes)
        let next_node_id = Arc::new(Mutex::new(u64::MAX / 2)); // Start from the middle of u64 range to avoid conflicts

        // Counters for logging
        let intersection_count = Arc::new(Mutex::new(0));
        let skipped_layer_diff = Arc::new(Mutex::new(0));

        // Process segments in parallel chunks to improve performance
        // We need to be careful with parallel processing when modifying shared data
        if self.parallel_processing {
            // Split work into chunks for parallel processing
            let chunk_size = (total_segments / rayon::current_num_threads().max(1)).max(1);
            let chunks: Vec<_> = (0..total_segments)
                .collect::<Vec<_>>()
                .chunks(chunk_size)
                .map(|c| c.to_vec())
                .collect();

            chunks.par_iter().for_each(|chunk| {
                for &segment_idx in chunk {
                    self.process_segment_intersections_for_segment(
                        segment_idx,
                        &road_segments,
                        &segment_layers,
                        &grid_index,
                        &cell_size,
                        &new_segments,
                        &segments_to_remove,
                        &processed_pairs,
                        &intersection_nodes,
                        &next_node_id,
                        &intersection_count,
                        &skipped_layer_diff,
                    );

                    progress_bar.inc(1);
                }
            });
        } else {
            // Sequential processing for debugging or if parallel processing causes issues
            for segment_idx in 0..total_segments {
                self.process_segment_intersections_for_segment(
                    segment_idx,
                    &road_segments,
                    &segment_layers,
                    &grid_index,
                    &cell_size,
                    &new_segments,
                    &segments_to_remove,
                    &processed_pairs,
                    &intersection_nodes,
                    &next_node_id,
                    &intersection_count,
                    &skipped_layer_diff,
                );

                progress_bar.inc(1);
            }
        }

        progress_bar.finish_with_message("Intersection processing complete");

        // Remove segments marked for removal
        let segments_to_remove = segments_to_remove.lock().unwrap();
        let mut ordered_indices: Vec<usize> = segments_to_remove.iter().cloned().collect();
        ordered_indices.sort_unstable_by(|a, b| b.cmp(a)); // Sort in descending order

        for idx in ordered_indices {
            // Only remove if index is valid
            if idx < road_segments.len() {
                road_segments.swap_remove(idx);
            }
        }

        // Add all new segments
        let new_segments_to_add = new_segments.lock().unwrap();
        road_segments.extend(new_segments_to_add.iter().cloned());

        let intersection_count = *intersection_count.lock().unwrap();
        let skipped_layer_diff = *skipped_layer_diff.lock().unwrap();

        info!(
            "Processed {} segment intersections, skipped {} due to layer differences. Total segments: {} (in {:?})",
            intersection_count,
            skipped_layer_diff,
            road_segments.len(),
            start_time.elapsed()
        );

        Ok(road_segments)
    }

    /// Process intersections for a single segment - can be called in parallel
    fn process_segment_intersections_for_segment(
        &self,
        segment_idx: usize,
        road_segments: &[WaySegment],
        segment_layers: &[(i8, bool, bool)],
        grid_index: &HashMap<(i32, i32), Vec<usize>>,
        cell_size: &f64,
        new_segments: &Arc<Mutex<Vec<WaySegment>>>,
        segments_to_remove: &Arc<Mutex<HashSet<usize>>>,
        processed_pairs: &Arc<Mutex<HashSet<(usize, usize)>>>,
        intersection_nodes: &Arc<Mutex<HashMap<(OrderedFloat<f64>, OrderedFloat<f64>), u64>>>,
        next_node_id: &Arc<Mutex<u64>>,
        intersection_count: &Arc<Mutex<usize>>,
        skipped_layer_diff: &Arc<Mutex<usize>>,
    ) {
        // Skip if already marked for removal
        if segments_to_remove.lock().unwrap().contains(&segment_idx) {
            return;
        }

        let segment = &road_segments[segment_idx];
        let bbox = segment.bounding_box();

        // Calculate grid cells to check
        let min_x_cell = (bbox.min().x / cell_size).floor() as i32;
        let max_x_cell = (bbox.max().x / cell_size).ceil() as i32;
        let min_y_cell = (bbox.min().y / cell_size).floor() as i32;
        let max_y_cell = (bbox.max().y / cell_size).ceil() as i32;

        // Get segment's layer info
        let (segment_layer, segment_is_bridge, segment_is_tunnel) = segment_layers[segment_idx];

        // Collect potentially intersecting segments
        let mut potential_intersections = HashSet::new();
        for x in min_x_cell..=max_x_cell {
            for y in min_y_cell..=max_y_cell {
                if let Some(cell_segments) = grid_index.get(&(x, y)) {
                    potential_intersections.extend(cell_segments);
                }
            }
        }

        // Process each segment line
        for i in 0..segment.coordinates.len() - 1 {
            let seg_line = geo::Line::new(segment.coordinates[i], segment.coordinates[i + 1]);

            // Check against each potentially intersecting segment
            for &other_idx in &potential_intersections {
                // Skip self and already processed pairs
                {
                    let processed_pairs_lock = processed_pairs.lock().unwrap();
                    if other_idx <= segment_idx
                        || processed_pairs_lock.contains(&(segment_idx, other_idx))
                        || processed_pairs_lock.contains(&(other_idx, segment_idx))
                    {
                        continue;
                    }
                }

                // Skip segments already marked for removal
                if segments_to_remove.lock().unwrap().contains(&other_idx) {
                    continue;
                }

                let other_segment = &road_segments[other_idx];

                // Skip if segments already share nodes (they're already connected)
                if self.segments_share_nodes(segment, other_segment) {
                    continue;
                }

                // Get other segment's layer info
                let (other_layer, other_is_bridge, other_is_tunnel) = segment_layers[other_idx];

                // Skip if segments are on different layers (bridges, tunnels, etc.)
                if segment_layer != other_layer
                    || segment_is_bridge != other_is_bridge
                    || segment_is_tunnel != other_is_tunnel
                {
                    *skipped_layer_diff.lock().unwrap() += 1;
                    continue;
                }

                // Check each line segment of the other road
                for j in 0..other_segment.coordinates.len() - 1 {
                    let other_line = geo::Line::new(
                        other_segment.coordinates[j],
                        other_segment.coordinates[j + 1],
                    );

                    // Check for intersection
                    if let Some(intersection) = line_intersection(seg_line, other_line) {
                        match intersection {
                            LineIntersection::SinglePoint {
                                intersection: point,
                                ..
                            } => {
                                // Skip if intersection is at an endpoint of either segment
                                if self.is_point_at_segment_endpoint(segment, point)
                                    || self.is_point_at_segment_endpoint(other_segment, point)
                                {
                                    continue;
                                }

                                // Round to avoid floating point precision issues
                                let rounded_x = (point.x * 1e7).round() / 1e7;
                                let rounded_y = (point.y * 1e7).round() / 1e7;

                                // Get or create a node ID for this intersection
                                let intersection_node_id = {
                                    let mut intersection_nodes_lock =
                                        intersection_nodes.lock().unwrap();
                                    *intersection_nodes_lock
                                        .entry((OrderedFloat(rounded_x), OrderedFloat(rounded_y)))
                                        .or_insert_with(|| {
                                            let mut id_lock = next_node_id.lock().unwrap();
                                            *id_lock -= 1;
                                            *id_lock
                                        })
                                };

                                // Split first segment
                                let (seg1a, seg1b) = self.split_segment(
                                    segment,
                                    intersection_node_id,
                                    Coord {
                                        x: point.x,
                                        y: point.y,
                                    },
                                    i,
                                );

                                // Split second segment
                                let (seg2a, seg2b) = self.split_segment(
                                    other_segment,
                                    intersection_node_id,
                                    Coord {
                                        x: point.x,
                                        y: point.y,
                                    },
                                    j,
                                );

                                // Add new segments to our list
                                {
                                    let mut new_segments_lock = new_segments.lock().unwrap();
                                    new_segments_lock.push(seg1a);
                                    new_segments_lock.push(seg1b);
                                    new_segments_lock.push(seg2a);
                                    new_segments_lock.push(seg2b);
                                }

                                // Mark original segments for removal
                                {
                                    let mut segments_to_remove_lock =
                                        segments_to_remove.lock().unwrap();
                                    segments_to_remove_lock.insert(segment_idx);
                                    segments_to_remove_lock.insert(other_idx);
                                }

                                // Mark this pair as processed
                                {
                                    let mut processed_pairs_lock = processed_pairs.lock().unwrap();
                                    processed_pairs_lock.insert((segment_idx, other_idx));
                                }

                                *intersection_count.lock().unwrap() += 1;

                                // Only process one intersection per segment pair
                                break;
                            }
                            _ => continue, // Skip other intersection types
                        }
                    }
                }
            }
        }
    }

    /// Helper function to check if segments share any nodes
    fn segments_share_nodes(&self, segment1: &WaySegment, segment2: &WaySegment) -> bool {
        for node1 in &segment1.nodes {
            if segment2.nodes.contains(node1) {
                return true;
            }
        }
        false
    }

    /// Helper function to check if a point is at a segment endpoint
    fn is_point_at_segment_endpoint(&self, segment: &WaySegment, point: geo::Coord<f64>) -> bool {
        // Check first and last coordinates
        let first = &segment.coordinates[0];
        let last = &segment.coordinates[segment.coordinates.len() - 1];

        let point_coord = Coord {
            x: point.x,
            y: point.y,
        };

        self.coord_equals(first, &point_coord) || self.coord_equals(last, &point_coord)
    }

    /// Helper function for coordinate comparison with epsilon
    fn coord_equals(&self, a: &Coord<f64>, b: &Coord<f64>) -> bool {
        (a.x - b.x).abs() < std::f64::EPSILON && (a.y - b.y).abs() < std::f64::EPSILON
    }

    /// Helper function to split a segment at an intersection point
    fn split_segment(
        &self,
        segment: &WaySegment,
        intersection_node_id: u64,
        intersection_coord: Coord<f64>,
        split_index: usize,
    ) -> (WaySegment, WaySegment) {
        // Create first segment with coordinates up to the split point (inclusive)
        let mut first_segment_coords = segment.coordinates[0..=split_index].to_vec();
        first_segment_coords.push(intersection_coord);

        // Create second segment with coordinates from the split point (inclusive) to the end
        let mut second_segment_coords = vec![intersection_coord];
        second_segment_coords.extend_from_slice(&segment.coordinates[split_index + 1..]);

        // Create first segment nodes
        let mut first_segment_nodes = segment.nodes[0..=split_index].to_vec();
        first_segment_nodes.push(intersection_node_id);

        // Create second segment nodes
        let mut second_segment_nodes = vec![intersection_node_id];
        second_segment_nodes.extend_from_slice(&segment.nodes[split_index + 1..]);

        // Generate a deterministic ID for the second segment
        // Using a combination of the original ID and a hash of the coordinates
        let second_id_part = (intersection_coord.x.to_bits() & 0xFFFF0000)
            | (intersection_coord.y.to_bits() & 0xFFFF);
        let second_id = segment.id ^ second_id_part;

        // Create first segment (preserve original ID and metadata)
        let first_segment = WaySegment {
            id: segment.id,
            nodes: first_segment_nodes,
            coordinates: first_segment_coords,
            is_oneway: segment.is_oneway,
            highway_type: segment.highway_type.clone(),
            max_speed: segment.max_speed,
            connections: vec![],
            name: segment.name.clone(),
            metadata: segment.metadata.clone(),
            original_id: segment.original_id.or(Some(segment.id)),
            split_id: Some(intersection_node_id),
        };

        // Create second segment (with deterministic new ID and preserving metadata)
        let second_segment = WaySegment {
            id: second_id,
            nodes: second_segment_nodes,
            coordinates: second_segment_coords,
            is_oneway: segment.is_oneway,
            highway_type: segment.highway_type.clone(),
            max_speed: segment.max_speed,
            connections: vec![],
            name: segment.name.clone(),
            metadata: segment.metadata.clone(),
            original_id: segment.original_id.or(Some(segment.id)),
            split_id: Some(intersection_node_id),
        };

        (first_segment, second_segment)
    }

    /// Build connectivity for road segments in a memory-efficient way
    fn build_connectivity(&self, mut road_segments: Vec<WaySegment>) -> Result<Vec<WaySegment>> {
        let start_time = Instant::now();
        info!("Building road network connectivity");

        // Create segment lookup (we need this in memory for efficient connectivity building)
        let mut segment_map: HashMap<u64, usize> = HashMap::new();
        for (i, segment) in road_segments.iter().enumerate() {
            segment_map.insert(segment.id, i);
        }

        // First, build a node-to-segment index
        let mut node_connections: HashMap<u64, Vec<u64>> = HashMap::new();
        for segment in &road_segments {
            for &node_id in &segment.nodes {
                node_connections
                    .entry(node_id)
                    .or_default()
                    .push(segment.id);
            }
        }

        // Track split segments by their connecting nodes
        let mut split_node_segments: HashMap<u64, Vec<u64>> = HashMap::new();
        for segment in &road_segments {
            if let Some(split_id) = segment.split_id {
                split_node_segments
                    .entry(split_id)
                    .or_default()
                    .push(segment.id);
            }
        }

        let total_segments = road_segments.len();
        let progress_bar = ProgressBar::new(total_segments as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[{elapsed_precise}] {bar:40.yellow/blue} {pos:>7}/{len:7} {percent}% {msg}",
                )?
                .progress_chars("##-"),
        );

        // Use a vector of segment_id -> connections to build in parallel
        let mut segment_connections: Vec<(u64, HashSet<u64>)> = Vec::with_capacity(total_segments);
        for _ in 0..total_segments {
            segment_connections.push((0, HashSet::new()));
        }

        if self.parallel_processing {
            let segments_arc = Arc::new(road_segments.clone());
            let segment_map_arc = Arc::new(segment_map.clone());
            let node_connections_arc = Arc::new(node_connections.clone());
            let split_node_segments_arc = Arc::new(split_node_segments.clone());

            segment_connections.par_iter_mut().enumerate().for_each(
                |(idx, (segment_id, connections))| {
                    let segment = &segments_arc[idx];
                    *segment_id = segment.id;

                    // Special handling for split segments - ensure they're connected at the split point
                    if let Some(split_id) = segment.split_id {
                        if let Some(related_segments) = split_node_segments_arc.get(&split_id) {
                            for &other_id in related_segments {
                                if other_id != segment.id {
                                    // Connect the segments if they share the split node
                                    if let Some(&other_idx) = segment_map_arc.get(&other_id) {
                                        let other_segment = &segments_arc[other_idx];

                                        // Find positions of the split node in each segment
                                        if let (Some(this_pos), Some(other_pos)) = (
                                            segment.nodes.iter().position(|&n| n == split_id),
                                            other_segment.nodes.iter().position(|&n| n == split_id),
                                        ) {
                                            // Handle oneway restrictions at split points
                                            let this_is_entry = this_pos == 0;
                                            let this_is_exit = this_pos == segment.nodes.len() - 1;
                                            let other_is_entry = other_pos == 0;
                                            let other_is_exit =
                                                other_pos == other_segment.nodes.len() - 1;

                                            // Connect if directionality allows
                                            let can_go_this_to_other = (!segment.is_oneway
                                                || this_is_exit)
                                                && (!other_segment.is_oneway || other_is_entry);

                                            if can_go_this_to_other {
                                                connections.insert(other_id);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Regular connectivity through shared nodes (prioritizing endpoints)
                    let endpoint_nodes = [
                        *segment.nodes.first().unwrap(),
                        *segment.nodes.last().unwrap(),
                    ];

                    for &node in &endpoint_nodes {
                        if let Some(connected_segments) = node_connections_arc.get(&node) {
                            for &other_id in connected_segments {
                                if other_id != segment.id {
                                    if let Some(&other_idx) = segment_map_arc.get(&other_id) {
                                        let other_segment = &segments_arc[other_idx];

                                        // Check if segments have compatible layers
                                        let this_layer = segment
                                            .metadata
                                            .as_ref()
                                            .and_then(|m| {
                                                m.get("layer").map(|l| l.parse::<i8>().unwrap_or(0))
                                            })
                                            .unwrap_or(0);

                                        let other_layer = other_segment
                                            .metadata
                                            .as_ref()
                                            .and_then(|m| {
                                                m.get("layer").map(|l| l.parse::<i8>().unwrap_or(0))
                                            })
                                            .unwrap_or(0);

                                        // Skip connections between different layers
                                        if this_layer != other_layer {
                                            continue;
                                        }

                                        // Find which node in the other segment matches this node
                                        if let Some(other_node_idx) =
                                            other_segment.nodes.iter().position(|&n| n == node)
                                        {
                                            let this_is_start =
                                                node == *segment.nodes.first().unwrap();
                                            let this_is_end =
                                                node == *segment.nodes.last().unwrap();
                                            let other_is_start = other_node_idx == 0;
                                            let other_is_end =
                                                other_node_idx == other_segment.nodes.len() - 1;

                                            // Handle oneway restrictions
                                            let can_connect_from_this_to_other = (this_is_end
                                                || !segment.is_oneway)
                                                && (other_is_start || !other_segment.is_oneway);

                                            if can_connect_from_this_to_other {
                                                connections.insert(other_id);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    progress_bar.inc(1);
                },
            );
        } else {
            // Sequential processing
            for (idx, segment) in road_segments.iter().enumerate() {
                let (segment_id, connections) = &mut segment_connections[idx];
                *segment_id = segment.id;

                // Process the same connectivity logic as above, but sequentially
                // ... (logic similar to parallel version)
                unimplemented!("parallel only");

                progress_bar.inc(1);
            }
        }

        // Apply connections to the road segments
        let mut connection_count = 0;
        for (idx, segment) in road_segments.iter_mut().enumerate() {
            let (_, connections) = &segment_connections[idx];
            segment.connections = connections.iter().cloned().collect();
            segment.connections.sort_unstable();
            connection_count += segment.connections.len();
        }

        progress_bar.finish_with_message(format!("Built {} connections", connection_count));

        info!(
            "Built {} connections among {} segments in {:?}",
            connection_count,
            road_segments.len(),
            start_time.elapsed()
        );

        Ok(road_segments)
    }

    /// Generate tiles in a memory-efficient way, processing in chunks
    fn generate_tiles(&self, road_segments: Vec<WaySegment>, output_dir: &str) -> Result<()> {
        let start_time = Instant::now();
        info!("Generating adaptive tiles");

        // Create output directory
        std::fs::create_dir_all(output_dir)?;

        // Use a tile-based approach where we assign segments to tiles,
        // then process and write each tile independently
        let mut tiles: HashMap<String, Vec<u64>> = HashMap::new();
        let mut segment_tile_map: HashMap<u64, String> = HashMap::new();

        // Progress bar for tile generation
        let progress_bar = ProgressBar::new(road_segments.len() as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[{elapsed_precise}] {bar:40.green/blue} {pos:>7}/{len:7} {percent}% {msg}",
                )?
                .progress_chars("##-"),
        );

        // First pass: assign segments to tiles
        for segment in &road_segments {
            let centroid = segment.centroid();
            let mut current_size = self.config.base_tile_size;
            let mut current_depth = 0;

            // Calculate initial tile ID
            let mut tile_id = self.calculate_tile_id(&centroid, current_size, current_depth);

            // Adaptive tile splitting
            loop {
                let segments = tiles.entry(tile_id.clone()).or_default();

                // Check density or max depth
                if segments.len() < self.config.min_tile_density
                    || current_depth >= self.config.max_split_depth
                {
                    // Store segment in this tile
                    segment_tile_map.insert(segment.id, tile_id.clone());
                    segments.push(segment.id);
                    break;
                }

                // Split tile and try again
                current_size /= 2.0;
                current_depth += 1;
                tile_id = self.calculate_tile_id(&centroid, current_size, current_depth);
            }

            progress_bar.inc(1);
        }

        progress_bar.finish_with_message(format!("Assigned segments to {} tiles", tiles.len()));

        // Create an index for looking up segments by ID
        let segment_index: HashMap<u64, usize> = road_segments
            .iter()
            .enumerate()
            .map(|(idx, segment)| (segment.id, idx))
            .collect();

        // Second pass: process and write each tile independently
        let total_tiles = tiles.len();
        let progress_bar = ProgressBar::new(total_tiles as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[{elapsed_precise}] {bar:40.magenta/blue} {pos:>7}/{len:7} {percent}% {msg}",
                )?
                .progress_chars("##-"),
        );

        if self.parallel_processing {
            // Process tiles in parallel for better performance
            let segment_index_arc = Arc::new(segment_index);
            let road_segments_arc = Arc::new(road_segments);

            tiles.par_iter().for_each(|(tile_id, segment_ids)| {
                let tile_path = Path::new(output_dir).join(format!("{}.bin", tile_id));

                // Create tile road segments
                let mut tile_road_segments = Vec::with_capacity(segment_ids.len());
                for &seg_id in segment_ids {
                    if let Some(&idx) = segment_index_arc.get(&seg_id) {
                        tile_road_segments.push(road_segments_arc[idx].clone());
                    }
                }

                // Create segment index for this tile
                let mut segment_index_map = HashMap::new();
                for segment in &tile_road_segments {
                    segment_index_map.insert(segment.id, tile_id.clone());
                }

                // Create tile index
                let tile_index = TileIndex {
                    tile_id: tile_id.clone(),
                    bbox: self.calculate_bbox(tile_id),
                    road_segments: tile_road_segments,
                    segment_index: segment_index_map,
                };

                // Serialize and write tile
                let config = bincode::config::standard();
                match bincode::serde::encode_to_vec(&tile_index, config) {
                    Ok(serialized_tile) => {
                        if let Err(e) = std::fs::write(&tile_path, serialized_tile) {
                            warn!("Failed to write tile {}: {}", tile_id, e);
                        }
                    }
                    Err(e) => {
                        warn!("Failed to serialize tile {}: {}", tile_id, e);
                    }
                }

                progress_bar.inc(1);
            });
        } else {
            // Sequential processing
            for (tile_id, segment_ids) in &tiles {
                // Process tile (similar logic to parallel version)
                // ...
                unimplemented!("parallel only");

                progress_bar.inc(1);
            }
        }

        progress_bar.finish_with_message("Tile generation complete");

        info!(
            "Generated {} tiles with adaptive sizing in {:?}",
            tiles.len(),
            start_time.elapsed()
        );

        Ok(())
    }

    // Helper function to generate tile ID string
    fn calculate_tile_id(&self, centroid: &Point<f64>, tile_size: f64, depth: u8) -> String {
        let x_tile = (centroid.x() / tile_size).floor() as i32;
        let y_tile = (centroid.y() / tile_size).floor() as i32;
        format!("{}_{}_{}", x_tile, y_tile, depth)
    }

    // Calculate bounding box for a tile
    fn calculate_bbox(&self, tile_id: &str) -> geo::Rect<f64> {
        let parts: Vec<&str> = tile_id.split('_').collect();
        let x_tile: i32 = parts[0].parse().unwrap();
        let y_tile: i32 = parts[1].parse().unwrap();
        let depth: u8 = parts[2].parse().unwrap();

        let tile_size = self.config.base_tile_size / 2f64.powi(depth as i32);

        geo::Rect::new(
            geo::Coord {
                x: x_tile as f64 * tile_size,
                y: y_tile as f64 * tile_size,
            },
            geo::Coord {
                x: (x_tile as f64 + 1.0) * tile_size,
                y: (y_tile as f64 + 1.0) * tile_size,
            },
        )
    }

    /// Clean up temporary files after processing
    fn cleanup_temp_files(&self) -> Result<()> {
        info!("Cleaning up temporary files");

        let temp_dir = Path::new(&self.temp_dir);
        if temp_dir.exists() {
            for entry in fs::read_dir(temp_dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.is_file()
                    && path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .map(|name| name.starts_with("node_batch_") && name.ends_with(".bin"))
                        .unwrap_or(false)
                {
                    fs::remove_file(path)?;
                }
            }
        }

        Ok(())
    }

    /// Helper method to count elements in an OSM PBF file
    fn count_elements(
        &self,
        pbf_path: &str,
        count_nodes: bool,
        count_ways: bool,
        count_relations: bool,
    ) -> Result<u64> {
        let reader = ElementReader::from_path(pbf_path)?;
        let mut count = 0u64;

        reader.for_each(|element| match element {
            Element::Node(_) | Element::DenseNode(_) if count_nodes => count += 1,
            Element::Way(_) if count_ways => count += 1,
            Element::Relation(_) if count_relations => count += 1,
            _ => {}
        })?;

        Ok(count)
    }
}

/// OSM Network Processor
pub struct OsmProcessor {
    config: TileConfig,
}

impl OsmProcessor {
    pub fn new(config: TileConfig) -> Self {
        Self { config }
    }

    /// Process entire OSM PBF file and generate tiles
    pub fn process_pbf(&self, pbf_path: &str, output_dir: &str) -> Result<()> {
        let start_time = Instant::now();
        let mut road_segments = Vec::new();
        let mut node_locations = HashMap::new();
        let mut node_connections = HashMap::new(); // For building topology

        // First pass: collect node locations
        self.collect_node_locations(pbf_path, &mut node_locations)?;

        // Second pass: process ways and create road segments
        self.process_ways(
            pbf_path,
            &node_locations,
            &mut road_segments,
            &mut node_connections,
        )?;

        // Build connectivity
        info!(
            "Building road connectivity graph for {} segments",
            road_segments.len()
        );
        self.build_connectivity(&mut road_segments, &node_connections)?;

        // Generate tiles
        info!("Generating adaptive tiles");
        self.generate_tiles(road_segments, output_dir)?;

        info!("OSM processing completed in {:?}", start_time.elapsed());

        Ok(())
    }

    fn collect_node_locations(
        &self,
        pbf_path: &str,
        node_locations: &mut HashMap<u64, Coord<f64>>,
    ) -> Result<()> {
        let start_time = Instant::now();
        let reader = ElementReader::from_path(pbf_path)?;

        reader.for_each(|element| match element {
            Element::Node(node) => {
                node_locations.insert(
                    node.id() as u64,
                    Coord {
                        x: node.lon(),
                        y: node.lat(),
                    },
                );
            }
            Element::DenseNode(node) => {
                node_locations.insert(
                    node.id() as u64,
                    Coord {
                        x: node.lon(),
                        y: node.lat(),
                    },
                );
            }
            _ => {}
        })?;

        debug!(
            "Collected {} node locations in {:?}",
            node_locations.len(),
            start_time.elapsed()
        );
        Ok(())
    }

    fn process_ways(
        &self,
        pbf_path: &str,
        node_locations: &HashMap<u64, Coord<f64>>,
        road_segments: &mut Vec<WaySegment>,
        node_connections: &mut HashMap<u64, Vec<u64>>,
    ) -> Result<()> {
        let start_time = Instant::now();
        let reader = ElementReader::from_path(pbf_path)?;

        // Track highway types to filter for important roads
        let important_highway_types: HashSet<&str> = [
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "motorway_link",
            "trunk_link",
            "primary_link",
            "secondary_link",
            "tertiary_link",
            "residential",
            "unclassified",
            "service",
        ]
        .iter()
        .cloned()
        .collect();

        reader.for_each(|element| {
            if let osmpbf::Element::Way(way) = element {
                // Filter for highway ways
                let highway_type = way
                    .tags()
                    .find(|(key, _)| *key == "highway")
                    .map(|(_, value)| value.to_string());

                if let Some(highway) = highway_type {
                    // Skip insignificant highways
                    if !important_highway_types.contains(highway.as_str()) {
                        return;
                    }

                    let is_oneway = way.tags().any(|(key, value)| {
                        (key == "oneway" && (value == "yes" || value == "1"))
                            || (key == "highway" && value == "motorway")
                    });

                    // Extract max speed if available
                    let max_speed =
                        way.tags()
                            .find(|(key, _)| *key == "maxspeed")
                            .and_then(|(_, value)| {
                                // Parse speed value (handle units)
                                let v = value.trim();
                                if v.ends_with(" mph") {
                                    v.trim_end_matches(" mph")
                                        .parse::<f64>()
                                        .ok()
                                        .map(|mph| mph * 1.60934)
                                } else if v.ends_with(" km/h") {
                                    v.trim_end_matches(" km/h").parse::<f64>().ok()
                                } else {
                                    v.parse::<f64>().ok()
                                }
                            });

                    // Extract road name if available
                    let name = way
                        .tags()
                        .find(|(key, _)| *key == "name")
                        .map(|(_, value)| value.to_string());

                    let node_refs = way.refs().map(|id| id as u64).collect::<Vec<_>>();
                    let coordinates: Vec<Coord<f64>> = node_refs
                        .iter()
                        .filter_map(|&node_id| node_locations.get(&node_id).cloned())
                        .collect();

                    if coordinates.len() > 1 {
                        let segment = WaySegment {
                            id: way.id() as u64,
                            nodes: node_refs.clone(),
                            coordinates,
                            is_oneway,
                            highway_type: highway,
                            max_speed,
                            connections: Vec::new(),
                            name,
                            ..Default::default()
                        };

                        road_segments.push(segment);

                        // Build node-to-way index for connectivity
                        for &node_id in &node_refs {
                            node_connections
                                .entry(node_id)
                                .or_default()
                                .push(way.id() as u64);
                        }
                    }
                }
            }
        })?;

        debug!(
            "Processed {} road segments in {:?}",
            road_segments.len(),
            start_time.elapsed()
        );
        Ok(())
    }

    fn build_connectivity(
        &self,
        road_segments: &mut [WaySegment],
        node_connections: &HashMap<u64, Vec<u64>>,
    ) -> Result<()> {
        let start_time = Instant::now();

        // Create segment lookup
        let mut segment_map: HashMap<u64, usize> = HashMap::new();
        for (i, segment) in road_segments.iter().enumerate() {
            segment_map.insert(segment.id, i);
        }

        // Function to find shared nodes between segments (prefixed with _ to avoid warning)
        let _find_shared_nodes = |segment1: &WaySegment, segment2: &WaySegment| -> Vec<u64> {
            let nodes1: HashSet<u64> = segment1.nodes.iter().cloned().collect();
            segment2
                .nodes
                .iter()
                .filter(|n| nodes1.contains(n))
                .cloned()
                .collect()
        };

        // Build connections based on shared nodes
        let mut connection_count = 0;

        // Create a map to store segment connections that we'll apply later
        let mut segment_connections: HashMap<u64, Vec<u64>> = HashMap::new();

        // First pass: collect connection information
        for segment in road_segments.iter() {
            // Get endpoints for faster connectivity check
            if segment.nodes.len() < 2 {
                continue;
            }

            let segment_id = segment.id;
            let start_node = *segment.nodes.first().unwrap();
            let end_node = *segment.nodes.last().unwrap();

            // For oneway roads, only consider connections at the end node
            let connection_nodes = if segment.is_oneway {
                vec![end_node]
            } else {
                vec![start_node, end_node]
            };

            // For each connection node, find other segments that share this node
            for &node in &connection_nodes {
                if let Some(connected_segments) = node_connections.get(&node) {
                    for &other_id in connected_segments {
                        // Skip self-connections
                        if other_id == segment_id {
                            continue;
                        }

                        // Get other segment index
                        if let Some(&other_idx) = segment_map.get(&other_id) {
                            let other_segment = &road_segments[other_idx];

                            // Check for oneway restriction on the other segment
                            if other_segment.is_oneway {
                                // Only connect if we're at the start node of the other segment
                                if other_segment.nodes.first() == Some(&node) {
                                    segment_connections
                                        .entry(segment_id)
                                        .or_default()
                                        .push(other_id);
                                    connection_count += 1;
                                }
                            } else {
                                // Non-oneway segments can connect at any point
                                segment_connections
                                    .entry(segment_id)
                                    .or_default()
                                    .push(other_id);
                                connection_count += 1;
                            }
                        }
                    }
                }
            }
        }

        // Second pass: apply connections and deduplicate
        for segment in road_segments.iter_mut() {
            if let Some(connections) = segment_connections.get(&segment.id) {
                segment.connections = connections.clone();
                // Remove duplicates
                segment.connections.sort_unstable();
                segment.connections.dedup();
            }
        }

        debug!(
            "Built {} connections in {:?}",
            connection_count,
            start_time.elapsed()
        );
        Ok(())
    }

    fn generate_tiles(&self, road_segments: Vec<WaySegment>, output_dir: &str) -> Result<()> {
        let start_time = Instant::now();
        let mut tiles: HashMap<String, Vec<WaySegment>> = HashMap::new();
        let mut segment_tile_map: HashMap<u64, String> = HashMap::new();

        for segment in road_segments {
            let centroid = segment.centroid();
            let mut current_size = self.config.base_tile_size;
            let mut current_depth = 0;

            // Calculate initial tile ID
            let mut tile_id = self.calculate_tile_id(&centroid, current_size, current_depth);

            // Adaptive tile splitting
            loop {
                let segments = tiles.entry(tile_id.clone()).or_default();

                // Check density or max depth
                if segments.len() < self.config.min_tile_density
                    || current_depth >= self.config.max_split_depth
                {
                    // Store segment in this tile
                    segment_tile_map.insert(segment.id, tile_id.clone());
                    segments.push(segment.clone());
                    break;
                }

                // Split tile and try again
                current_size /= 2.0;
                current_depth += 1;
                tile_id = self.calculate_tile_id(&centroid, current_size, current_depth);
            }
        }

        // Convert to final TileIndex format and write to disk
        std::fs::create_dir_all(output_dir)?;
        for (tile_id, segments) in &tiles {
            let bbox = self.calculate_bbox(tile_id);

            // Create segment index for this tile
            let mut segment_index = HashMap::new();
            for segment in segments {
                segment_index.insert(segment.id, tile_id.clone());
            }

            let tile_index = TileIndex {
                tile_id: tile_id.clone(),
                bbox,
                road_segments: segments.clone(),
                segment_index,
            };

            // Serialize and write tile
            let tile_path = Path::new(output_dir).join(format!("{}.bin", tile_id));
            let config = bincode::config::standard();
            let serialized_tile = bincode::serde::encode_to_vec(&tile_index, config)?;
            std::fs::write(tile_path, serialized_tile)?;
        }

        info!(
            "Generated {} tiles with adaptive sizing in {:?}",
            tiles.len(),
            start_time.elapsed()
        );
        Ok(())
    }

    // Helper function to generate tile ID string
    fn calculate_tile_id(&self, centroid: &Point<f64>, tile_size: f64, depth: u8) -> String {
        let x_tile = (centroid.x() / tile_size).floor() as i32;
        let y_tile = (centroid.y() / tile_size).floor() as i32;
        format!("{}_{}_{}", x_tile, y_tile, depth)
    }

    // Updated bounding box calculation
    fn calculate_bbox(&self, tile_id: &str) -> geo::Rect<f64> {
        let parts: Vec<&str> = tile_id.split('_').collect();
        let x_tile: i32 = parts[0].parse().unwrap();
        let y_tile: i32 = parts[1].parse().unwrap();
        let depth: u8 = parts[2].parse().unwrap();

        let tile_size = self.config.base_tile_size / 2f64.powi(depth as i32);

        geo::Rect::new(
            geo::Coord {
                x: x_tile as f64 * tile_size,
                y: y_tile as f64 * tile_size,
            },
            geo::Coord {
                x: (x_tile as f64 + 1.0) * tile_size,
                y: (y_tile as f64 + 1.0) * tile_size,
            },
        )
    }
}
