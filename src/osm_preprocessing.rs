use anyhow::{Result, anyhow};
use geo::Haversine;
use geo::{Coord, LineString, Point, algorithm::Distance};
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, info, trace, warn};
use osmpbf::{Element, ElementReader};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use std::str::FromStr;
use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::{Arc, Mutex},
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

// Chunk metadata for streaming processing
#[derive(Debug, Serialize, Deserialize)]
struct ChunkMetadata {
    chunk_id: usize,
    node_count: usize,
    way_count: usize,
    bounding_box: Option<geo::Rect<f64>>,
}

// Intermediate node storage
#[derive(Debug, Serialize, Deserialize)]
struct NodeEntry {
    id: u64,
    coord: Coord<f64>,
}

// Intermediate way storage
#[derive(Debug, Serialize, Deserialize)]
struct WayEntry {
    id: u64,
    nodes: Vec<u64>,
    is_oneway: bool,
    highway_type: String,
    max_speed: Option<f64>,
    name: Option<String>,
    metadata: Option<BTreeMap<String, String>>,
}

/// OSM Network Processor with streaming support
pub struct StreamingOsmProcessor {
    config: TileConfig,
    // Configuration for streaming processing
    chunk_size: usize,         // Number of elements per chunk
    node_batch_size: usize,    // Number of nodes to process in memory at once
    way_batch_size: usize,     // Number of ways to process in memory at once
    temp_dir: Option<PathBuf>, // Optional custom temp directory
}

impl StreamingOsmProcessor {
    pub fn new(config: TileConfig) -> Self {
        Self {
            config,
            chunk_size: 1_000_000,      // Process 1M elements per chunk
            node_batch_size: 5_000_000, // Process 5M nodes at once
            way_batch_size: 500_000,    // Process 500K ways at once
            temp_dir: None,
        }
    }

    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    pub fn with_batch_sizes(mut self, node_batch_size: usize, way_batch_size: usize) -> Self {
        self.node_batch_size = node_batch_size;
        self.way_batch_size = way_batch_size;
        self
    }

    pub fn with_temp_dir(mut self, temp_dir: PathBuf) -> Self {
        self.temp_dir = Some(temp_dir);
        self
    }

    /// Process entire OSM PBF file and generate tiles with streaming approach
    pub fn process_pbf(&self, pbf_path: &str, output_dir: &str) -> Result<()> {
        let start_time = Instant::now();

        // Create temporary directory for intermediate data
        let temp_dir = match &self.temp_dir {
            Some(dir) => {
                std::fs::create_dir_all(dir)?;
                dir.clone()
            }
            None => PathBuf::from_str("/tmp/chronotopia").unwrap(),
        };

        info!("Using temporary directory: {:?}", temp_dir);

        // Step 1: Split the PBF into chunks and extract nodes
        info!("Step 1: Splitting PBF and extracting nodes");
        let node_chunks_info = self.split_pbf_and_extract_nodes(pbf_path, &temp_dir)?;

        // Step 2: Process nodes to create node location lookup files
        info!("Step 2: Processing node locations");
        let node_location_files = self.process_node_chunks(&temp_dir, &node_chunks_info)?;

        // Step 3: Extract ways from PBF
        info!("Step 3: Extracting ways");
        let way_chunks_info = self.extract_ways(pbf_path, &temp_dir)?;

        // Step 4: Process ways to create way segments
        info!("Step 4: Processing ways into segments");
        let segment_files =
            self.process_way_chunks(&temp_dir, &way_chunks_info, &node_location_files)?;

        // Step 5: Build connectivity between segments
        info!("Step 5: Building connectivity");
        let connected_segment_files = self.build_connectivity(&temp_dir, &segment_files)?;

        // Step 6: Generate tiles
        info!("Step 6: Generating adaptive tiles");
        self.generate_tiles(&temp_dir, &connected_segment_files, output_dir)?;

        // Cleanup temporary files (optional, comment out to keep for debugging)
        if self.temp_dir.is_none() {
            info!("Cleaning up temporary files");
            std::fs::remove_dir_all(&temp_dir)?;
        }

        info!("OSM processing completed in {:?}", start_time.elapsed());
        Ok(())
    }

    /// Split the PBF file into chunks and extract nodes to temporary files
    fn split_pbf_and_extract_nodes(
        &self,
        pbf_path: &str,
        temp_dir: &Path,
    ) -> Result<Vec<ChunkMetadata>> {
        let start_time = Instant::now();
        let reader = ElementReader::from_path(pbf_path)?;

        // Create nodes directory
        let nodes_dir = temp_dir.join("nodes");
        std::fs::create_dir_all(&nodes_dir)?;

        // Get the estimated size for progress bar
        let file_size = std::fs::metadata(pbf_path)?.len();
        let pb = ProgressBar::new(file_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) - Extracting nodes",
                )?
                .progress_chars("##-"),
        );

        let mut chunk_id = 0;
        let mut element_count = 0;
        let mut current_chunk_nodes = Vec::new();
        let mut chunk_metadata = Vec::new();

        // Process elements in chunks
        reader.for_each(|element| {
            pb.inc(1);

            // Only process nodes in this step
            match element {
                Element::Node(node) => {
                    current_chunk_nodes.push(NodeEntry {
                        id: node.id() as u64,
                        coord: Coord {
                            x: node.lon(),
                            y: node.lat(),
                        },
                    });
                }
                Element::DenseNode(node) => {
                    current_chunk_nodes.push(NodeEntry {
                        id: node.id() as u64,
                        coord: Coord {
                            x: node.lon(),
                            y: node.lat(),
                        },
                    });
                }
                _ => {}
            }

            element_count += 1;

            // If we've reached the chunk size, write to disk
            if element_count >= self.chunk_size || current_chunk_nodes.len() >= self.node_batch_size
            {
                // Write current batch of nodes
                if !current_chunk_nodes.is_empty() {
                    let node_file = nodes_dir.join(format!("nodes_{}.bin", chunk_id));
                    let file = File::create(&node_file).unwrap();
                    let mut writer = BufWriter::new(file);

                    // Serialize nodes
                    let config = bincode::config::standard();
                    bincode::serde::encode_into_std_write(
                        &current_chunk_nodes,
                        &mut writer,
                        config,
                    )
                    .unwrap();

                    // Add metadata
                    chunk_metadata.push(ChunkMetadata {
                        chunk_id,
                        node_count: current_chunk_nodes.len(),
                        way_count: 0,
                        bounding_box: None,
                    });

                    // Reset for next chunk
                    current_chunk_nodes = Vec::new();
                    chunk_id += 1;
                    element_count = 0;
                }
            }
        })?;

        // Write final batch if any
        if !current_chunk_nodes.is_empty() {
            let node_file = nodes_dir.join(format!("nodes_{}.bin", chunk_id));
            let file = File::create(&node_file)?;
            let mut writer = BufWriter::new(file);

            // Serialize nodes
            let config = bincode::config::standard();
            bincode::serde::encode_into_std_write(&current_chunk_nodes, &mut writer, config)?;

            // Add metadata
            chunk_metadata.push(ChunkMetadata {
                chunk_id,
                node_count: current_chunk_nodes.len(),
                way_count: 0,
                bounding_box: None,
            });
        }

        pb.finish_with_message("Node extraction complete");

        debug!(
            "Split PBF into {} node chunks in {:?}",
            chunk_metadata.len(),
            start_time.elapsed()
        );

        Ok(chunk_metadata)
    }

    /// Process node chunks to create node location lookup files
    fn process_node_chunks(
        &self,
        temp_dir: &Path,
        node_chunks_info: &[ChunkMetadata],
    ) -> Result<Vec<PathBuf>> {
        let start_time = Instant::now();
        let nodes_dir = temp_dir.join("nodes");
        let node_lookups_dir = temp_dir.join("node_lookups");
        std::fs::create_dir_all(&node_lookups_dir)?;

        let pb = ProgressBar::new(node_chunks_info.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) - Processing node chunks")?
            .progress_chars("##-"));

        let mut node_location_files = Vec::new();

        // Process each node chunk
        for chunk in node_chunks_info {
            pb.inc(1);

            let node_file = nodes_dir.join(format!("nodes_{}.bin", chunk.chunk_id));
            let lookup_file = node_lookups_dir.join(format!("node_lookup_{}.bin", chunk.chunk_id));

            // Read node chunk
            let file = File::open(&node_file)?;
            let mut reader = BufReader::new(file);

            // Deserialize node entries
            let config = bincode::config::standard();
            let nodes: Vec<NodeEntry> = bincode::serde::decode_from_std_read(&mut reader, config)?;

            // Create node location map
            let mut node_locations = HashMap::new();
            for node in nodes {
                node_locations.insert(node.id, node.coord);
            }

            // Write node location map
            let file = File::create(&lookup_file)?;
            let mut writer = BufWriter::new(file);
            bincode::serde::encode_into_std_write(&node_locations, &mut writer, config)?;

            node_location_files.push(lookup_file);
        }

        pb.finish_with_message("Node lookup processing complete");

        debug!(
            "Processed {} node lookup files in {:?}",
            node_location_files.len(),
            start_time.elapsed()
        );

        Ok(node_location_files)
    }

    /// Extract ways from PBF file
    fn extract_ways(&self, pbf_path: &str, temp_dir: &Path) -> Result<Vec<ChunkMetadata>> {
        let start_time = Instant::now();
        let reader = ElementReader::from_path(pbf_path)?;

        // Create ways directory
        let ways_dir = temp_dir.join("ways");
        std::fs::create_dir_all(&ways_dir)?;

        // Get the estimated size for progress bar
        let file_size = std::fs::metadata(pbf_path)?.len();
        let pb = ProgressBar::new(file_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) - Extracting ways",
                )?
                .progress_chars("##-"),
        );

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

        let mut chunk_id = 0;
        let mut element_count = 0;
        let mut current_chunk_ways = Vec::new();
        let mut chunk_metadata = Vec::new();

        // Process elements in chunks
        reader.for_each(|element| {
            pb.inc(1);

            // Only process ways in this step
            if let Element::Way(way) = element {
                // Check if it's a highway
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

                    // Collect metadata
                    let mut metadata = BTreeMap::new();
                    for (key, value) in way.tags() {
                        metadata.insert(key.to_string(), value.to_string());
                    }

                    // Get node references
                    let node_refs = way.refs().map(|id| id as u64).collect::<Vec<_>>();

                    if node_refs.len() > 1 {
                        current_chunk_ways.push(WayEntry {
                            id: way.id() as u64,
                            nodes: node_refs,
                            is_oneway,
                            highway_type: highway,
                            max_speed,
                            name,
                            metadata: Some(metadata),
                        });
                    }
                }
            }

            element_count += 1;

            // If we've reached the chunk size, write to disk
            if element_count >= self.chunk_size || current_chunk_ways.len() >= self.way_batch_size {
                // Write current batch of ways
                if !current_chunk_ways.is_empty() {
                    let way_file = ways_dir.join(format!("ways_{}.bin", chunk_id));
                    let file = File::create(&way_file).unwrap();
                    let mut writer = BufWriter::new(file);

                    // Serialize ways
                    let config = bincode::config::standard();
                    bincode::serde::encode_into_std_write(&current_chunk_ways, &mut writer, config)
                        .unwrap();

                    // Add metadata
                    chunk_metadata.push(ChunkMetadata {
                        chunk_id,
                        node_count: 0,
                        way_count: current_chunk_ways.len(),
                        bounding_box: None,
                    });

                    // Reset for next chunk
                    current_chunk_ways = Vec::new();
                    chunk_id += 1;
                    element_count = 0;
                }
            }
        })?;

        // Write final batch if any
        if !current_chunk_ways.is_empty() {
            let way_file = ways_dir.join(format!("ways_{}.bin", chunk_id));
            let file = File::create(&way_file)?;
            let mut writer = BufWriter::new(file);

            // Serialize ways
            let config = bincode::config::standard();
            bincode::serde::encode_into_std_write(&current_chunk_ways, &mut writer, config)?;

            // Add metadata
            chunk_metadata.push(ChunkMetadata {
                chunk_id,
                node_count: 0,
                way_count: current_chunk_ways.len(),
                bounding_box: None,
            });
        }

        pb.finish_with_message("Way extraction complete");

        debug!(
            "Extracted {} way chunks in {:?}",
            chunk_metadata.len(),
            start_time.elapsed()
        );

        Ok(chunk_metadata)
    }

    /// Process way chunks to create road segments
    fn process_way_chunks(
        &self,
        temp_dir: &Path,
        way_chunks_info: &[ChunkMetadata],
        node_lookup_files: &[PathBuf],
    ) -> Result<Vec<PathBuf>> {
        let start_time = Instant::now();
        let ways_dir = temp_dir.join("ways");
        let segments_dir = temp_dir.join("segments");
        std::fs::create_dir_all(&segments_dir)?;

        let pb = ProgressBar::new(way_chunks_info.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) - Processing way chunks")?
            .progress_chars("##-"));

        // Load all node lookup files into memory (can be optimized with mmap for very large datasets)
        info!("Loading node lookup tables");
        let mut all_node_locations = HashMap::new();

        for lookup_file in node_lookup_files {
            let file = File::open(lookup_file)?;
            let mut reader = BufReader::new(file);

            // Deserialize node lookup
            let config = bincode::config::standard();
            let node_lookup: HashMap<u64, Coord<f64>> =
                bincode::serde::decode_from_std_read(&mut reader, config)?;

            // Merge with main lookup
            all_node_locations.extend(node_lookup);
        }

        info!("Loaded {} node locations", all_node_locations.len());

        let mut segment_files = Vec::new();

        // Process each way chunk
        for chunk in way_chunks_info {
            pb.inc(1);

            let way_file = ways_dir.join(format!("ways_{}.bin", chunk.chunk_id));
            let segment_file = segments_dir.join(format!("segments_{}.bin", chunk.chunk_id));

            // Read way chunk
            let file = File::open(&way_file)?;
            let mut reader = BufReader::new(file);

            // Deserialize way entries
            let config = bincode::config::standard();
            let ways: Vec<WayEntry> = bincode::serde::decode_from_std_read(&mut reader, config)?;

            // Create road segments
            let mut road_segments = Vec::new();
            let mut node_connections = HashMap::new();

            for way in ways {
                // Get coordinates for nodes
                let coordinates: Vec<Coord<f64>> = way
                    .nodes
                    .iter()
                    .filter_map(|&node_id| all_node_locations.get(&node_id).cloned())
                    .collect();

                if coordinates.len() > 1 {
                    let segment = WaySegment {
                        id: way.id,
                        nodes: way.nodes.clone(),
                        coordinates,
                        is_oneway: way.is_oneway,
                        highway_type: way.highway_type,
                        max_speed: way.max_speed,
                        connections: Vec::new(),
                        name: way.name,
                        metadata: way.metadata,
                        original_id: None,
                        split_id: None,
                    };

                    road_segments.push(segment);

                    // Build node-to-way index for connectivity
                    for &node_id in &way.nodes {
                        node_connections
                            .entry(node_id)
                            .or_insert_with(Vec::new)
                            .push(way.id);
                    }
                }
            }

            // Write segments and node connections
            let file = File::create(&segment_file)?;
            let mut writer = BufWriter::new(file);

            // Serialize as (segments, connections)
            let data = (road_segments, node_connections);
            bincode::serde::encode_into_std_write(&data, &mut writer, config)?;

            segment_files.push(segment_file);
        }

        pb.finish_with_message("Segment processing complete");

        debug!(
            "Processed {} segment files in {:?}",
            segment_files.len(),
            start_time.elapsed()
        );

        Ok(segment_files)
    }

    /// Build connectivity between segments
    fn build_connectivity(
        &self,
        temp_dir: &Path,
        segment_files: &[PathBuf],
    ) -> Result<Vec<PathBuf>> {
        let start_time = Instant::now();
        let connected_dir = temp_dir.join("connected");
        std::fs::create_dir_all(&connected_dir)?;

        let pb = ProgressBar::new(segment_files.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) - Building connectivity")?
            .progress_chars("##-"));

        // We need to merge all the node connections from all segments
        // to properly build connectivity across chunk boundaries
        info!("Merging node connections");
        let mut all_node_connections = HashMap::new();
        let mut all_segments = Vec::new();
        let mut segment_map = HashMap::new();

        // First pass: collect all segments and node connections
        for segment_file in segment_files {
            let file = File::open(segment_file)?;
            let mut reader = BufReader::new(file);

            // Deserialize (segments, connections)
            let config = bincode::config::standard();
            let (segments, connections): (Vec<WaySegment>, HashMap<u64, Vec<u64>>) =
                bincode::serde::decode_from_std_read(&mut reader, config)?;

            // Add segments to master list and build segment index
            for (i, segment) in segments.iter().enumerate() {
                segment_map.insert(segment.id, all_segments.len() + i);
            }
            all_segments.extend(segments);

            // Merge node connections
            for (node, ways) in connections {
                all_node_connections
                    .entry(node)
                    .or_insert_with(Vec::new)
                    .extend(ways);
            }
        }

        // Deduplicate connections
        for ways in all_node_connections.values_mut() {
            ways.sort_unstable();
            ways.dedup();
        }

        info!("Building connectivity for {} segments", all_segments.len());

        // Create a map to store segment connections
        let mut segment_connections: HashMap<u64, Vec<u64>> = HashMap::new();

        // Process each segment
        for segment in &all_segments {
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
                if let Some(connected_segments) = all_node_connections.get(&node) {
                    for &other_id in connected_segments {
                        // Skip self-connections
                        if other_id == segment_id {
                            continue;
                        }

                        // Get other segment index
                        if let Some(&other_idx) = segment_map.get(&other_id) {
                            let other_segment = &all_segments[other_idx];

                            // Check for oneway restriction on the other segment
                            if other_segment.is_oneway {
                                // Only connect if we're at the start node of the other segment
                                if other_segment.nodes.first() == Some(&node) {
                                    segment_connections
                                        .entry(segment_id)
                                        .or_insert_with(Vec::new)
                                        .push(other_id);
                                }
                            } else {
                                // Non-oneway segments can connect at any point
                                segment_connections
                                    .entry(segment_id)
                                    .or_insert_with(Vec::new)
                                    .push(other_id);
                            }
                        }
                    }
                }
            }
        }

        // Apply connections to segments
        for segment in all_segments.iter_mut() {
            if let Some(connections) = segment_connections.get(&segment.id) {
                segment.connections = connections.clone();
                // Remove duplicates
                segment.connections.sort_unstable();
                segment.connections.dedup();
            }
        }

        // Split segments back into chunks for further processing
        // This time based on geographic location to help with tile generation
        info!("Splitting connected segments into geographic chunks");

        let mut connected_files = Vec::new();
        let num_chunks = (all_segments.len() as f64 / self.way_batch_size as f64).ceil() as usize;

        // Use a geographic partition based on longitude (can be improved with quadtrees for real-world use)
        let mut min_lon = f64::MAX;
        let mut max_lon = f64::MIN;

        for segment in &all_segments {
            if let Some(first) = segment.coordinates.first() {
                min_lon = min_lon.min(first.x);
                max_lon = max_lon.max(first.x);
            }
        }

        let lon_step = if num_chunks > 1 {
            (max_lon - min_lon) / num_chunks as f64
        } else {
            max_lon - min_lon
        };

        // Create vector of vectors to hold segments for each chunk
        let mut chunk_segments: Vec<Vec<WaySegment>> =
            (0..num_chunks).map(|_| Vec::new()).collect();

        // Assign segments to chunks based on centroid longitude
        for segment in all_segments {
            let centroid = segment.centroid();
            let chunk_idx = if num_chunks > 1 {
                ((centroid.x() - min_lon) / lon_step)
                    .floor()
                    .min((num_chunks - 1) as f64) as usize
            } else {
                0
            };

            chunk_segments[chunk_idx].push(segment);
        }

        // Write each chunk to disk
        for (i, segments) in chunk_segments.iter().enumerate() {
            pb.inc(1);

            if segments.is_empty() {
                continue;
            }

            let connected_file = connected_dir.join(format!("connected_{}.bin", i));
            let file = File::create(&connected_file)?;
            let mut writer = BufWriter::new(file);

            // Serialize segments
            let config = bincode::config::standard();
            bincode::serde::encode_into_std_write(&segments, &mut writer, config)?;

            connected_files.push(connected_file);
        }

        pb.finish_with_message("Connectivity building complete");

        debug!(
            "Built connectivity and created {} connected segment files in {:?}",
            connected_files.len(),
            start_time.elapsed()
        );

        Ok(connected_files)
    }

    /// Generate tiles from connected segments
    fn generate_tiles(
        &self,
        temp_dir: &Path,
        connected_segment_files: &[PathBuf],
        output_dir: &str,
    ) -> Result<()> {
        let start_time = Instant::now();

        // Create progress bar
        let pb = ProgressBar::new(connected_segment_files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) - Generating tiles",
                )?
                .progress_chars("##-"),
        );

        // Create output directory
        std::fs::create_dir_all(output_dir)?;

        // Maps to track tiles across all chunks
        let mut all_tiles = HashMap::new();
        let mut all_segment_tile_map = HashMap::new();

        // Process each connected segment file
        for connected_file in connected_segment_files {
            pb.inc(1);

            // Read segments
            let file = File::open(connected_file)?;
            let mut reader = BufReader::new(file);

            // Deserialize connected segments
            let config = bincode::config::standard();
            let segments: Vec<WaySegment> =
                bincode::serde::decode_from_std_read(&mut reader, config)?;

            // Process segments for tiling
            for segment in segments {
                let centroid = segment.centroid();
                let mut current_size = self.config.base_tile_size;
                let mut current_depth = 0;

                // Calculate initial tile ID
                let mut tile_id = self.calculate_tile_id(&centroid, current_size, current_depth);

                // Adaptive tile splitting
                loop {
                    let segments = all_tiles.entry(tile_id.clone()).or_insert_with(Vec::new);

                    // Check density or max depth
                    if segments.len() < self.config.min_tile_density
                        || current_depth >= self.config.max_split_depth
                    {
                        // Store segment in this tile
                        all_segment_tile_map.insert(segment.id, tile_id.clone());
                        segments.push(segment.clone());
                        break;
                    }

                    // Split tile and try again
                    current_size /= 2.0;
                    current_depth += 1;
                    tile_id = self.calculate_tile_id(&centroid, current_size, current_depth);
                }
            }
        }

        // Create a new progress bar for tile writing
        let pb_write = ProgressBar::new(all_tiles.len() as u64);
        pb_write.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) - Writing tiles",
                )?
                .progress_chars("##-"),
        );

        // Write tiles to disk
        for (tile_id, segments) in &all_tiles {
            pb_write.inc(1);

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

        pb_write.finish_with_message("Tile writing complete");

        info!(
            "Generated {} tiles with adaptive sizing in {:?}",
            all_tiles.len(),
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
