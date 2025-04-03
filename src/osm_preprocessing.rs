use anyhow::Result;
use geo::Haversine;
use geo::{Coord, LineString, Point, algorithm::Distance};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::{debug, info, warn};
use osmpbf::{Element, ElementReader};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, RwLock},
    time::Instant,
};

use crate::route_matcher::{TileConfig, calculate_heading};

/// Represents a road segment in the processed network
#[derive(Clone, Debug, Serialize, Deserialize)]
#[derive(Default)]
pub struct WaySegment {
    pub id: u64,
    pub osm_way_id: u64,
    pub nodes: Vec<u64>,
    pub coordinates: Vec<Coord<f64>>,
    pub is_oneway: bool,
    pub highway_type: String,
    pub max_speed: Option<f64>,
    pub connections: Vec<u64>,
    pub name: Option<String>,
    pub metadata: Option<BTreeMap<String, String>>, // Store additional OSM tags
}

// Default implementation for backward compatibility

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
#[derive(Debug, Serialize, Deserialize, Clone)]
struct ChunkMetadata {
    chunk_id: usize,
    node_count: usize,
    way_count: usize,
    bounding_box: Option<geo::Rect<f64>>,
    file_path: PathBuf,
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

/// OSM Network Processor with parallel streaming support
pub struct OSMProcessor {
    config: TileConfig,
    // Configuration for streaming processing
    chunk_size: usize,                 // Number of elements per chunk
    node_batch_size: usize,            // Number of nodes to process in memory at once
    way_batch_size: usize,             // Number of ways to process in memory at once
    num_threads: usize,                // Number of threads to use
    temp_dir: Option<PathBuf>,         // Optional custom temp directory
    next_segment_id: Arc<AtomicUsize>, // Track the next segment ID (0-based)
}

impl OSMProcessor {
    pub fn new(config: TileConfig) -> Self {
        Self {
            config,
            chunk_size: 1_000_000,      // Process 1M elements per chunk
            node_batch_size: 5_000_000, // Process 5M nodes at once
            way_batch_size: 500_000,    // Process 500K ways at once
            num_threads: rayon::current_num_threads(), // Use all available CPUs by default
            temp_dir: None,
            next_segment_id: Arc::new(AtomicUsize::new(0)),
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

    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    pub fn with_temp_dir(mut self, temp_dir: PathBuf) -> Self {
        self.temp_dir = Some(temp_dir);
        self
    }

    /// Process entire OSM PBF file and generate tiles with streaming approach
    pub fn process_pbf(&self, pbf_path: &str, output_dir: &str) -> Result<()> {
        let start_time = Instant::now();

        // Initialize rayon thread pool with configured threads
        rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .build_global()
            .unwrap_or_else(|e| warn!("Failed to configure thread pool: {}", e));

        // Create temporary directory for intermediate data
        let temp_dir = match &self.temp_dir {
            Some(dir) => {
                std::fs::create_dir_all(dir)?;
                dir.clone()
            }
            None => PathBuf::from_str("/tmp/ct").unwrap(),
        };

        info!("Using temporary directory: {:?}", temp_dir);
        info!("Using {} worker threads", self.num_threads);

        // Create multi-progress for all steps
        let mp = MultiProgress::new();

        // Step 1: Extract PBF elements in iterative chunks
        info!("Step 1: Extracting elements from PBF");
        let (node_chunks_info, way_chunks_info) =
            self.extract_elements_in_chunks(pbf_path, &temp_dir, &mp)?;

        // Step 2: Process nodes to create node location lookup files (in parallel)
        info!("Step 2: Processing node locations in parallel");
        let node_location_files =
            self.process_node_chunks_parallel(&temp_dir, &node_chunks_info, &mp)?;

        // Step 3: Process ways to create way segments (in parallel)
        info!("Step 3: Processing ways into segments in parallel");
        let segment_files = self.process_way_chunks_parallel(
            &temp_dir,
            &way_chunks_info,
            &node_location_files,
            &mp,
        )?;

        // Step 4: Build connectivity between segments (in parallel)
        info!("Step 4: Building connectivity in parallel");
        let connected_segment_files =
            self.build_connectivity_parallel(&temp_dir, &segment_files, &mp)?;

        // Step 5: Generate tiles (in parallel)
        info!("Step 5: Generating adaptive tiles in parallel");
        self.generate_tiles_parallel(&temp_dir, &connected_segment_files, output_dir, &mp)?;

        // Cleanup temporary files (optional, comment out to keep for debugging)
        if self.temp_dir.is_none() {
            info!("Cleaning up temporary files");
            std::fs::remove_dir_all(&temp_dir)?;
        }

        info!("OSM processing completed in {:?}", start_time.elapsed());
        Ok(())
    }

    /// Extract elements in chunks to overcome PBF parser limitations
    fn extract_elements_in_chunks(
        &self,
        pbf_path: &str,
        temp_dir: &Path,
        mp: &MultiProgress,
    ) -> Result<(Vec<ChunkMetadata>, Vec<ChunkMetadata>)> {
        let start_time = Instant::now();

        // Create directories
        let nodes_dir = temp_dir.join("nodes");
        let ways_dir = temp_dir.join("ways");
        std::fs::create_dir_all(&nodes_dir)?;
        std::fs::create_dir_all(&ways_dir)?;

        // Get file size for progress bar
        let file_size = std::fs::metadata(pbf_path)?.len();
        let pb = mp.add(ProgressBar::new(file_size));
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) - Extracting elements")?
            .progress_chars("##-"));

        // Important highway types to filter
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

        // Counters for chunk IDs
        let node_chunk_counter = AtomicUsize::new(0);
        let way_chunk_counter = AtomicUsize::new(0);

        // Collection for storing metadata
        let node_chunks = Arc::new(Mutex::new(Vec::new()));
        let way_chunks = Arc::new(Mutex::new(Vec::new()));

        // Read PBF file and collect elements
        let reader = ElementReader::from_path(pbf_path)?;

        // Buffers for batching
        let mut nodes_buffer: Vec<NodeEntry> = Vec::with_capacity(self.node_batch_size);
        let mut ways_buffer: Vec<WayEntry> = Vec::with_capacity(self.way_batch_size);
        let mut elements_processed = 0;

        reader.for_each(|element| {
            pb.inc(1);
            elements_processed += 1;

            match element {
                Element::Node(node) => {
                    nodes_buffer.push(NodeEntry {
                        id: node.id() as u64,
                        coord: Coord {
                            x: node.lon(),
                            y: node.lat(),
                        },
                    });
                }
                Element::DenseNode(node) => {
                    nodes_buffer.push(NodeEntry {
                        id: node.id() as u64,
                        coord: Coord {
                            x: node.lon(),
                            y: node.lat(),
                        },
                    });
                }
                Element::Way(way) => {
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
                        let max_speed = way.tags().find(|(key, _)| *key == "maxspeed").and_then(
                            |(_, value)| {
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
                            },
                        );

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
                            ways_buffer.push(WayEntry {
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
                _ => {}
            }

            // Process node buffer if it's full or we've reached a chunk boundary
            if (nodes_buffer.len() >= self.node_batch_size || elements_processed % self.chunk_size == 0) && !nodes_buffer.is_empty() {
                // Get a new chunk ID
                let chunk_id = node_chunk_counter.fetch_add(1, Ordering::SeqCst);

                // Process these nodes (can be done in parallel for large batches)
                let file_path = nodes_dir.join(format!("nodes_{}.bin", chunk_id));
                let nodes_to_write = std::mem::replace(
                    &mut nodes_buffer,
                    Vec::with_capacity(self.node_batch_size),
                );

                // Write in parallel - this would be even better with async I/O
                if let Ok(file) = File::create(&file_path) {
                    let mut writer = BufWriter::new(file);
                    let config = bincode::config::standard();

                    if bincode::serde::encode_into_std_write(
                        &nodes_to_write,
                        &mut writer,
                        config,
                    )
                    .is_ok()
                    {
                        let mut chunks = node_chunks.lock().unwrap();
                        chunks.push(ChunkMetadata {
                            chunk_id,
                            node_count: nodes_to_write.len(),
                            way_count: 0,
                            bounding_box: None,
                            file_path: file_path.clone(),
                        });
                    }
                }
            }

            // Process way buffer if it's full or we've reached a chunk boundary
            if (ways_buffer.len() >= self.way_batch_size || elements_processed % self.chunk_size == 0) && !ways_buffer.is_empty() {
                // Get a new chunk ID
                let chunk_id = way_chunk_counter.fetch_add(1, Ordering::SeqCst);

                // Process these ways
                let file_path = ways_dir.join(format!("ways_{}.bin", chunk_id));
                let ways_to_write = std::mem::replace(
                    &mut ways_buffer,
                    Vec::with_capacity(self.way_batch_size),
                );

                // Write in parallel
                if let Ok(file) = File::create(&file_path) {
                    let mut writer = BufWriter::new(file);
                    let config = bincode::config::standard();

                    if bincode::serde::encode_into_std_write(
                        &ways_to_write,
                        &mut writer,
                        config,
                    )
                    .is_ok()
                    {
                        let mut chunks = way_chunks.lock().unwrap();
                        chunks.push(ChunkMetadata {
                            chunk_id,
                            node_count: 0,
                            way_count: ways_to_write.len(),
                            bounding_box: None,
                            file_path: file_path.clone(),
                        });
                    }
                }
            }
        })?;

        // Process any remaining nodes
        if !nodes_buffer.is_empty() {
            let chunk_id = node_chunk_counter.fetch_add(1, Ordering::SeqCst);
            let file_path = nodes_dir.join(format!("nodes_{}.bin", chunk_id));

            if let Ok(file) = File::create(&file_path) {
                let mut writer = BufWriter::new(file);
                let config = bincode::config::standard();

                if bincode::serde::encode_into_std_write(&nodes_buffer, &mut writer, config).is_ok()
                {
                    let mut chunks = node_chunks.lock().unwrap();
                    chunks.push(ChunkMetadata {
                        chunk_id,
                        node_count: nodes_buffer.len(),
                        way_count: 0,
                        bounding_box: None,
                        file_path: file_path.clone(),
                    });
                }
            }
        }

        // Process any remaining ways
        if !ways_buffer.is_empty() {
            let chunk_id = way_chunk_counter.fetch_add(1, Ordering::SeqCst);
            let file_path = ways_dir.join(format!("ways_{}.bin", chunk_id));

            if let Ok(file) = File::create(&file_path) {
                let mut writer = BufWriter::new(file);
                let config = bincode::config::standard();

                if bincode::serde::encode_into_std_write(&ways_buffer, &mut writer, config).is_ok()
                {
                    let mut chunks = way_chunks.lock().unwrap();
                    chunks.push(ChunkMetadata {
                        chunk_id,
                        node_count: 0,
                        way_count: ways_buffer.len(),
                        bounding_box: None,
                        file_path: file_path.clone(),
                    });
                }
            }
        }

        pb.finish_with_message("Element extraction complete");

        // Get results
        let node_chunks_info = node_chunks.lock().unwrap().clone();
        let way_chunks_info = way_chunks.lock().unwrap().clone();

        debug!(
            "Split PBF into {} node chunks and {} way chunks in {:?}",
            node_chunks_info.len(),
            way_chunks_info.len(),
            start_time.elapsed()
        );

        Ok((node_chunks_info, way_chunks_info))
    }

    /// Process node chunks to create node location lookup files in parallel
    fn process_node_chunks_parallel(
        &self,
        temp_dir: &Path,
        node_chunks_info: &[ChunkMetadata],
        mp: &MultiProgress,
    ) -> Result<Vec<PathBuf>> {
        let start_time = Instant::now();
        let node_lookups_dir = temp_dir.join("node_lookups");
        std::fs::create_dir_all(&node_lookups_dir)?;

        let pb = mp.add(ProgressBar::new(node_chunks_info.len() as u64));
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) - Processing node chunks")?
            .progress_chars("##-"));

        // Create thread-safe collection for results
        let node_location_files = Arc::new(Mutex::new(Vec::new()));

        // Process chunks in parallel using rayon
        node_chunks_info.par_iter().for_each(|chunk| {
            // Read node chunk
            if let Ok(file) = File::open(&chunk.file_path) {
                let mut reader = BufReader::new(file);

                // Deserialize node entries
                let config = bincode::config::standard();
                if let Ok(nodes) = bincode::serde::decode_from_std_read::<Vec<NodeEntry>, _, _>(
                    &mut reader,
                    config,
                ) {
                    // Create node location map
                    let mut node_locations = HashMap::new();
                    for node in nodes {
                        node_locations.insert(node.id, node.coord);
                    }

                    // Write node location map
                    let lookup_file =
                        node_lookups_dir.join(format!("node_lookup_{}.bin", chunk.chunk_id));
                    if let Ok(file) = File::create(&lookup_file) {
                        let mut writer = BufWriter::new(file);
                        if bincode::serde::encode_into_std_write(
                            &node_locations,
                            &mut writer,
                            config,
                        )
                        .is_ok()
                        {
                            // Add to results
                            let mut files = node_location_files.lock().unwrap();
                            files.push(lookup_file);
                        }
                    }
                }
            }

            // Update progress
            pb.inc(1);
        });

        pb.finish_with_message("Node lookup processing complete");

        // Get results
        let result = node_location_files.lock().unwrap().clone();

        debug!(
            "Processed {} node lookup files in parallel in {:?}",
            result.len(),
            start_time.elapsed()
        );

        Ok(result)
    }

    /// Process way chunks to create road segments in parallel
    fn process_way_chunks_parallel(
        &self,
        temp_dir: &Path,
        way_chunks_info: &[ChunkMetadata],
        node_lookup_files: &[PathBuf],
        mp: &MultiProgress,
    ) -> Result<Vec<PathBuf>> {
        let start_time = Instant::now();
        let segments_dir = temp_dir.join("segments");
        std::fs::create_dir_all(&segments_dir)?;

        let pb = mp.add(ProgressBar::new(way_chunks_info.len() as u64));
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) - Processing way chunks")?
            .progress_chars("##-"));

        // Create shared node lookup table
        info!("Loading node lookup tables");
        let node_lookup_pb = mp.add(ProgressBar::new(node_lookup_files.len() as u64));
        node_lookup_pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) - Loading node lookups")?
            .progress_chars("##-"));

        // Load node lookups in parallel
        let all_node_locations = Arc::new(RwLock::new(HashMap::new()));

        node_lookup_files.par_iter().for_each(|lookup_file| {
            if let Ok(file) = File::open(lookup_file) {
                let mut reader = BufReader::new(file);

                // Deserialize node lookup
                let config = bincode::config::standard();
                if let Ok(node_lookup) = bincode::serde::decode_from_std_read::<
                    HashMap<u64, Coord<f64>>,
                    _,
                    _,
                >(&mut reader, config)
                {
                    // Merge with main lookup
                    let mut locations = all_node_locations.write().unwrap();
                    locations.extend(node_lookup);
                }
            }
            node_lookup_pb.inc(1);
        });

        node_lookup_pb.finish_with_message("Node lookups loaded");

        let node_count = all_node_locations.read().unwrap().len();
        info!("Loaded {} node locations", node_count);

        // Create thread-safe collection for results
        let segment_files = Arc::new(Mutex::new(Vec::new()));

        // Process chunks in parallel using rayon
        way_chunks_info.par_iter().for_each(|chunk| {
            // Read way chunk
            if let Ok(file) = File::open(&chunk.file_path) {
                let mut reader = BufReader::new(file);

                // Deserialize way entries
                let config = bincode::config::standard();
                if let Ok(ways) =
                    bincode::serde::decode_from_std_read::<Vec<WayEntry>, _, _>(&mut reader, config)
                {
                    // Create road segments
                    let mut road_segments = Vec::new();
                    let mut node_connections = HashMap::new();
                    let node_locations = all_node_locations.read().unwrap();

                    for way in ways {
                        // Get coordinates for nodes
                        let coordinates: Vec<Coord<f64>> = way
                            .nodes
                            .iter()
                            .filter_map(|&node_id| node_locations.get(&node_id).cloned())
                            .collect();

                        if coordinates.len() > 1 {
                            // Assign new sequential ID
                            let new_id = self.next_segment_id.fetch_add(1, Ordering::SeqCst) as u64;

                            let segment = WaySegment {
                                id: new_id,
                                osm_way_id: way.id, // Original OSM way ID
                                nodes: way.nodes.clone(),
                                coordinates,
                                is_oneway: way.is_oneway,
                                highway_type: way.highway_type,
                                max_speed: way.max_speed,
                                connections: Vec::new(),
                                name: way.name,
                                metadata: way.metadata,
                            };

                            road_segments.push(segment);

                            // Build node-to-way index for connectivity using the new ID
                            for &node_id in &way.nodes {
                                node_connections
                                    .entry(node_id)
                                    .or_insert_with(Vec::new)
                                    .push(new_id); // Use new ID for connections
                            }
                        }
                    }

                    // Write segments and node connections
                    let segment_file =
                        segments_dir.join(format!("segments_{}.bin", chunk.chunk_id));
                    if let Ok(file) = File::create(&segment_file) {
                        let mut writer = BufWriter::new(file);

                        // Serialize as (segments, connections)
                        let data = (road_segments, node_connections);
                        if bincode::serde::encode_into_std_write(&data, &mut writer, config).is_ok()
                        {
                            // Add to results
                            let mut files = segment_files.lock().unwrap();
                            files.push(segment_file);
                        }
                    }
                }
            }

            pb.inc(1);
        });

        pb.finish_with_message("Segment processing complete");

        // Get results
        let result = segment_files.lock().unwrap().clone();

        debug!(
            "Processed {} segment files in parallel in {:?}",
            result.len(),
            start_time.elapsed()
        );

        Ok(result)
    }

    /// Build connectivity between segments in parallel
    fn build_connectivity_parallel(
        &self,
        temp_dir: &Path,
        segment_files: &[PathBuf],
        mp: &MultiProgress,
    ) -> Result<Vec<PathBuf>> {
        let start_time = Instant::now();
        let connected_dir = temp_dir.join("connected");
        std::fs::create_dir_all(&connected_dir)?;

        let pb = mp.add(ProgressBar::new(segment_files.len() as u64));
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) - Merging segment data")?
            .progress_chars("##-"));

        // We need to merge all the node connections from all segments
        // to properly build connectivity across chunk boundaries
        info!("Building connectivity graph");

        // Load all segments and connections in parallel
        let all_segments = Arc::new(Mutex::new(Vec::new()));
        let all_node_connections = Arc::new(Mutex::new(HashMap::new()));

        // First pass: collect all segments and node connections in parallel
        segment_files.par_iter().for_each(|segment_file| {
            if let Ok(file) = File::open(segment_file) {
                let mut reader = BufReader::new(file);

                // Deserialize (segments, connections)
                let config = bincode::config::standard();
                if let Ok((segments, connections)) = bincode::serde::decode_from_std_read::<
                    (Vec<WaySegment>, HashMap<u64, Vec<u64>>),
                    _,
                    _,
                >(&mut reader, config)
                {
                    // Add segments to master list
                    {
                        let mut all_segs = all_segments.lock().unwrap();
                        all_segs.extend(segments);
                    }

                    // Merge node connections
                    {
                        let mut all_conns = all_node_connections.lock().unwrap();
                        for (node, ways) in connections {
                            all_conns.entry(node).or_insert_with(Vec::new).extend(ways);
                        }
                    }
                }
            }

            pb.inc(1);
        });

        pb.finish_with_message("Segment data merged");

        // Process segment data
        info!("Building segment connectivity");
        let segment_pb = mp.add(ProgressBar::new(1));

        // Create segment map for lookup - use the segment's new ID as the key
        let mut all_segs = all_segments.lock().unwrap();
        let mut all_conns = all_node_connections.lock().unwrap();

        segment_pb.set_length(all_segs.len() as u64);
        segment_pb.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) - Building connectivity")?
        .progress_chars("##-"));

        // Deduplicate connections
        for ways in all_conns.values_mut() {
            ways.sort_unstable();
            ways.dedup();
        }

        // Create segment map for lookup using the new ID
        let mut segment_map = HashMap::new();
        for (i, segment) in all_segs.iter().enumerate() {
            segment_map.insert(segment.id, i);
        }

        // Build connections in parallel using chunks of segments
        let segment_chunks: Vec<Vec<_>> = all_segs
            .chunks_mut(5000)
            .map(|chunk| chunk.to_vec())
            .collect();

        let segment_connections = Arc::new(Mutex::new(HashMap::<u64, Vec<u64>>::new()));

        segment_chunks.par_iter().for_each(|segments| {
            let mut local_connections = HashMap::new();

            for segment in segments {
                if segment.nodes.len() < 2 {
                    continue;
                }

                segment_pb.inc(1);

                let segment_id = segment.id; // Use new ID for connectivity
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
                    if let Some(connected_segments) = all_conns.get(&node) {
                        for &other_id in connected_segments {
                            // Skip self-connections
                            if other_id == segment_id {
                                continue;
                            }

                            // Get other segment index
                            if let Some(&other_idx) = segment_map.get(&other_id) {
                                let other_segment = &all_segs[other_idx];

                                // Check for oneway restriction on the other segment
                                if other_segment.is_oneway {
                                    // Only connect if we're at the start node of the other segment
                                    if other_segment.nodes.first() == Some(&node) {
                                        local_connections
                                            .entry(segment_id)
                                            .or_insert_with(Vec::new)
                                            .push(other_id);
                                    }
                                } else {
                                    // Non-oneway segments can connect at any point
                                    local_connections
                                        .entry(segment_id)
                                        .or_insert_with(Vec::new)
                                        .push(other_id);
                                }
                            }
                        }
                    }
                }
            }

            // Merge connections
            let mut all_connections = segment_connections.lock().unwrap();
            for (id, connections) in local_connections {
                all_connections
                    .entry(id)
                    .or_default()
                    .extend(connections);
            }
        });

        // Apply connections to segments
        let connections = segment_connections.lock().unwrap();
        for segment in all_segs.iter_mut() {
            if let Some(conns) = connections.get(&segment.id) {
                segment.connections = conns.clone();
                segment.connections.sort_unstable();
                segment.connections.dedup();
            }
        }

        segment_pb.finish_with_message("Connectivity built");

        // Split segments into geographic chunks for parallel tile generation
        info!("Splitting connected segments into geographic chunks");
        let chunk_pb = mp.add(ProgressBar::new(all_segs.len() as u64));
        chunk_pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) - Creating geographic chunks")?
            .progress_chars("##-"));

        // Use a geographic partition based on longitude (can be improved with quadtrees)
        let num_chunks = (all_segs.len() as f64 / self.way_batch_size as f64).ceil() as usize;
        let num_chunks = num_chunks.max(self.num_threads);

        let mut min_lon = f64::MAX;
        let mut max_lon = f64::MIN;

        for segment in &*all_segs {
            if let Some(first) = segment.coordinates.first() {
                min_lon = min_lon.min(first.x);
                max_lon = max_lon.max(first.x);
            }
            chunk_pb.inc(1);
        }

        let lon_step = if num_chunks > 1 {
            (max_lon - min_lon) / num_chunks as f64
        } else {
            max_lon - min_lon
        };

        // Create chunks by geography
        let mut chunk_segments: Vec<Vec<WaySegment>> =
            (0..num_chunks).map(|_| Vec::new()).collect();

        for segment in all_segs.drain(..) {
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

        chunk_pb.finish_with_message("Geographic chunks created");

        // Write chunks to disk in parallel
        let write_pb = mp.add(ProgressBar::new(chunk_segments.len() as u64));
        write_pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) - Writing connected segments")?
            .progress_chars("##-"));

        let connected_files = Arc::new(Mutex::new(Vec::new()));

        chunk_segments
            .par_iter()
            .enumerate()
            .for_each(|(i, segments)| {
                if segments.is_empty() {
                    write_pb.inc(1);
                    return;
                }

                let connected_file = connected_dir.join(format!("connected_{}.bin", i));
                if let Ok(file) = File::create(&connected_file) {
                    let mut writer = BufWriter::new(file);

                    // Serialize segments
                    let config = bincode::config::standard();
                    if bincode::serde::encode_into_std_write(segments, &mut writer, config).is_ok()
                    {
                        let mut files = connected_files.lock().unwrap();
                        files.push(connected_file);
                    }
                }

                write_pb.inc(1);
            });

        write_pb.finish_with_message("Connected segment files written");

        // Get results
        let result = connected_files.lock().unwrap().clone();

        debug!(
            "Built connectivity and created {} connected segment files in parallel in {:?}",
            result.len(),
            start_time.elapsed()
        );

        Ok(result)
    }

    /// Generate tiles from connected segments in parallel
    fn generate_tiles_parallel(
        &self,
        temp_dir: &Path,
        connected_segment_files: &[PathBuf],
        output_dir: &str,
        mp: &MultiProgress,
    ) -> Result<()> {
        let start_time = Instant::now();

        // Create progress bars
        let pb = mp.add(ProgressBar::new(connected_segment_files.len() as u64));
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) - Loading segments for tiling")?
            .progress_chars("##-"));

        // Create output directory
        std::fs::create_dir_all(output_dir)?;

        // Load all segments in parallel, build tiles, and write in parallel
        let all_tiles = Arc::new(Mutex::new(HashMap::new()));
        let all_segment_tile_map = Arc::new(Mutex::new(HashMap::new()));
        let segment_count = Arc::new(AtomicUsize::new(0));

        // Process segment files in parallel
        connected_segment_files
            .par_iter()
            .for_each(|connected_file| {
                if let Ok(file) = File::open(connected_file) {
                    let mut reader = BufReader::new(file);

                    // Deserialize connected segments
                    let config = bincode::config::standard();
                    if let Ok(segments) = bincode::serde::decode_from_std_read::<
                        Vec<WaySegment>,
                        _,
                        _,
                    >(&mut reader, config)
                    {
                        // Local tile collections to minimize lock contention
                        let mut local_tiles = HashMap::new();
                        let mut local_segment_tile_map = HashMap::new();

                        // Update total segment count
                        segment_count.fetch_add(segments.len(), Ordering::Relaxed);

                        // Process segments for tiling
                        for segment in segments {
                            let centroid = segment.centroid();
                            let mut current_size = self.config.base_tile_size;
                            let mut current_depth = 0;

                            // Calculate initial tile ID
                            let mut tile_id =
                                self.calculate_tile_id(&centroid, current_size, current_depth);

                            // Adaptive tile splitting
                            loop {
                                let segments =
                                    local_tiles.entry(tile_id.clone()).or_insert_with(Vec::new);

                                // Check density or max depth
                                if segments.len() < self.config.min_tile_density
                                    || current_depth >= self.config.max_split_depth
                                {
                                    // Store segment in this tile
                                    local_segment_tile_map.insert(segment.id, tile_id.clone());
                                    segments.push(segment.clone());
                                    break;
                                }

                                // Split tile and try again
                                current_size /= 2.0;
                                current_depth += 1;
                                tile_id =
                                    self.calculate_tile_id(&centroid, current_size, current_depth);
                            }
                        }

                        // Merge local collections into global ones
                        {
                            let mut tiles = all_tiles.lock().unwrap();
                            for (tile_id, segments) in local_tiles {
                                tiles
                                    .entry(tile_id)
                                    .or_insert_with(Vec::new)
                                    .extend(segments);
                            }
                        }

                        {
                            let mut segment_tile_map = all_segment_tile_map.lock().unwrap();
                            segment_tile_map.extend(local_segment_tile_map);
                        }
                    }
                }

                pb.inc(1);
            });

        pb.finish_with_message("Segments loaded and tiled");

        // Get tile count for progress
        let tile_count = all_tiles.lock().unwrap().len();
        info!(
            "Writing {} tiles for {} segments",
            tile_count,
            segment_count.load(Ordering::Relaxed)
        );

        let write_pb = mp.add(ProgressBar::new(tile_count as u64));
        write_pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) - Writing tiles",
                )?
                .progress_chars("##-"),
        );

        // Extract tiles for parallel writing
        let mut tiles_to_write = Vec::new();
        {
            let mut tiles = all_tiles.lock().unwrap();
            let segment_tile_map = all_segment_tile_map.lock().unwrap().clone();

            for (tile_id, segments) in tiles.drain() {
                let bbox = self.calculate_bbox(&tile_id);

                // Create segment index for this tile
                let mut segment_index = HashMap::new();
                for segment in &segments {
                    if let Some(tid) = segment_tile_map.get(&segment.id) {
                        segment_index.insert(segment.id, tid.clone());
                    }
                }

                let tile_index = TileIndex {
                    tile_id: tile_id.clone(),
                    bbox,
                    road_segments: segments,
                    segment_index,
                };

                tiles_to_write.push((tile_id, tile_index));
            }
        }

        // Write tiles in parallel
        tiles_to_write.par_iter().for_each(|(tile_id, tile_index)| {
            let tile_path = Path::new(output_dir).join(format!("{}.bin", tile_id));
            let config = bincode::config::standard();

            if let Ok(serialized_tile) = bincode::serde::encode_to_vec(tile_index, config) {
                let _ = std::fs::write(&tile_path, serialized_tile);
            }

            write_pb.inc(1);
        });

        write_pb.finish_with_message("Tile writing complete");

        info!(
            "Generated {} tiles with adaptive sizing in parallel in {:?}",
            tiles_to_write.len(),
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
