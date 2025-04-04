use anyhow::{Result, anyhow, bail};
use geo::{BoundingRect, Rect};
use geo::{Coord, Haversine, Intersects, Line, LineString, Point, algorithm::Distance};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::{debug, info, trace, warn};
use rayon::prelude::*;
use rstar::{AABB, RTree, RTreeObject};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::osm_preprocessing::{TileIndex, WaySegment};
use crate::route_matcher::TileConfig;
use crate::tile_loader::TileLoader;

// Structure to track intersection work between tiles
pub struct IntersectionProcessor {
    // Configuration
    tile_directory: PathBuf,
    temp_directory: PathBuf,
    scratch_file: PathBuf,
    max_cached_tiles: usize,
    tile_config: TileConfig,

    // State tracking
    next_segment_id: AtomicU64,
    processed_tiles: Mutex<HashSet<String>>,

    // Tile loader with local modifications
    tile_loader: Arc<Mutex<TileLoader>>,
}

// Helper type alias
type LayerValue = i8;

// Represents a point where segments intersect
#[derive(Debug, Clone)]
struct IntersectionPoint {
    point: Point<f64>,
    segment_ids: Vec<u64>,
    layer_info: HashMap<u64, LayerValue>,
    is_bridge: HashMap<u64, bool>,
    is_tunnel: HashMap<u64, bool>,
}

// Represents a segment split at an intersection
#[derive(Debug, Clone)]
struct SplitSegment {
    original_id: u64,
    new_segments: Vec<WaySegment>,
}

// Wrapper for WaySegment to make it compatible with RTree
struct IndexedSegment {
    segment: WaySegment,
    bbox: Rect<f64>,
}

impl RTreeObject for IndexedSegment {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners(
            [self.bbox.min().x, self.bbox.min().y],
            [self.bbox.max().x, self.bbox.max().y],
        )
    }
}

impl IndexedSegment {
    fn new(segment: WaySegment) -> Self {
        // Calculate bounding rectangle for the segment
        let linestring = LineString::from(segment.coordinates.clone());
        let bbox = linestring.bounding_rect().unwrap_or(Rect::new(
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 0.0, y: 0.0 },
        ));

        Self { segment, bbox }
    }
}

impl IntersectionProcessor {
    pub fn new(
        tile_directory: impl Into<PathBuf>,
        temp_directory: impl Into<PathBuf>,
        max_cached_tiles: usize,
        tile_config: TileConfig,
    ) -> Result<Self> {
        let tile_directory = tile_directory.into();
        let temp_directory = temp_directory.into();

        if !tile_directory.exists() {
            bail!("Tile directory does not exist: {:?}", tile_directory);
        }

        // Create temp directory if it doesn't exist
        std::fs::create_dir_all(&temp_directory)?;

        let scratch_file = temp_directory.join("segment_id_counter.bin");

        // Initialize or read next segment ID from scratch file
        let next_segment_id = if scratch_file.exists() {
            let mut file = File::open(&scratch_file)?;
            let mut buffer = [0u8; 8];
            file.read_exact(&mut buffer)?;
            AtomicU64::new(u64::from_le_bytes(buffer))
        } else {
            // Start from 0 if file doesn't exist
            AtomicU64::new(0)
        };

        // Create TileLoader with caching
        let tile_loader = TileLoader::new(
            tile_directory.to_string_lossy().to_string(),
            max_cached_tiles,
            tile_config.clone(),
        );

        Ok(Self {
            tile_directory,
            temp_directory,
            scratch_file,
            max_cached_tiles,
            tile_config,
            next_segment_id,
            processed_tiles: Mutex::new(HashSet::new()),
            tile_loader: Arc::new(Mutex::new(tile_loader)),
        })
    }

    // Process a single tile and its intersections (with progress bar)
    fn process_tile(&self, tile_id: &str, progress: Option<&ProgressBar>) -> Result<()> {
        // Skip if already processed
        {
            let processed_tiles = self.processed_tiles.lock().unwrap();
            if processed_tiles.contains(tile_id) {
                if let Some(pb) = progress {
                    pb.inc(1);
                }
                return Ok(());
            }
        }

        // 1. Load the target tile
        let target_tile = {
            let mut loader = self.tile_loader.lock().unwrap();
            loader.load_tile(tile_id)?.clone()
        };

        // 2. Find neighboring tiles that could have intersections
        let neighboring_tile_ids = self.find_neighboring_tile_ids(tile_id)?;

        // 3. Load all neighboring tiles
        let neighboring_tiles = self.load_neighboring_tiles(&neighboring_tile_ids)?;

        // 4. Find all potential intersections
        let intersections = self.find_intersections(&target_tile, &neighboring_tiles)?;
        debug!(
            "Found {} potential intersections in tile {}",
            intersections.len(),
            tile_id
        );

        // 5. Split segments at intersections
        let (updated_segments, new_segments) =
            self.split_segments_at_intersections(&target_tile, intersections)?;

        // 6. Update connections between segments
        let final_segments = self.update_segment_connections(updated_segments, new_segments)?;

        // 7. Write the updated tile back to disk
        self.write_updated_tile(tile_id, target_tile, final_segments)?;

        // 8. Mark tile as processed
        {
            let mut processed_tiles = self.processed_tiles.lock().unwrap();
            processed_tiles.insert(tile_id.to_string());
        }

        if let Some(pb) = progress {
            pb.inc(1);
        }

        Ok(())
    }

    // Main entry point with progress bar
    pub fn process_all_tiles(&self) -> Result<()> {
        info!("Starting intersection processing for all tiles");

        // Get all tile IDs from the directory
        let tile_ids = self.list_all_tile_ids()?;
        info!("Found {} tiles to process", tile_ids.len());

        // Setup progress bar
        let mp = MultiProgress::new();
        let pb = mp.add(ProgressBar::new(tile_ids.len() as u64));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("##-"),
        );
        pb.set_message("Processing tiles");

        // Process tiles in parallel with rayon
        tile_ids
            .par_iter()
            .try_for_each(|tile_id| self.process_tile(tile_id, Some(&pb)))?;

        pb.finish_with_message("All tiles processed successfully");
        info!("Completed intersection processing for all tiles");

        // Update the scratch file with the final segment ID
        let final_id = self.next_segment_id.load(Ordering::SeqCst);
        let mut file = File::create(&self.scratch_file)?;
        file.write_all(&final_id.to_le_bytes())?;

        Ok(())
    }

    // List all tile IDs from the directory
    fn list_all_tile_ids(&self) -> Result<Vec<String>> {
        let mut tile_ids = Vec::new();

        let entries = std::fs::read_dir(&self.tile_directory)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().is_some_and(|ext| ext == "bin") {
                if let Some(file_stem) = path.file_stem() {
                    if let Some(tile_id) = file_stem.to_str() {
                        tile_ids.push(tile_id.to_string());
                    }
                }
            }
        }

        Ok(tile_ids)
    }

    // Find neighboring tile IDs based on the current tile ID
    fn find_neighboring_tile_ids(&self, tile_id: &str) -> Result<Vec<String>> {
        // Parse tile ID (format: "{x}_{y}_{depth}")
        let parts: Vec<&str> = tile_id.split('_').collect();
        if parts.len() != 3 {
            return Err(anyhow!("Invalid tile ID format: {}", tile_id));
        }

        let x: i32 = parts[0].parse()?;
        let y: i32 = parts[1].parse()?;
        let depth: u8 = parts[2].parse()?;

        // Get all 8 surrounding tiles
        let mut neighbors = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue; // Skip the tile itself
                }

                let neighbor_id = format!("{}_{}_{}", x + dx, y + dy, depth);

                // Check if the neighbor tile exists
                let neighbor_path = self.tile_directory.join(format!("{}.bin", neighbor_id));
                if neighbor_path.exists() {
                    neighbors.push(neighbor_id);
                }
            }
        }

        Ok(neighbors)
    }

    // Load all neighboring tiles
    fn load_neighboring_tiles(&self, tile_ids: &[String]) -> Result<HashMap<String, TileIndex>> {
        let mut tiles = HashMap::new();

        for tile_id in tile_ids {
            let tile = {
                let mut loader = self.tile_loader.lock().unwrap();
                loader.load_tile(tile_id)?.clone()
            };
            tiles.insert(tile_id.clone(), tile);
        }

        Ok(tiles)
    }

    // Find all potential intersections between target tile segments and neighboring tiles
    fn find_intersections(
        &self,
        target_tile: &TileIndex,
        neighboring_tiles: &HashMap<String, TileIndex>,
    ) -> Result<Vec<IntersectionPoint>> {
        let mut intersections = Vec::new();

        // First, find intersections within the target tile
        let internal_intersections = self.find_internal_intersections(target_tile)?;
        intersections.extend(internal_intersections);

        // Then, find intersections with neighboring tiles
        for neighbor_tile in neighboring_tiles.values() {
            let boundary_intersections =
                self.find_boundary_intersections(target_tile, neighbor_tile)?;
            intersections.extend(boundary_intersections);
        }

        // Deduplicate intersections that are very close to each other
        let deduplicated = self.deduplicate_intersections(intersections);

        Ok(deduplicated)
    }

    // Find internal intersections using R-tree spatial indexing
    fn find_internal_intersections(&self, tile: &TileIndex) -> Result<Vec<IntersectionPoint>> {
        let mut intersections = Vec::new();
        let segments = &tile.road_segments;

        // Skip the expensive work if there are too few segments
        if segments.len() < 2 {
            return Ok(intersections);
        }

        // Build R-tree index
        let indexed_segments: Vec<IndexedSegment> = segments
            .iter()
            .map(|segment| IndexedSegment::new(segment.clone()))
            .collect();

        let rtree = RTree::bulk_load(indexed_segments);

        // For each segment, find potential intersections using the R-tree
        for segment1 in segments {
            // Create a bounding rectangle for the segment
            let linestring1 = LineString::from(segment1.coordinates.clone());
            let bbox1 = match linestring1.bounding_rect() {
                Some(rect) => rect,
                None => continue, // Skip segments with no valid bounding box
            };

            // Query the R-tree for overlapping bounding boxes
            let query_box = AABB::from_corners(
                [bbox1.min().x, bbox1.min().y],
                [bbox1.max().x, bbox1.max().y],
            );

            // Get potential intersecting segments
            let potential_intersections = rtree.locate_in_envelope_intersecting(&query_box);

            for indexed_segment2 in potential_intersections {
                let segment2 = &indexed_segment2.segment;

                // Skip if it's the same segment or if segments are already connected
                if segment1.id == segment2.id
                    || segment1.connections.contains(&segment2.id)
                    || segment2.connections.contains(&segment1.id)
                {
                    continue;
                }

                // Get layer info and bridge/tunnel status
                let layer1 = self.get_segment_layer(segment1);
                let layer2 = self.get_segment_layer(segment2);

                let is_bridge1 = self.is_segment_bridge(segment1);
                let is_bridge2 = self.is_segment_bridge(segment2);

                let is_tunnel1 = self.is_segment_tunnel(segment1);
                let is_tunnel2 = self.is_segment_tunnel(segment2);

                // Skip if on different layers or one is a bridge/tunnel and the other isn't
                if layer1 != layer2 || is_bridge1 != is_bridge2 || is_tunnel1 != is_tunnel2 {
                    continue;
                }

                // Check for intersections between line segments
                if let Some(intersection) = self.find_line_intersections(segment1, segment2) {
                    let mut layer_info = HashMap::new();
                    layer_info.insert(segment1.id, layer1);
                    layer_info.insert(segment2.id, layer2);

                    let mut is_bridge = HashMap::new();
                    is_bridge.insert(segment1.id, is_bridge1);
                    is_bridge.insert(segment2.id, is_bridge2);

                    let mut is_tunnel = HashMap::new();
                    is_tunnel.insert(segment1.id, is_tunnel1);
                    is_tunnel.insert(segment2.id, is_tunnel2);

                    intersections.push(IntersectionPoint {
                        point: intersection,
                        segment_ids: vec![segment1.id, segment2.id],
                        layer_info,
                        is_bridge,
                        is_tunnel,
                    });
                }
            }
        }

        Ok(intersections)
    }

    // Find boundary intersections using R-tree spatial indexing
    fn find_boundary_intersections(
        &self,
        tile1: &TileIndex,
        tile2: &TileIndex,
    ) -> Result<Vec<IntersectionPoint>> {
        let mut intersections = Vec::new();

        // Skip the expensive work if either tile has too few segments
        if tile1.road_segments.is_empty() || tile2.road_segments.is_empty() {
            return Ok(intersections);
        }

        // Build R-tree index for tile2
        let indexed_segments: Vec<IndexedSegment> = tile2
            .road_segments
            .iter()
            .map(|segment| IndexedSegment::new(segment.clone()))
            .collect();

        let rtree = RTree::bulk_load(indexed_segments);

        // For each segment in tile1, find potential intersections with segments in tile2
        for segment1 in &tile1.road_segments {
            // Create a bounding rectangle for the segment
            let linestring1 = LineString::from(segment1.coordinates.clone());
            let bbox1 = match linestring1.bounding_rect() {
                Some(rect) => rect,
                None => continue, // Skip segments with no valid bounding box
            };

            // Query the R-tree for overlapping bounding boxes
            let query_box = AABB::from_corners(
                [bbox1.min().x, bbox1.min().y],
                [bbox1.max().x, bbox1.max().y],
            );

            // Get potential intersecting segments from tile2
            let potential_intersections = rtree.locate_in_envelope_intersecting(&query_box);

            for indexed_segment2 in potential_intersections {
                let segment2 = &indexed_segment2.segment;

                // Skip if segments are already connected
                if segment1.connections.contains(&segment2.id)
                    || segment2.connections.contains(&segment1.id)
                {
                    continue;
                }

                // Get layer info and bridge/tunnel status
                let layer1 = self.get_segment_layer(segment1);
                let layer2 = self.get_segment_layer(segment2);

                let is_bridge1 = self.is_segment_bridge(segment1);
                let is_bridge2 = self.is_segment_bridge(segment2);

                let is_tunnel1 = self.is_segment_tunnel(segment1);
                let is_tunnel2 = self.is_segment_tunnel(segment2);

                // Skip if on different layers or one is a bridge/tunnel and the other isn't
                if layer1 != layer2 || is_bridge1 != is_bridge2 || is_tunnel1 != is_tunnel2 {
                    continue;
                }

                // Check for intersections between line segments
                if let Some(intersection) = self.find_line_intersections(segment1, segment2) {
                    let mut layer_info = HashMap::new();
                    layer_info.insert(segment1.id, layer1);
                    layer_info.insert(segment2.id, layer2);

                    let mut is_bridge = HashMap::new();
                    is_bridge.insert(segment1.id, is_bridge1);
                    is_bridge.insert(segment2.id, is_bridge2);

                    let mut is_tunnel = HashMap::new();
                    is_tunnel.insert(segment1.id, is_tunnel1);
                    is_tunnel.insert(segment2.id, is_tunnel2);

                    intersections.push(IntersectionPoint {
                        point: intersection,
                        segment_ids: vec![segment1.id, segment2.id],
                        layer_info,
                        is_bridge,
                        is_tunnel,
                    });
                }
            }
        }

        Ok(intersections)
    }

    // Find line segment intersections between two road segments
    fn find_line_intersections(
        &self,
        segment1: &WaySegment,
        segment2: &WaySegment,
    ) -> Option<Point<f64>> {
        // Validate input segments
        if segment1.coordinates.len() < 2 || segment2.coordinates.len() < 2 {
            return None;
        }

        // Create LineStrings from segment coordinates
        let line_string1 = LineString::from(segment1.coordinates.clone());
        let line_string2 = LineString::from(segment2.coordinates.clone());

        // Quick check using geo's Intersects trait
        if !line_string1.intersects(&line_string2) {
            return None;
        }

        // Check each line segment pair for precise intersection point
        // For multi-point linestrings, this properly checks all constituent segments
        for line1 in line_string1.lines() {
            for line2 in line_string2.lines() {
                if let Some(intersection) = line_intersection(&line1, &line2) {
                    // Verify it's not at an endpoint (which would indicate the segments are already connected)
                    if !is_point_at_endpoint(&intersection, segment1)
                        && !is_point_at_endpoint(&intersection, segment2)
                    {
                        return Some(intersection);
                    }
                }
            }
        }

        None
    }

    // Deduplicate intersections that are very close to each other
    fn deduplicate_intersections(
        &self,
        intersections: Vec<IntersectionPoint>,
    ) -> Vec<IntersectionPoint> {
        if intersections.is_empty() {
            return Vec::new();
        }

        const DEDUP_DISTANCE_THRESHOLD: f64 = 1.0; // 1 meter

        let mut result = Vec::new();
        let mut processed = HashSet::new();

        for (i, intersection) in intersections.iter().enumerate() {
            if processed.contains(&i) {
                continue;
            }

            let mut combined_intersection = intersection.clone();
            processed.insert(i);

            // Look for nearby intersections to combine
            for (j, other) in intersections.iter().enumerate().skip(i + 1) {
                if processed.contains(&j) {
                    continue;
                }

                // Use Haversine.distance instead of haversine_distance
                let distance = Haversine.distance(intersection.point, other.point);
                if distance <= DEDUP_DISTANCE_THRESHOLD {
                    // Merge this intersection
                    for &seg_id in &other.segment_ids {
                        if !combined_intersection.segment_ids.contains(&seg_id) {
                            combined_intersection.segment_ids.push(seg_id);

                            // Copy layer and bridge/tunnel info if available
                            if let Some(&layer) = other.layer_info.get(&seg_id) {
                                combined_intersection.layer_info.insert(seg_id, layer);
                            }
                            if let Some(&is_bridge) = other.is_bridge.get(&seg_id) {
                                combined_intersection.is_bridge.insert(seg_id, is_bridge);
                            }
                            if let Some(&is_tunnel) = other.is_tunnel.get(&seg_id) {
                                combined_intersection.is_tunnel.insert(seg_id, is_tunnel);
                            }
                        }
                    }

                    processed.insert(j);
                }
            }

            result.push(combined_intersection);
        }

        result
    }

    // Split segments at intersection points
    fn split_segments_at_intersections(
        &self,
        tile: &TileIndex,
        intersections: Vec<IntersectionPoint>,
    ) -> Result<(Vec<WaySegment>, Vec<WaySegment>)> {
        let mut updated_segments = Vec::new();
        let mut new_segments = Vec::new();

        // Keep track of processed segments to avoid duplicate processing
        let mut processed_segment_ids = HashSet::new();

        // Process each intersection
        for intersection in &intersections {
            for &segment_id in &intersection.segment_ids {
                // Skip if this segment has already been processed
                if processed_segment_ids.contains(&segment_id) {
                    continue;
                }

                // Find the segment in the tile
                if let Some(segment) = tile.road_segments.iter().find(|s| s.id == segment_id) {
                    // Find all intersections for this segment
                    let segment_intersections: Vec<&IntersectionPoint> = intersections
                        .iter()
                        .filter(|int| int.segment_ids.contains(&segment_id))
                        .collect();

                    // Split the segment at all intersections
                    let split_result =
                        self.split_segment_at_points(segment, &segment_intersections)?;

                    // Add the split segments to the results
                    if split_result.new_segments.len() > 1 {
                        // The segment was split
                        new_segments.extend(split_result.new_segments);
                    } else {
                        // No split occurred, keep the original segment
                        updated_segments.push(segment.clone());
                    }

                    processed_segment_ids.insert(segment_id);
                }
            }
        }

        // Add remaining segments that weren't part of any intersection
        for segment in &tile.road_segments {
            if !processed_segment_ids.contains(&segment.id) {
                updated_segments.push(segment.clone());
            }
        }

        Ok((updated_segments, new_segments))
    }

    // Split a single segment at multiple intersection points
    fn split_segment_at_points(
        &self,
        segment: &WaySegment,
        intersections: &[&IntersectionPoint],
    ) -> Result<SplitSegment> {
        // Validate segment has at least two points
        if segment.coordinates.len() < 2 {
            return Err(anyhow!("Cannot split segment with less than 2 points"));
        }

        if intersections.is_empty() {
            // No intersections, return the original segment
            return Ok(SplitSegment {
                original_id: segment.id,
                new_segments: vec![segment.clone()],
            });
        }

        // Create a LineString from the segment coordinates
        let line_string = LineString::from(segment.coordinates.clone());

        // Find projection points for each intersection
        let mut split_points = Vec::new();
        for intersection in intersections {
            // Enhanced projection to handle multi-segment linestrings
            let projection_result = project_point_to_line_string(&intersection.point, &line_string);

            if let Some((idx, t, _distance)) = projection_result {
                trace!(
                    "Found projection for intersection at segment idx={}, t={}",
                    idx, t
                );
                split_points.push((idx, t, intersection.point));
            } else {
                warn!(
                    "Failed to project intersection point to segment {}",
                    segment.id
                );
            }
        }

        // Sort split points by their position along the line
        split_points.sort_by(|a, b| {
            if a.0 != b.0 {
                a.0.cmp(&b.0)
            } else {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        if split_points.is_empty() {
            // No valid split points found
            return Ok(SplitSegment {
                original_id: segment.id,
                new_segments: vec![segment.clone()],
            });
        }

        // Split the segment into multiple segments
        let mut new_segments = Vec::new();
        let mut current_coords = Vec::new();
        let mut current_nodes = Vec::new();

        // Start with the first coordinate and node
        current_coords.push(segment.coordinates[0]);
        if !segment.nodes.is_empty() {
            current_nodes.push(segment.nodes[0]);
        }

        let mut last_idx = 0;
        let mut last_t = 0.0;

        for (idx, t, point) in split_points {
            // Add coordinates between last point and current split point
            for i in (last_idx + 1)..=idx {
                current_coords.push(segment.coordinates[i]);
                if i < segment.nodes.len() {
                    current_nodes.push(segment.nodes[i]);
                }
            }

            // Add the exact intersection point
            if idx == last_idx && (t - last_t).abs() < 1e-10 {
                // This is the same point as the last one, skip
                continue;
            }

            // Add the intersection point
            let intersection_coord = Coord {
                x: point.x(),
                y: point.y(),
            };

            // Don't add if it's identical to the last coordinate
            if let Some(last_coord) = current_coords.last() {
                if (last_coord.x - intersection_coord.x).abs() < 1e-10
                    && (last_coord.y - intersection_coord.y).abs() < 1e-10
                {
                    // Skip this point as it's a duplicate
                } else {
                    current_coords.push(intersection_coord);
                }
            } else {
                current_coords.push(intersection_coord);
            }

            // Create a new segment if we have enough coordinates
            if current_coords.len() >= 2 {
                // Generate a new segment ID
                let new_id = self.next_segment_id.fetch_add(1, Ordering::SeqCst);

                // Create the new segment with same properties as the original
                let new_segment = WaySegment {
                    id: new_id,
                    osm_way_id: segment.osm_way_id,
                    nodes: current_nodes.clone(),
                    coordinates: current_coords.clone(),
                    is_oneway: segment.is_oneway,
                    highway_type: segment.highway_type.clone(),
                    max_speed: segment.max_speed,
                    connections: Vec::new(), // Will be updated later
                    name: segment.name.clone(),
                    metadata: segment.metadata.clone(),
                };

                new_segments.push(new_segment);
            }

            // Start the next segment with the intersection coordinate
            current_coords = vec![intersection_coord];

            // Important: Create a virtual node ID for this intersection if one doesn't exist
            // This helps maintain connectivity at intersection points
            let intersection_node_id = if idx < segment.nodes.len() {
                // Use existing node if this is at a node position
                segment.nodes[idx]
            } else {
                // Create a virtual node ID based on the intersection coordinates
                // This is a simple hash approach; you might want to use a more sophisticated method
                let x_int = (intersection_coord.x * 1_000_000.0) as u64;
                let y_int = (intersection_coord.y * 1_000_000.0) as u64;
                (x_int << 32) | y_int
            };

            current_nodes = vec![intersection_node_id];

            last_idx = idx;
            last_t = t;
        }

        // Add the remaining coordinates
        for i in (last_idx + 1)..segment.coordinates.len() {
            current_coords.push(segment.coordinates[i]);
            if i < segment.nodes.len() {
                current_nodes.push(segment.nodes[i]);
            }
        }

        // Create the final segment if we have enough coordinates
        if current_coords.len() >= 2 {
            // Generate a new segment ID
            let new_id = self.next_segment_id.fetch_add(1, Ordering::SeqCst);

            // Create the new segment
            let new_segment = WaySegment {
                id: new_id,
                osm_way_id: segment.osm_way_id,
                nodes: current_nodes,
                coordinates: current_coords,
                is_oneway: segment.is_oneway,
                highway_type: segment.highway_type.clone(),
                max_speed: segment.max_speed,
                connections: Vec::new(), // Will be updated later
                name: segment.name.clone(),
                metadata: segment.metadata.clone(),
            };

            new_segments.push(new_segment);
        }

        // If we couldn't create any valid segments, return the original
        if new_segments.is_empty() {
            return Ok(SplitSegment {
                original_id: segment.id,
                new_segments: vec![segment.clone()],
            });
        }

        Ok(SplitSegment {
            original_id: segment.id,
            new_segments,
        })
    }

    // Update connections between all segments after splitting
    fn update_segment_connections(
        &self,
        original_segments: Vec<WaySegment>,
        new_segments: Vec<WaySegment>,
    ) -> Result<Vec<WaySegment>> {
        let mut result = Vec::new();

        // Create lookup maps
        let mut id_map = HashMap::new();
        let mut segments_by_id = HashMap::new();

        // Map from original segment ID to all new segment IDs that replaced it
        for segment in &new_segments {
            let orig_id = segment.osm_way_id; // Use OSM way ID to identify original segments
            id_map
                .entry(orig_id)
                .or_insert_with(Vec::new)
                .push(segment.id);
            segments_by_id.insert(segment.id, segment.clone());
        }

        // Create a node-to-segment index for the new segments
        let mut node_to_segments: HashMap<u64, Vec<u64>> = HashMap::new();
        for segment in &new_segments {
            if !segment.nodes.is_empty() {
                // Add segment ID to the node's list of segments
                if let Some(first_node) = segment.nodes.first() {
                    node_to_segments
                        .entry(*first_node)
                        .or_default()
                        .push(segment.id);
                }

                if let Some(last_node) = segment.nodes.last() {
                    node_to_segments
                        .entry(*last_node)
                        .or_default()
                        .push(segment.id);
                }
            }
        }

        // Include original segments that weren't split
        for segment in &original_segments {
            segments_by_id.insert(segment.id, segment.clone());

            // Also add these to the node index
            if !segment.nodes.is_empty() {
                if let Some(first_node) = segment.nodes.first() {
                    node_to_segments
                        .entry(*first_node)
                        .or_default()
                        .push(segment.id);
                }

                if let Some(last_node) = segment.nodes.last() {
                    node_to_segments
                        .entry(*last_node)
                        .or_default()
                        .push(segment.id);
                }
            }
        }

        // Process original segments to update connections
        for mut segment in original_segments {
            // Keep original segment connections
            let mut new_connections = segment.connections.clone();

            // Replace connections to split segments
            for conn_id in &segment.connections {
                if let Some(new_ids) = id_map.get(conn_id) {
                    // Remove the original connection
                    new_connections.retain(|&id| id != *conn_id);

                    // Add connections to all new segments that replaced it
                    new_connections.extend(new_ids);
                }
            }

            // Add new connections based on shared nodes
            if !segment.nodes.is_empty() {
                let first_node = *segment.nodes.first().unwrap();
                let last_node = *segment.nodes.last().unwrap();

                // Connect to segments that share the first node
                if let Some(connected_segments) = node_to_segments.get(&first_node) {
                    for &other_id in connected_segments {
                        if other_id != segment.id && !new_connections.contains(&other_id) {
                            new_connections.push(other_id);
                        }
                    }
                }

                // Connect to segments that share the last node
                if let Some(connected_segments) = node_to_segments.get(&last_node) {
                    for &other_id in connected_segments {
                        if other_id != segment.id && !new_connections.contains(&other_id) {
                            new_connections.push(other_id);
                        }
                    }
                }
            }

            segment.connections = new_connections;
            result.push(segment);
        }

        // Process new segments to set up their connections
        for mut segment in new_segments {
            let mut connections = Vec::new();

            // For each split segment, connect to neighboring split segments
            let split_segments = id_map.get(&segment.osm_way_id).cloned().unwrap_or_default();

            // Connect to other segments from the same original segment
            for &other_id in &split_segments {
                if other_id != segment.id {
                    // Check if they share an endpoint - this works for linestrings with multiple points
                    if segments_share_endpoint(&segment, &segments_by_id[&other_id]) {
                        connections.push(other_id);
                    }
                }
            }

            // Connect based on shared nodes - critical for maintaining topology
            if !segment.nodes.is_empty() {
                let first_node = *segment.nodes.first().unwrap();
                let last_node = *segment.nodes.last().unwrap();

                // Connect to segments that share the first node
                if let Some(connected_segments) = node_to_segments.get(&first_node) {
                    for &other_id in connected_segments {
                        if other_id != segment.id && !connections.contains(&other_id) {
                            connections.push(other_id);
                        }
                    }
                }

                // Connect to segments that share the last node
                if let Some(connected_segments) = node_to_segments.get(&last_node) {
                    for &other_id in connected_segments {
                        if other_id != segment.id && !connections.contains(&other_id) {
                            connections.push(other_id);
                        }
                    }
                }
            }

            // Also connect based on spatial proximity for robustness
            if segment.coordinates.len() >= 2 {
                let start_point = segment.coordinates.first().unwrap();
                let end_point = segment.coordinates.last().unwrap();

                for (other_id, other_segment) in &segments_by_id {
                    if *other_id == segment.id || connections.contains(other_id) {
                        continue;
                    }

                    // Check if they are spatially close
                    if other_segment.coordinates.len() >= 2 {
                        let other_start = other_segment.coordinates.first().unwrap();
                        let other_end = other_segment.coordinates.last().unwrap();

                        if points_are_close(start_point, other_start)
                            || points_are_close(start_point, other_end)
                            || points_are_close(end_point, other_start)
                            || points_are_close(end_point, other_end)
                        {
                            connections.push(*other_id);
                        }
                    }
                }
            }

            segment.connections = connections;
            result.push(segment);
        }

        Ok(result)
    }

    // Write the updated tile back to disk
    fn write_updated_tile(
        &self,
        tile_id: &str,
        mut original_tile: TileIndex,
        updated_segments: Vec<WaySegment>,
    ) -> Result<()> {
        // Update the road segments in the tile
        original_tile.road_segments = updated_segments;

        // Update the segment index
        let mut segment_index = HashMap::new();
        for segment in &original_tile.road_segments {
            segment_index.insert(segment.id, tile_id.to_string());
        }
        original_tile.segment_index = segment_index;

        // Write the updated tile
        let tile_path = self.tile_directory.join(format!("{}.bin", tile_id));

        // Create a temporary file
        let temp_path = self.temp_directory.join(format!("{}.tmp", tile_id));

        // Write to temporary file first
        let file = File::create(&temp_path)?;
        let mut writer = BufWriter::new(file);
        let config = bincode::config::standard();
        bincode::serde::encode_into_std_write(&original_tile, &mut writer, config)?;
        writer.flush()?;

        // Rename the temporary file to the final location
        std::fs::rename(temp_path, tile_path)?;

        Ok(())
    }

    // Get layer value from segment metadata
    fn get_segment_layer(&self, segment: &WaySegment) -> LayerValue {
        segment
            .metadata
            .as_ref()
            .and_then(|m| m.get("layer").map(|l| l.parse::<LayerValue>().unwrap_or(0)))
            .unwrap_or(0)
    }

    // Check if segment is a bridge
    fn is_segment_bridge(&self, segment: &WaySegment) -> bool {
        segment
            .metadata
            .as_ref()
            .and_then(|m| m.get("bridge").map(|v| v == "yes"))
            .unwrap_or(false)
    }

    // Check if segment is a tunnel
    fn is_segment_tunnel(&self, segment: &WaySegment) -> bool {
        segment
            .metadata
            .as_ref()
            .and_then(|m| m.get("tunnel").map(|v| v == "yes"))
            .unwrap_or(false)
    }
}

// Check if a point is at an endpoint of a segment
fn is_point_at_endpoint(point: &Point<f64>, segment: &WaySegment) -> bool {
    if segment.coordinates.is_empty() {
        return false;
    }

    let first = &segment.coordinates[0];
    let last = &segment.coordinates[segment.coordinates.len() - 1];

    points_are_close(
        &Coord {
            x: point.x(),
            y: point.y(),
        },
        first,
    ) || points_are_close(
        &Coord {
            x: point.x(),
            y: point.y(),
        },
        last,
    )
}

// Check if two points are very close to each other
fn points_are_close(p1: &Coord<f64>, p2: &Coord<f64>) -> bool {
    const POINT_DISTANCE_THRESHOLD: f64 = 0.5; // 0.5 meters

    let point1 = Point::new(p1.x, p1.y);
    let point2 = Point::new(p2.x, p2.y);

    Haversine.distance(point1, point2) < POINT_DISTANCE_THRESHOLD
}

// Check if two segments share an endpoint
fn segments_share_endpoint(segment1: &WaySegment, segment2: &WaySegment) -> bool {
    if segment1.coordinates.is_empty() || segment2.coordinates.is_empty() {
        return false;
    }

    let s1_first = segment1.coordinates.first().unwrap();
    let s1_last = segment1.coordinates.last().unwrap();
    let s2_first = segment2.coordinates.first().unwrap();
    let s2_last = segment2.coordinates.last().unwrap();

    points_are_close(s1_first, s2_first)
        || points_are_close(s1_first, s2_last)
        || points_are_close(s1_last, s2_first)
        || points_are_close(s1_last, s2_last)
}

// Calculate the intersection point of two line segments
fn line_intersection(line1: &Line<f64>, line2: &Line<f64>) -> Option<Point<f64>> {
    // Convert to simple points for easier calculation
    let p1 = (line1.start.x, line1.start.y);
    let p2 = (line1.end.x, line1.end.y);
    let p3 = (line2.start.x, line2.start.y);
    let p4 = (line2.end.x, line2.end.y);

    // Calculate denominators
    let d = (p4.1 - p3.1) * (p2.0 - p1.0) - (p4.0 - p3.0) * (p2.1 - p1.1);

    // Check if lines are parallel (d close to 0)
    if d.abs() < 1e-10 {
        return None;
    }

    // Calculate intersection parameters
    let u_a = ((p4.0 - p3.0) * (p1.1 - p3.1) - (p4.1 - p3.1) * (p1.0 - p3.0)) / d;
    let u_b = ((p2.0 - p1.0) * (p1.1 - p3.1) - (p2.1 - p1.1) * (p1.0 - p3.0)) / d;

    // Check if intersection is within both line segments
    if (0.0..=1.0).contains(&u_a) && (0.0..=1.0).contains(&u_b) {
        // Calculate the intersection point
        let x = p1.0 + u_a * (p2.0 - p1.0);
        let y = p1.1 + u_a * (p2.1 - p1.1);

        Some(Point::new(x, y))
    } else {
        None
    }
}

// Project a point onto a LineString and return the segment index, parameter, and distance
// This enhanced version returns an Option with all values together for better error handling
fn project_point_to_line_string(
    point: &Point<f64>,
    line_string: &LineString<f64>,
) -> Option<(usize, f64, f64)> {
    // Validate the linestring has at least one segment
    if line_string.0.len() < 2 {
        return None;
    }

    let mut closest_idx = 0;
    let mut closest_t = 0.0;
    let mut closest_dist = f64::MAX;
    let mut found_valid_projection = false;

    // Process each segment in the linestring
    for (i, line) in line_string.lines().enumerate() {
        let p = (point.x(), point.y());
        let a = (line.start.x, line.start.y);
        let b = (line.end.x, line.end.y);

        // Vector from a to b
        let ab_x = b.0 - a.0;
        let ab_y = b.1 - a.1;

        // Vector from a to p
        let ap_x = p.0 - a.0;
        let ap_y = p.1 - a.1;

        // Calculate dot product
        let dot_product = ap_x * ab_x + ap_y * ab_y;

        // Calculate squared length of ab
        let ab_squared = ab_x * ab_x + ab_y * ab_y;

        // Calculate parameter t (projection)
        let t = if ab_squared < 1e-10 {
            0.0 // Handle degenerate case where a = b
        } else {
            dot_product / ab_squared
        };

        // Clamp t to line segment
        let t_clamped = t.max(0.0).min(1.0);

        // Calculate the projected point
        let proj_x = a.0 + t_clamped * ab_x;
        let proj_y = a.1 + t_clamped * ab_y;

        // Calculate distance to projected point
        let dx = p.0 - proj_x;
        let dy = p.1 - proj_y;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist < closest_dist {
            closest_dist = dist;
            closest_idx = i;
            closest_t = t_clamped;
            found_valid_projection = true;
        }
    }

    if found_valid_projection {
        Some((closest_idx, closest_t, closest_dist))
    } else {
        None
    }
}
