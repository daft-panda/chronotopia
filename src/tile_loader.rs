use anyhow::{Result, anyhow};
use log::{debug, info, trace, warn};
use ordered_float::OrderedFloat;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

use crate::osm_preprocessing::{TileGraph, TileIndex, WaySegment};
use crate::route_matcher::TileConfig;

// Optimization: Add cache statistics for monitoring
#[derive(Debug, Clone)]
struct CacheStats {
    hits: usize,
    misses: usize,
    evictions: usize,
    total_load_time_ms: u64,
}

/// Manages tile loading and caching with LRU eviction
#[derive(Debug)]
pub struct TileLoader {
    pub(crate) tile_directory: String,
    pub(crate) loaded_tiles: HashMap<String, TileIndex>,
    max_cached_tiles: usize,
    tile_config: TileConfig,
    segment_index: HashMap<u64, String>,
    /// Track last access time for LRU eviction
    tile_access_times: HashMap<String, SystemTime>,
    /// Ordered queue for quick access to LRU tiles
    lru_queue: VecDeque<String>,
    /// Cache statistics
    stats: CacheStats,
}

impl TileLoader {
    pub fn new(tile_directory: String, max_cached_tiles: usize, tile_config: TileConfig) -> Self {
        Self {
            tile_directory,
            loaded_tiles: HashMap::with_capacity(max_cached_tiles),
            max_cached_tiles,
            tile_config,
            segment_index: HashMap::new(),
            tile_access_times: HashMap::with_capacity(max_cached_tiles),
            lru_queue: VecDeque::with_capacity(max_cached_tiles),
            stats: CacheStats {
                hits: 0,
                misses: 0,
                evictions: 0,
                total_load_time_ms: 0,
            },
        }
    }

    /// Get tile file path
    fn tile_path(&self, tile_id: &str) -> PathBuf {
        Path::new(&self.tile_directory).join(format!("{}.bin", tile_id))
    }

    /// Load a tile and return a reference to it
    pub fn load_tile(&mut self, tile_id: &str) -> Result<&TileIndex> {
        // Check if already loaded
        if self.loaded_tiles.contains_key(tile_id) {
            // Cache hit
            self.stats.hits += 1;
            self.update_tile_access(tile_id);
            return self
                .loaded_tiles
                .get(tile_id)
                .ok_or_else(|| anyhow!("Tile disappeared from cache"));
        }

        // Cache miss
        self.stats.misses += 1;
        let load_start = Instant::now();
        trace!("Loading tile {} from disk", tile_id);

        // Evict if necessary
        if self.loaded_tiles.len() >= self.max_cached_tiles {
            self.evict_least_recently_used();
        }

        // Load tile from disk
        let tile_path = self.tile_path(tile_id);
        let tile_bytes = std::fs::read(&tile_path)
            .map_err(|e| anyhow!("Failed to read tile {}: {}", tile_id, e))?;

        let config = bincode::config::standard();
        let (tile_index, _): (TileIndex, _) =
            bincode::serde::decode_from_slice(&tile_bytes, config)
                .map_err(|e| anyhow!("Invalid tile {}: {}", tile_id, e))?;

        // Update segment index
        for (seg_id, tile_id_value) in &tile_index.segment_index {
            self.segment_index.insert(*seg_id, tile_id_value.clone());
        }

        // Insert tile and update LRU
        let tile_id_clone = tile_id.to_string();
        self.loaded_tiles.insert(tile_id_clone.clone(), tile_index);
        self.update_tile_access(&tile_id_clone);

        // Update stats
        let elapsed = load_start.elapsed();
        self.stats.total_load_time_ms += elapsed.as_millis() as u64;
        trace!("Tile {} loaded in {:?}", tile_id, elapsed);

        // Return reference to loaded tile
        self.loaded_tiles
            .get(tile_id)
            .ok_or_else(|| anyhow!("Tile {} disappeared from cache", tile_id))
    }

    /// Load all tiles that intersect with a bounding box
    pub fn load_tile_range(
        &mut self,
        bbox: geo::Rect<f64>,
        buffer: f64,
        max_tiles_per_depth: usize,
    ) -> Result<HashSet<String>> {
        let load_start = Instant::now();

        // Expand bbox by buffer
        let expanded_bbox = geo::Rect::new(
            geo::Coord {
                x: bbox.min().x - buffer,
                y: bbox.min().y - buffer,
            },
            geo::Coord {
                x: bbox.max().x + buffer,
                y: bbox.max().y + buffer,
            },
        );

        debug!(
            "Loading tiles in bbox: {:?} with buffer {:.5}°",
            expanded_bbox, buffer
        );

        let mut loaded_tiles = HashSet::new();

        // Start with most detailed levels first (higher depth = more detailed)
        for depth in (0..=self.tile_config.max_split_depth).rev() {
            let tiles_at_depth =
                self.load_tiles_at_depth(expanded_bbox, depth, max_tiles_per_depth)?;
            loaded_tiles.extend(tiles_at_depth);

            // If we found enough tiles at this level, don't go to less detailed levels
            if loaded_tiles.len() >= max_tiles_per_depth / 2 {
                debug!(
                    "Found {} tiles at depth {}, skipping less detailed levels",
                    loaded_tiles.len(),
                    depth
                );
                break;
            }
        }

        debug!(
            "Loaded {} tiles in {:?}",
            loaded_tiles.len(),
            load_start.elapsed()
        );
        Ok(loaded_tiles)
    }

    /// Load tiles at a specific depth level that intersect with a bbox
    fn load_tiles_at_depth(
        &mut self,
        bbox: geo::Rect<f64>,
        depth: u8,
        max_tiles: usize,
    ) -> Result<HashSet<String>> {
        // Calculate tile size at this depth
        let tile_size = self.tile_config.base_tile_size / 2f64.powi(depth.into());

        // Calculate tile coordinate ranges
        let min_x = (bbox.min().x / tile_size).floor() as i32;
        let max_x = (bbox.max().x / tile_size).ceil() as i32;
        let min_y = (bbox.min().y / tile_size).floor() as i32;
        let max_y = (bbox.max().y / tile_size).ceil() as i32;

        let width = (max_x - min_x + 1) as usize;
        let height = (max_y - min_y + 1) as usize;
        let total_tiles = width * height;

        debug!(
            "Depth {}: {} potential tiles ({}x{})",
            depth, total_tiles, width, height
        );

        let mut loaded = HashSet::new();

        // If too many potential tiles, focus on center area
        if total_tiles > max_tiles {
            let center_x = (min_x + max_x) / 2;
            let center_y = (min_y + max_y) / 2;

            // Generate spiral coordinates from center
            let coordinates = self.generate_spiral_coordinates(
                center_x, center_y, min_x, max_x, min_y, max_y, max_tiles,
            );

            // Load tiles in spiral order
            for (x, y) in coordinates {
                let tile_id = format!("{}_{}_{}", x, y, depth);

                // Check if file exists before trying to load
                if Path::new(&self.tile_path(&tile_id)).exists() {
                    self.load_tile(&tile_id)?;
                    loaded.insert(tile_id);

                    if loaded.len() >= max_tiles {
                        break;
                    }
                }
            }
        } else {
            // Load all tiles in the range
            for x in min_x..=max_x {
                for y in min_y..=max_y {
                    let tile_id = format!("{}_{}_{}", x, y, depth);

                    // Check if file exists before trying to load
                    if Path::new(&self.tile_path(&tile_id)).exists() {
                        self.load_tile(&tile_id)?;
                        loaded.insert(tile_id);
                    }
                }
            }
        }

        Ok(loaded)
    }

    /// Generate coordinates in a spiral pattern from center outward
    fn generate_spiral_coordinates(
        &self,
        center_x: i32,
        center_y: i32,
        min_x: i32,
        max_x: i32,
        min_y: i32,
        max_y: i32,
        max_points: usize,
    ) -> Vec<(i32, i32)> {
        let mut result = Vec::with_capacity(max_points);
        let mut visited = HashSet::new();

        // Add center point
        result.push((center_x, center_y));
        visited.insert((center_x, center_y));

        // Maximum radius needed
        let max_radius = (max_x - min_x).max(max_y - min_y) as usize;

        // Directions: right, down, left, up
        let directions = [(1, 0), (0, 1), (-1, 0), (0, -1)];

        let mut x = center_x;
        let mut y = center_y;
        let mut dir_idx = 0;
        let mut steps = 1;
        let mut step_count = 0;

        // Generate points in spiral pattern
        while result.len() < max_points && steps <= max_radius as i32 {
            // Move in current direction
            x += directions[dir_idx].0;
            y += directions[dir_idx].1;
            step_count += 1;

            // Check if valid and not visited
            if x >= min_x && x <= max_x && y >= min_y && y <= max_y && !visited.contains(&(x, y)) {
                result.push((x, y));
                visited.insert((x, y));
            }

            // Check if we need to change direction
            if step_count == steps {
                dir_idx = (dir_idx + 1) % 4;
                step_count = 0;

                // Increase steps after completing half a circle
                if dir_idx % 2 == 0 {
                    steps += 1;
                }
            }
        }

        result
    }

    /// Update access time for a tile
    fn update_tile_access(&mut self, tile_id: &str) {
        // Update access time
        let now = SystemTime::now();
        self.tile_access_times.insert(tile_id.to_string(), now);

        // Update LRU queue - remove if exists, then add to back
        if let Some(pos) = self.lru_queue.iter().position(|id| id == tile_id) {
            self.lru_queue.remove(pos);
        }
        self.lru_queue.push_back(tile_id.to_string());
    }

    /// Evict least recently used tile
    fn evict_least_recently_used(&mut self) {
        if self.loaded_tiles.is_empty() {
            return;
        }

        self.stats.evictions += 1;

        // Use LRU queue for fast eviction
        if let Some(oldest_tile) = self.lru_queue.pop_front() {
            trace!("Evicting LRU tile: {}", oldest_tile);
            self.loaded_tiles.remove(&oldest_tile);
            self.tile_access_times.remove(&oldest_tile);

            // Remove segment indices for this tile
            self.segment_index
                .retain(|_, tile_id| tile_id != &oldest_tile);
            return;
        }

        // Fallback: Find oldest access time
        if let Some((oldest_tile, _)) = self.tile_access_times.iter().min_by_key(|(_, time)| {
            OrderedFloat(
                time.duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64(),
            )
        }) {
            let oldest_tile = oldest_tile.clone();
            warn!(
                "LRU queue empty, falling back to time-based eviction for tile: {}",
                oldest_tile
            );

            self.loaded_tiles.remove(&oldest_tile);
            self.tile_access_times.remove(&oldest_tile);
            self.segment_index
                .retain(|_, tile_id| tile_id != &oldest_tile);
        }
    }

    /// Find which tile contains a segment
    pub fn find_segment_tile(&self, seg_id: u64) -> Result<String> {
        // First check segment index cache
        if let Some(tile_id) = self.segment_index.get(&seg_id) {
            return Ok(tile_id.clone());
        }

        // Check loaded tiles
        for (tile_id, tile) in &self.loaded_tiles {
            if tile.road_segments.iter().any(|s| s.id == seg_id) {
                return Ok(tile_id.clone());
            }
        }

        // Need to search on disk
        warn!(
            "Segment {} not found in loaded tiles, searching disk",
            seg_id
        );

        // More efficient disk search: check segment index first
        if let Ok(indices) = self.load_segment_indices() {
            if let Some(tile_id) = indices.get(&seg_id) {
                return Ok(tile_id.clone());
            }
        }

        // Full search as fallback
        self.search_segment_in_all_tiles(seg_id)
    }

    /// Load segment indices from a central index file if available
    fn load_segment_indices(&self) -> Result<HashMap<u64, String>> {
        let index_path = Path::new(&self.tile_directory).join("segment_index.bin");

        if index_path.exists() {
            let index_bytes = std::fs::read(&index_path)
                .map_err(|e| anyhow!("Failed to read segment index: {}", e))?;

            let config = bincode::config::standard();
            let (indices, _): (HashMap<u64, String>, _) =
                bincode::serde::decode_from_slice(&index_bytes, config)
                    .map_err(|e| anyhow!("Invalid segment index: {}", e))?;

            return Ok(indices);
        }

        Err(anyhow!("Segment index not available"))
    }

    /// Search for a segment in all tile files
    fn search_segment_in_all_tiles(&self, seg_id: u64) -> Result<String> {
        let search_start = Instant::now();

        // Read all tile files in directory
        let entries = std::fs::read_dir(&self.tile_directory)
            .map_err(|e| anyhow!("Failed to read tile directory: {}", e))?;

        for entry in entries {
            let entry = entry.map_err(|e| anyhow!("Failed to read directory entry: {}", e))?;
            let path = entry.path();

            // Check if it's a tile file
            if path.is_file() && path.extension().is_some_and(|ext| ext == "bin") {
                let file_name = path.file_stem().unwrap().to_string_lossy();

                // Skip segment index file
                if file_name == "segment_index" {
                    continue;
                }

                // Load and check the tile
                let tile_bytes = std::fs::read(&path)
                    .map_err(|e| anyhow!("Failed to read tile file {}: {}", file_name, e))?;

                let config = bincode::config::standard();
                let (tile_index, _): (TileIndex, _) =
                    bincode::serde::decode_from_slice(&tile_bytes, config)
                        .map_err(|e| anyhow!("Invalid tile data in {}: {}", file_name, e))?;

                // Check if segment is in this tile
                if tile_index.road_segments.iter().any(|s| s.id == seg_id) {
                    debug!(
                        "Found segment {} in disk tile {} after {:?}",
                        seg_id,
                        file_name,
                        search_start.elapsed()
                    );
                    return Ok(file_name.into_owned());
                }
            }
        }

        warn!(
            "Segment {} not found in any tile after extensive search ({:?})!",
            seg_id,
            search_start.elapsed()
        );
        Err(anyhow!(
            "Segment {} not found in any tile (searched all tiles on disk)",
            seg_id
        ))
    }

    /// Get a segment by ID, loading the containing tile if necessary
    pub fn get_segment(&mut self, seg_id: u64) -> Result<WaySegment> {
        // Find which tile contains this segment
        let tile_id = self.find_segment_tile(seg_id)?;

        // Load the tile
        let tile = self.load_tile(&tile_id)?;

        // Find the optimized segment
        let opt_segment = tile
            .road_segments
            .iter()
            .find(|s| s.id == seg_id)
            .ok_or_else(|| anyhow!("Segment {} missing from tile {}", seg_id, tile_id))?;

        // Convert the optimized segment back to a full WaySegment using the tile's metadata
        Ok(opt_segment.to_way_segment(&tile.metadata))
    }

    /// Get OSM way metadata using a bounding box to select relevant tiles
    pub fn get_way_metadata_from_bbox(
        &mut self,
        bbox: geo::Rect,
        way_ids: &[u64],
    ) -> Result<Vec<WaySegment>> {
        // Create a HashSet of way IDs for faster lookup
        let way_ids_set: HashSet<u64> = way_ids.iter().cloned().collect();

        // Load tiles that intersect with the bounding box
        info!("Loading tiles for bounding box: {:?}", bbox);
        let loaded_tiles = self.load_tile_range(bbox, 0.01, 100)?;
        info!("Loaded {} tiles for bounding box", loaded_tiles.len());

        // Collect metadata for all requested way IDs
        let mut metadata = Vec::new();
        let mut found_ways = HashSet::new();

        for tile_id in loaded_tiles {
            // Get all segments from this tile
            let segments = self.get_all_segments_from_tile(&tile_id)?;

            // Extract metadata for segments that match our way IDs
            for segment in &segments {
                let way_id = segment.osm_way_id;

                // Skip if not in our requested set or already processed
                if !way_ids_set.contains(&way_id) || found_ways.contains(&way_id) {
                    continue;
                }

                // Extract segment metadata
                let mut additional_tags = HashMap::new();

                if let Some(meta) = &segment.metadata {
                    for (key, value) in meta {
                        match key.as_str() {
                            // Skip keys that we extract specially
                            "name" | "highway" | "oneway" | "surface" | "maxspeed" | "lanes" => {}
                            // Store all other tags
                            _ => {
                                additional_tags.insert(key, value);
                            }
                        }
                    }
                }

                metadata.push(segment.clone());
                found_ways.insert(way_id);
            }
        }

        // Log any ways we couldn't find
        let missing_ways: Vec<u64> = way_ids_set.difference(&found_ways).cloned().collect();
        if !missing_ways.is_empty() {
            warn!(
                "Could not find metadata for {} OSM ways: {:?}",
                missing_ways.len(),
                missing_ways
            );
        }

        debug!(
            "Retrieved metadata for {} out of {} OSM ways",
            found_ways.len(),
            way_ids_set.len()
        );

        Ok(metadata)
    }

    // Add a method to access the tile's graph data
    pub fn get_tile_graph(&mut self, tile_id: &str) -> Result<Option<&TileGraph>> {
        let tile = self.load_tile(tile_id)?;
        Ok(tile.tile_graph.as_ref())
    }

    // Additional utility function to get all segments from a tile, converting them from optimized format
    pub fn get_all_segments_from_tile(&mut self, tile_id: &str) -> Result<Vec<WaySegment>> {
        let tile = self.load_tile(tile_id)?;

        Ok(tile
            .road_segments
            .iter()
            .map(|opt_segment| opt_segment.to_way_segment(&tile.metadata))
            .collect())
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> (usize, usize, usize, f64) {
        (
            self.stats.hits,
            self.stats.misses,
            self.stats.evictions,
            self.stats.total_load_time_ms as f64 / 1000.0,
        )
    }

    /// Reset cache statistics
    pub fn reset_stats(&mut self) {
        self.stats = CacheStats {
            hits: 0,
            misses: 0,
            evictions: 0,
            total_load_time_ms: 0,
        };
    }
}
