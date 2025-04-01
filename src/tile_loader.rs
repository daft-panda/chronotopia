use anyhow::{Result, anyhow};
use log::{debug, warn};
use ordered_float::OrderedFloat;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

use crate::mapmatcher::TileConfig;
use crate::osm_preprocessing::{TileIndex, WaySegment};

/// Manages tile loading and caching with LRU eviction
pub struct TileLoader {
    tile_directory: String,
    pub loaded_tiles: HashMap<String, TileIndex>,
    max_cached_tiles: usize,
    tile_config: TileConfig,
    segment_index: HashMap<u64, String>,
    /// Track last access time for LRU eviction
    tile_access_times: HashMap<String, SystemTime>,
    /// Ordered queue for quick access to LRU tiles
    lru_queue: VecDeque<String>,
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
        }
    }

    pub fn find_tiles_for_coordinate(
        &self,
        coordinate: geo_types::Point<f64>,
    ) -> Result<HashSet<String>> {
        let mut candidate_tiles = HashSet::new();

        for depth in 0..=self.tile_config.max_split_depth {
            let tile_size = self.tile_config.base_tile_size / (2.0f64.powi(depth as i32));

            let x_tile = (coordinate.x() / tile_size).floor() as i32;
            let y_tile = (coordinate.y() / tile_size).floor() as i32;

            let tile_id = format!("{}_{}_{}", x_tile, y_tile, depth);

            if self.check_tile_exists(&tile_id)? {
                candidate_tiles.insert(tile_id);

                // If we found a tile at more detailed level, we can stop
                if depth > 0 && !candidate_tiles.is_empty() {
                    break;
                }
            }
        }

        Ok(candidate_tiles)
    }

    fn tile_path(&self, tile_id: &str) -> PathBuf {
        Path::new(&self.tile_directory).join(format!("{}.bin", tile_id))
    }

    pub fn tile_bbox(tile_id: &str) -> Result<geo::Rect<f64>> {
        let parts: Vec<&str> = tile_id.split('_').collect();
        if parts.len() != 3 {
            return Err(anyhow!("Invalid tile ID format: {}", tile_id));
        }

        let x_tile = parts[0]
            .parse::<i32>()
            .map_err(|_| anyhow!("Invalid tile X coordinate: {}", parts[0]))?;
        let y_tile = parts[1]
            .parse::<i32>()
            .map_err(|_| anyhow!("Invalid tile Y coordinate: {}", parts[1]))?;
        let depth = parts[2]
            .parse::<i32>()
            .map_err(|_| anyhow!("Invalid tile depth: {}", parts[2]))?;

        let tile_size = 0.1 / 2f64.powi(depth);

        Ok(geo::Rect::new(
            geo::Coord {
                x: x_tile as f64 * tile_size,
                y: y_tile as f64 * tile_size,
            },
            geo::Coord {
                x: (x_tile as f64 + 1.0) * tile_size,
                y: (y_tile as f64 + 1.0) * tile_size,
            },
        ))
    }

    // Updated tile existence check with error handling
    fn check_tile_exists(&self, tile_id: &str) -> Result<bool> {
        let tile_path = self.tile_path(tile_id);
        Ok(tile_path.exists())
    }

    pub fn load_tile(&mut self, tile_id: &str) -> Result<&TileIndex> {
        // Check if already loaded (to avoid borrow checker issues, we'll handle this separately)
        let needs_loading = !self.loaded_tiles.contains_key(tile_id);

        if needs_loading {
            let load_start = Instant::now();
            debug!("Loading tile {} from disk", tile_id);

            // Phase 1: Check if we need to evict a tile
            if self.loaded_tiles.len() >= self.max_cached_tiles {
                self.evict_least_recently_used();
            }

            // Phase 2: Load tile from disk
            let tile_path = self.tile_path(tile_id);
            let tile_bytes = std::fs::read(&tile_path)
                .map_err(|e| anyhow!("Failed to read tile {}: {}", tile_id, e))?;

            let config = bincode::config::standard();
            let (tile_index, _): (TileIndex, _) =
                bincode::serde::decode_from_slice(&tile_bytes, config)
                    .map_err(|e| anyhow!("Invalid tile {}: {}", tile_id, e))?;

            // Phase 3: Update segment index
            for (seg_id, tile_id_value) in &tile_index.segment_index {
                self.segment_index.insert(*seg_id, tile_id_value.clone());
            }

            // Phase 4: Insert and update LRU
            let tile_id_clone = tile_id.to_string();
            self.loaded_tiles.insert(tile_id_clone.clone(), tile_index);
            self.update_tile_access(&tile_id_clone);

            debug!("Tile {} loaded in {:?}", tile_id, load_start.elapsed());
        } else {
            // Just update the access time for LRU
            self.update_tile_access(tile_id);
        }

        self.loaded_tiles
            .get(tile_id)
            .ok_or_else(|| anyhow!("Tile {} disappeared from cache", tile_id))
    }

    pub fn load_tile_range(
        &mut self,
        bbox: geo::Rect<f64>,
        buffer: f64,
        max_tiles_per_depth: usize,
    ) -> Result<HashSet<String>> {
        let load_start = Instant::now();
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
        let mut tiles = HashSet::new();

        debug!(
            "Loading tiles in bbox: {:?} with buffer {:.5}°",
            expanded_bbox, buffer
        );

        // Start with most detailed levels first
        for depth in (0..=self.tile_config.max_split_depth).rev() {
            let tile_size = self.tile_config.base_tile_size / 2f64.powi(depth.into());

            let min_x = (expanded_bbox.min().x / tile_size).floor() as i32;
            let max_x = (expanded_bbox.max().x / tile_size).ceil() as i32;
            let min_y = (expanded_bbox.min().y / tile_size).floor() as i32;
            let max_y = (expanded_bbox.max().y / tile_size).ceil() as i32;

            let width = (max_x - min_x + 1) as usize;
            let height = (max_y - min_y + 1) as usize;
            let total_tiles = width * height;

            debug!(
                "Depth {}: {} potential tiles ({}x{})",
                depth, total_tiles, width, height
            );

            if total_tiles > max_tiles_per_depth {
                debug!(
                    "Too many potential tiles at depth {}, limiting to center area",
                    depth
                );

                // If too many tiles at this level, focus on center area
                let center_x = (min_x + max_x) / 2;
                let center_y = (min_y + max_y) / 2;

                // Calculate radius to stay under max_tiles_per_depth
                let radius = ((max_tiles_per_depth as f64).sqrt() / 2.0).ceil() as i32;

                let bounded_min_x = (center_x - radius).max(min_x);
                let bounded_max_x = (center_x + radius).min(max_x);
                let bounded_min_y = (center_y - radius).max(min_y);
                let bounded_max_y = (center_y + radius).min(max_y);

                let tiles_at_depth = self.load_tiles_in_range(
                    bounded_min_x,
                    bounded_max_x,
                    bounded_min_y,
                    bounded_max_y,
                    depth,
                    max_tiles_per_depth,
                )?;

                tiles.extend(tiles_at_depth);
            } else {
                // Load all tiles at this level
                let tiles_at_depth = self.load_tiles_in_range(
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    depth,
                    max_tiles_per_depth,
                )?;

                tiles.extend(tiles_at_depth);
            }

            // If we found enough tiles at this level, don't go to less detailed levels
            if tiles.len() >= max_tiles_per_depth / 2 {
                debug!(
                    "Found {} tiles at depth {}, skipping less detailed levels",
                    tiles.len(),
                    depth
                );
                break;
            }
        }

        debug!("Loaded {} tiles in {:?}", tiles.len(), load_start.elapsed());
        Ok(tiles)
    }

    fn load_tiles_in_range(
        &mut self,
        min_x: i32,
        max_x: i32,
        min_y: i32,
        max_y: i32,
        depth: u8,
        max_tiles: usize,
    ) -> Result<HashSet<String>> {
        let mut tiles = HashSet::new();
        let mut loaded_count = 0;

        // Use spiral loading pattern from center outward
        let center_x = (min_x + max_x) / 2;
        let center_y = (min_y + max_y) / 2;

        // Generate coordinates in spiral pattern
        let mut spiral_coords = Vec::new();
        let width = max_x - min_x + 1;
        let height = max_y - min_y + 1;
        let max_radius = (width.max(height) / 2) as usize;

        // Add center point first
        spiral_coords.push((center_x, center_y));

        // Add points in spiraling outward pattern
        for r in 1..=max_radius {
            // Top edge (left to right)
            for x in (center_x - r as i32)..=(center_x + r as i32) {
                spiral_coords.push((x, center_y - r as i32));
            }
            // Right edge (top to bottom)
            for y in (center_y - r as i32 + 1)..=(center_y + r as i32) {
                spiral_coords.push((center_x + r as i32, y));
            }
            // Bottom edge (right to left)
            for x in (center_x - r as i32)..=(center_x + r as i32 - 1) {
                spiral_coords.push((x, center_y + r as i32));
            }
            // Left edge (bottom to top)
            for y in (center_y - r as i32 + 1)..=(center_y + r as i32 - 1) {
                spiral_coords.push((center_x - r as i32, y));
            }
        }

        // Filter coordinates to be within bounds and deduplicate
        let spiral_coords: Vec<_> = spiral_coords
            .into_iter()
            .filter(|&(x, y)| x >= min_x && x <= max_x && y >= min_y && y <= max_y)
            .collect();

        // Load tiles in spiral order
        for (x, y) in spiral_coords {
            if loaded_count >= max_tiles {
                break;
            }

            let tile_id = format!("{}_{}_{}", x, y, depth);

            if self.check_tile_exists(&tile_id)? {
                self.load_tile(&tile_id)?;
                tiles.insert(tile_id);
                loaded_count += 1;
            }
        }

        Ok(tiles)
    }

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

    fn evict_least_recently_used(&mut self) {
        if self.loaded_tiles.is_empty() {
            return;
        }

        // Option 1: Use LRU queue for fast eviction
        if let Some(oldest_tile) = self.lru_queue.pop_front() {
            debug!("Evicting LRU tile: {}", oldest_tile);
            self.loaded_tiles.remove(&oldest_tile);
            self.tile_access_times.remove(&oldest_tile);

            // Also clean up segments from this tile in segment_index
            self.segment_index
                .retain(|_, tile_id| tile_id != &oldest_tile);
            return;
        }

        // Option 2 (fallback): Find oldest access time
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

            // Also clean up segments from this tile in segment_index
            self.segment_index
                .retain(|_, tile_id| tile_id != &oldest_tile);
        }
    }

    pub fn find_segment_tile(&self, seg_id: u64) -> Result<String> {
        // First check segment index cache
        if let Some(tile_id) = self.segment_index.get(&seg_id) {
            return Ok(tile_id.clone());
        }

        // If not in index, search all loaded tiles
        debug!(
            "Segment {} not found in index, searching loaded tiles",
            seg_id
        );
        for (tile_id, tile) in &self.loaded_tiles {
            if tile.road_segments.iter().any(|s| s.id == seg_id) {
                debug!("Found segment {} in loaded tile {}", seg_id, tile_id);
                return Ok(tile_id.clone());
            }
        }

        // If not found in loaded tiles, search all tiles on disk
        warn!(
            "Segment {} not found in any loaded tile, searching all tiles on disk",
            seg_id
        );
        let search_start = std::time::Instant::now();

        // Check all possible tiles in the directory
        let entries = std::fs::read_dir(&self.tile_directory)
            .map_err(|e| anyhow!("Failed to read tile directory: {}", e))?;

        for entry in entries {
            let entry = entry.map_err(|e| anyhow!("Failed to read directory entry: {}", e))?;
            let path = entry.path();

            // Check if it's a bin file
            if path.is_file() && path.extension().is_some_and(|ext| ext == "bin") {
                let file_name = path.file_stem().unwrap().to_string_lossy();

                // Load the tile
                let tile_bytes = std::fs::read(&path)
                    .map_err(|e| anyhow!("Failed to read tile file {}: {}", file_name, e))?;

                let config = bincode::config::standard();
                let (tile_index, _): (TileIndex, _) =
                    bincode::serde::decode_from_slice(&tile_bytes, config)
                        .map_err(|e| anyhow!("Invalid tile data in {}: {}", file_name, e))?;

                // Check if segment exists in this tile
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

    pub fn get_segment(&mut self, seg_id: u64) -> Result<WaySegment> {
        let tile_id = self.find_segment_tile(seg_id)?;

        // Load tile and clone the segment we need
        let segment = {
            let tile = self.load_tile(&tile_id)?;
            tile.road_segments
                .iter()
                .find(|s| s.id == seg_id)
                .cloned()
                .ok_or_else(|| anyhow!("Segment {} missing from tile {}", seg_id, tile_id))?
        };

        Ok(segment)
    }
}
