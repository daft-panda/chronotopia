use anyhow::Result;
use geo::Haversine;
use geo::{Coord, LineString, Point, algorithm::Distance};
use log::{debug, info};
use osmpbf::{Element, ElementReader};
use serde::{Deserialize, Serialize};
use std::intrinsics::breakpoint;
use std::{
    collections::{HashMap, HashSet},
    path::Path,
    time::Instant,
};

use crate::mapmatcher::{TileConfig, calculate_heading};

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

/// Tile metadata and index
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TileIndex {
    pub tile_id: String,
    pub bbox: geo_types::Rect<f64>,
    pub road_segments: Vec<WaySegment>,
    pub segment_index: HashMap<u64, String>,
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
