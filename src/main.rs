mod io;
mod mapmatcher;
mod osm_preprocessing;
mod tile_loader;

use std::path::Path;

use anyhow::Result;
use chrono::{DateTime, Utc};
use geo::{Coord, Point};
use http::Method;
use io::google_maps_local_timeline::LocationHistoryEntry;
use log::{debug, info, warn};
use mapmatcher::{MapMatcher, MapMatcherConfig, TileConfig};
use osm_preprocessing::RoadSegment;
use proto::chronotopia_server::{Chronotopia, ChronotopiaServer};
use proto::{RequestParameters, Trip, Trips};
use serde_json::json;
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use tokio::time::Duration;
use tonic::transport::Server;
use tonic::{Request, Response, Status};
use tonic_web::GrpcWebLayer;
use tower_http::cors::{Any, CorsLayer};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .format_target(false)
        .format_timestamp(None)
        .target(env_logger::Target::Stderr)
        .init();
    info!("Starting Chronotopia");

    /*
    At the equator:
    1 degree latitude = ~111 km
    1 degree longitude = ~111 km

    So 0.07 degrees x 0.07 degrees = 0.0049 square degrees
    Area: 0.07 * 111 km * 0.07 * 111 km = (7.77 km)^2 ≈ 60.4 km²

    Close to 50km.
    */

    // Create a more efficient configuration with loop detection enabled
    let mapmatcher_config = MapMatcherConfig {
        osm_pbf_path: "/Users/wannes/Downloads/sudeste-latest.osm.pbf".to_string(),
        tile_cache_dir: "sample/tiles".to_string(),
        max_cached_tiles: 500,        // Increased for better performance
        max_matching_distance: 100.0, // More realistic matching distance in meters
        tile_config: TileConfig {
            base_tile_size: 0.5,    // Base tile size (degrees)
            min_tile_density: 5000, // Target ~5000 roads per tile
            max_split_depth: 2,
        },
        use_turn_restrictions: true,
        max_candidates_per_point: 15, // More candidates for better selection
        distance_weight: 0.5,         // Reduced distance weight to favor continuity
        heading_weight: 0.3,
        speed_weight: 0.1,
        max_tiles_per_depth: 100, // Limit tiles per depth to prevent hanging
        loop_penalty_weight: 80.0, // Strong penalty for loops
        continuity_bonus_weight: 30.0, // Strong bonus for road continuity
    };

    // Create map matcher
    let mapmatcher = MapMatcher::new(mapmatcher_config.clone())?;

    // Check if preprocessing is needed
    if !Path::new(&mapmatcher_config.tile_cache_dir).exists() {
        info!("Tile cache directory doesn't exist. Starting preprocessing...");
        mapmatcher.preprocess()?;
    } else {
        info!(
            "Using existing tile cache from {}",
            mapmatcher_config.tile_cache_dir
        );
    }

    let mut file =
        File::open("/Users/wannes/Downloads/chronotopia/sample/location-history.json").await?;

    let mut contents = vec![];
    file.read_to_end(&mut contents).await?;

    let location_history: Vec<LocationHistoryEntry> = serde_json::from_slice(&contents)?;
    info!("Loaded {} location history entries", location_history.len());

    let addr = "[::1]:10000".parse()?;
    let mut chronotopia_service = ChronotopiaService {
        location_history,
        mapmatcher,
    };

    // Build trips with timeout protection
    match tokio::time::timeout(
        Duration::from_secs(120), // 2 minute timeout
        chronotopia_service.build_trips(),
    )
    .await
    {
        Ok(result) => match result {
            Ok(_) => info!("Successfully built trips"),
            Err(e) => warn!("Error building trips: {}", e),
        },
        Err(_) => {
            warn!("Timeout occurred while building trips - continuing with server startup");
        }
    }

    let svc = ChronotopiaServer::new(chronotopia_service);

    let cors = CorsLayer::new()
        // allow `GET` and `POST` when accessing the resource
        .allow_methods([Method::GET, Method::POST])
        // allow requests from any origin
        .allow_origin(Any)
        .allow_headers(Any);

    info!("Starting gRPC server at {}", addr);
    Server::builder()
        .accept_http1(true)
        .layer(cors)
        .layer(GrpcWebLayer::new())
        .add_service(svc)
        .serve(addr)
        .await
        .unwrap();

    Ok(())
}

struct ChronotopiaService {
    location_history: Vec<LocationHistoryEntry>,
    mapmatcher: MapMatcher,
}

impl ChronotopiaService {
    async fn build_trips(&mut self) -> Result<()> {
        info!(
            "Building trips from {} location history entries",
            self.location_history.len()
        );
        let mut processed_count = 0;
        let mut successful_matches = 0;

        for entry in &self.location_history {
            if let LocationHistoryEntry::Timeline(tl) = entry {
                let trip: Trip = tl.into();
                if tl.timeline_path.len() < 2 {
                    debug!("Skipping trip with less than 2 points");
                    continue;
                }

                if let Some(start) = trip.start {
                    // Filter for trips after a certain timestamp
                    if start.seconds > 1742601500 {
                        info!("Processing trip starting at {}", start.seconds);

                        // Extract points and timestamps
                        let mut gps_points = Vec::new();
                        let mut timestamps = Vec::new();

                        for p in &trip.points {
                            if let Some(ts) = p.timestamp {
                                if let Some(dt) =
                                    DateTime::from_timestamp(ts.seconds, ts.nanos as u32)
                                {
                                    gps_points.push(Point::new(p.lon as f64, p.lat as f64));
                                    timestamps.push(dt);
                                }
                            }
                        }

                        // Skip if insufficient points
                        if gps_points.len() < 5 {
                            debug!(
                                "Skipping trip with {} GPS points (need at least 5)",
                                gps_points.len()
                            );
                            continue;
                        }

                        // Perform map matching with robust error handling
                        info!("Map matching trip with {} points", gps_points.len());
                        match self.mapmatcher.match_trace(&gps_points, &timestamps) {
                            Ok(result) => {
                                successful_matches += 1;
                                info!(
                                    "Map matching successful: {} segments matched ({}/{})",
                                    result.len(),
                                    successful_matches,
                                    processed_count + 1
                                );

                                debug!(
                                    "Matched Linestring GeoJSON: {}",
                                    road_segments_to_geojson(&result)
                                );
                            }
                            Err(e) => {
                                warn!("Map matching failed: {}", e);

                                // Try to determine why it failed and provide more details
                                if e.to_string().contains("Segment")
                                    && e.to_string().contains("not found")
                                {
                                    warn!(
                                        "Missing segment issue. Check if OSM data is complete and properly preprocessed."
                                    );
                                } else if e.to_string().contains("path") {
                                    warn!(
                                        "Path finding issue. Consider increasing max_matching_distance or checking road connectivity."
                                    );
                                } else {
                                    warn!("Unknown error. Check logs for more details.");
                                }
                            }
                        }

                        processed_count += 1;

                        // Break after processing a few trips to avoid hanging
                        if processed_count >= 5 {
                            info!("Processed {} trips (limit reached)", processed_count);
                            break;
                        }
                    }
                }
            }
        }

        info!(
            "Trip building completed. Processed: {}, Successful: {}",
            processed_count, successful_matches
        );
        Ok(())
    }
}

/// Converts matched road segments into a GeoJSON LineString feature collection
pub fn road_segments_to_geojson(segments: &[RoadSegment]) -> serde_json::Value {
    let mut coordinates = Vec::new();

    // Helper to convert geo::Coord to GeoJSON format
    let coord_to_json = |c: &Coord<f64>| vec![c.x, c.y];

    for (i, segment) in segments.iter().enumerate() {
        // For first segment, add all points
        if i == 0 {
            coordinates.extend(segment.coordinates.iter().map(coord_to_json));
            continue;
        }

        // Check connection to previous segment
        let prev_last = coordinates.last().unwrap();
        let current_first = &segment.coordinates[0];

        // Compare with epsilon for floating point precision
        if (prev_last[0] - current_first.x).abs() > 1e-9
            || (prev_last[1] - current_first.y).abs() > 1e-9
        {
            // No connection - add all points
            coordinates.extend(segment.coordinates.iter().map(coord_to_json));
        } else {
            // Connected - skip first point to avoid duplicate
            coordinates.extend(segment.coordinates.iter().skip(1).map(coord_to_json));
        }
    }

    json!({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            }
        }]
    })
}

#[tonic::async_trait]
impl Chronotopia for ChronotopiaService {
    async fn get_trips(
        &self,
        request: Request<RequestParameters>,
    ) -> Result<Response<Trips>, Status> {
        let mut trips = vec![];
        let req = request.into_inner();
        for entry in &self.location_history {
            if let LocationHistoryEntry::Timeline(tl) = entry {
                let trip: Trip = tl.into();
                if let Some(from) = req.from {
                    if let Some(start) = trip.start {
                        if start.seconds < from.seconds {
                            continue;
                        }
                    }
                }
                trips.push(trip);
            }
        }
        Ok(Response::new(Trips { trips }))
    }
}

pub mod proto {
    tonic::include_proto!("chronotopia");
}
