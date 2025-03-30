mod debug;
mod io;
mod mapmatcher;
mod osm_preprocessing;
mod route_matcher; // New route-based matching implementation
mod tile_loader;

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use chrono::DateTime;
use geo::{Coord, Point};
use http::Method;
use io::google_maps_local_timeline::LocationHistoryEntry;
use log::{debug, info, warn};
use mapmatcher::TileConfig;
use osm_preprocessing::RoadSegment;
use proto::chronotopia_server::{Chronotopia, ChronotopiaServer};
use proto::{LatLon, RequestParameters, RouteMatchTrace, Trip, Trips};
use route_matcher::{RouteMatchJob, RouteMatcher, RouteMatcherConfig, WindowTrace};
use serde_json::json;
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use tokio::sync::Mutex;
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
    info!("Starting Chronotopia with Route-Based Matcher");

    // Create configuration for route-based matcher
    let route_matcher_config = RouteMatcherConfig {
        osm_pbf_path: "/Users/wannes/Downloads/sudeste-latest.osm.pbf".to_string(),
        tile_cache_dir: "sample/tiles".to_string(),
        max_cached_tiles: 500,
        max_matching_distance: 150.0,
        tile_config: TileConfig {
            base_tile_size: 0.5,
            min_tile_density: 5000,
            max_split_depth: 2,
        },
        max_candidates_per_point: 10,
        max_tiles_per_depth: 100,
    };

    // Create route matcher
    let route_matcher = RouteMatcher::new(route_matcher_config.clone())?;

    // Check if preprocessing is needed
    if !Path::new(&route_matcher_config.tile_cache_dir).exists() {
        info!("Tile cache directory doesn't exist. Starting preprocessing...");
        route_matcher.preprocess()?;
    } else {
        info!(
            "Using existing tile cache from {}",
            route_matcher_config.tile_cache_dir
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
        route_matcher,
        last_job: Arc::new(Mutex::new(None)),
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
        .allow_methods([Method::GET, Method::POST])
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
    route_matcher: RouteMatcher,
    last_job: Arc<Mutex<Option<RouteMatchJob>>>,
}

impl ChronotopiaService {
    async fn build_trips(&mut self) -> Result<()> {
        info!(
            "Building trips from {} location history entries using Route-Based Matcher",
            self.location_history.len()
        );
        let mut processed_count = 0;
        let mut successful_matches = 0;
        let mut pending_job: Option<RouteMatchJob> = None;

        for entry in &self.location_history {
            if let LocationHistoryEntry::Timeline(tl) = entry {
                let trip: Trip = tl.into();
                if tl.timeline_path.len() < 2 {
                    debug!("Skipping trip with less than 2 points");
                    continue;
                }

                if let Some(start) = trip.start {
                    // Filter for trips after a certain timestamp
                    if start.seconds > 1742501500 {
                        info!("Processing trip starting at {}", start.seconds);

                        // Extract points and timestamps
                        let mut gps_points = Vec::new();
                        let mut timestamps = Vec::new();

                        for p in &trip.points {
                            if let Some(ts) = p.timestamp {
                                if let Some(dt) =
                                    DateTime::from_timestamp(ts.seconds, ts.nanos as u32)
                                {
                                    gps_points.push(Point::new(
                                        p.latlon.unwrap().lon as f64,
                                        p.latlon.unwrap().lat as f64,
                                    ));
                                    timestamps.push(dt);
                                }
                            }
                        }

                        if let Some(job) = pending_job.as_mut() {
                            // Check if the last timestamp of the pending job is close to the start of the current one
                            // If so, combine, if not, run the pending job
                            if *timestamps.first().unwrap() - job.timestamps.last().unwrap()
                                < chrono::Duration::hours(1)
                            {
                                debug!("Merging timeline entries");
                                job.gps_points.extend(gps_points);
                                job.timestamps.extend(timestamps);
                            } else {
                                // Perform map matching with debug way IDs
                                info!("Map matching trip with {} points", job.gps_points.len());

                                match self
                                    .route_matcher
                                    .match_trace(pending_job.as_mut().unwrap())
                                {
                                    Ok(result) => {
                                        successful_matches += 1;
                                        info!(
                                            "Map matching successful: {} segments matched ({}/{})",
                                            result.len(),
                                            successful_matches,
                                            processed_count + 1
                                        );

                                        // Save matched route to GeoJSON file
                                        let matched_geojson = road_segments_to_geojson(&result);
                                        let matched_output_path = format!(
                                            "sample/route_based_matched_{}.geojson",
                                            start.seconds
                                        );
                                        std::fs::write(
                                            matched_output_path.clone(),
                                            serde_json::to_string_pretty(&matched_geojson)?,
                                        )?;

                                        info!(
                                            "Route-based matched route saved to {}",
                                            matched_output_path
                                        );
                                    }
                                    Err(e) => {
                                        warn!("Route-based map matching failed: {}", e);
                                    }
                                }

                                if pending_job.is_some() {
                                    let mut last_job = self.last_job.lock().await;
                                    last_job.replace(pending_job.take().unwrap());
                                }

                                pending_job =
                                    Some(RouteMatchJob::new(gps_points, timestamps, None));
                                pending_job.as_mut().unwrap().activate_tracing();
                            }
                        } else {
                            pending_job = Some(RouteMatchJob::new(gps_points, timestamps, None));
                            pending_job.as_mut().unwrap().activate_tracing();
                        }

                        processed_count += 1;
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
            "properties": {
                "type": "matched_route",
                "color": "#FF0000",
                "weight": 4,
                "description": "Route-Based Matched Route"
            },
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

    async fn get_route_match_trace(
        &self,
        _request: Request<()>,
    ) -> Result<Response<RouteMatchTrace>, Status> {
        let last_routejob = self.last_job.lock().await;
        if let Some(rj) = last_routejob.as_ref() {
            let window_trace = rj.window_trace.borrow();
            let rmj: RouteMatchTrace = RouteMatchTrace {
                trip: None,
                window_traces: window_trace.iter().map(|v| v.into()).collect(),
            };
            Ok(Response::new(rmj))
        } else {
            Err(Status::not_found(""))
        }
    }
}

impl From<&RoadSegment> for proto::RoadSegment {
    fn from(value: &RoadSegment) -> Self {
        Self {
            id: value.id,
            coordinates: value
                .coordinates
                .iter()
                .map(|v| LatLon { lat: v.x, lon: v.y })
                .collect(),
            is_oneway: value.is_oneway,
            highway_type: value.highway_type.clone(),
            connections: value.connections.clone(),
            name: value.name.clone(),
        }
    }
}

impl From<&WindowTrace> for proto::WindowTrace {
    fn from(window_trace: &WindowTrace) -> Self {
        Self {
            start: window_trace.start as u32,
            end: window_trace.end as u32,
            segments: window_trace.segments.iter().map(|v| v.into()).collect(),
            bridge: window_trace.bridge,
        }
    }
}

pub mod proto {
    tonic::include_proto!("chronotopia");
}
