#![feature(core_intrinsics)]

mod debug;
mod io;
mod mapmatcher;
mod osm_preprocessing;
mod route_matcher; // New route-based matching implementation
mod tile_loader;

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use chrono::DateTime;
use geo::{Coord, Distance, Haversine, Point};
use http::Method;
use io::google_maps_local_timeline::LocationHistoryEntry;
use log::{debug, info, warn};
use mapmatcher::TileConfig;
use osm_preprocessing::WaySegment;
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
        .filter_level(log::LevelFilter::Trace)
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
            base_tile_size: 1.0,
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
        route_matcher: Mutex::new(route_matcher),
        last_job_trace: Arc::new(Mutex::new(None)),
        processed_count: 0,
        successful_matches: 0,
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
    route_matcher: Mutex<RouteMatcher>,
    last_job_trace: Arc<Mutex<Option<RouteMatchTrace>>>,
    successful_matches: usize,
    processed_count: usize,
}

impl ChronotopiaService {
    async fn build_trips(&mut self) -> Result<()> {
        info!(
            "Building trips from {} location history entries using Route-Based Matcher",
            self.location_history.len()
        );
        let mut pending_trip: Option<Trip> = None;
        let location_history = self.location_history.clone();

        for entry in &location_history {
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

                        if let Some(ptrip) = pending_trip.as_mut() {
                            // Check if the last timestamp of the pending job is close to the start of the current one
                            // If so, combine, if not, run the pending job
                            if trip.points.first().unwrap().timestamp.unwrap().seconds
                                - ptrip.points.first().unwrap().timestamp.unwrap().seconds
                                < chrono::Duration::hours(1).num_seconds()
                            {
                                debug!("Merging timeline entries");
                                ptrip.points.extend(trip.points);
                                ptrip.stop = trip.stop;

                                continue;
                            } else {
                                let trip = pending_trip.take().unwrap();
                                self.build_trip(trip).await?;
                                self.processed_count += 1;
                            }
                        }

                        pending_trip = Some(trip);
                    }
                }
            }

            // Handle last job
            if let Some(trip) = pending_trip.take() {
                self.build_trip(trip).await?;
                self.processed_count += 1;
            }
        }

        info!(
            "Trip building completed. Processed: {}, Successful: {}",
            self.processed_count, self.successful_matches
        );
        Ok(())
    }

    async fn build_trip(&mut self, trip: Trip) -> Result<RouteMatchJob> {
        // Extract points and timestamps
        let mut gps_points = Vec::new();
        let mut timestamps = Vec::new();

        for p in &trip.points {
            if let Some(ts) = p.timestamp {
                if let Some(dt) = DateTime::from_timestamp(ts.seconds, ts.nanos as u32) {
                    gps_points.push(Point::new(
                        p.latlon.unwrap().lon as f64,
                        p.latlon.unwrap().lat as f64,
                    ));
                    timestamps.push(dt);
                }
            }
        }

        let mut job = RouteMatchJob::new(gps_points, timestamps, None);
        job.activate_tracing();

        // Perform map matching with debug way IDs
        info!("Map matching trip with {} points", job.gps_points.len());
        let mut route_matcher = self.route_matcher.lock().await;

        match route_matcher.match_trace(&mut job) {
            Ok(result) => {
                self.successful_matches += 1;
                info!(
                    "Map matching successful: {} segments matched ({}/{})",
                    result.len(),
                    self.successful_matches,
                    self.processed_count + 1
                );

                // Save matched route to GeoJSON file
                let matched_geojson = road_segments_to_geojson(&result);
                let matched_output_path = format!(
                    "sample/route_based_matched_{}.geojson",
                    job.timestamps.first().unwrap().timestamp()
                );
                std::fs::write(
                    matched_output_path.clone(),
                    serde_json::to_string_pretty(&matched_geojson)?,
                )?;

                info!("Route-based matched route saved to {}", matched_output_path);
            }
            Err(e) => {
                warn!("Route-based map matching failed: {}", e);
            }
        }

        let mut last_job_trace = self.last_job_trace.lock().await;

        let rmj: RouteMatchTrace = {
            let window_trace = job.window_trace.borrow();
            let point_candidates = job.point_candidates_geojson.borrow();
            RouteMatchTrace {
                trip: Some(trip),
                window_traces: window_trace.iter().map(|v| v.into()).collect(),
                point_candidates: point_candidates
                    .iter()
                    .map(|v| serde_json::to_string(v).unwrap().into())
                    .collect(),
            }
        };
        last_job_trace.replace(rmj);

        Ok(job)
    }
}

/// Converts matched road segments into a GeoJSON LineString feature collection
pub fn road_segments_to_geojson(segments: &[WaySegment]) -> serde_json::Value {
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
        let last_routejob = self.last_job_trace.lock().await;
        if let Some(rj) = last_routejob.as_ref() {
            Ok(Response::new(rj.clone()))
        } else {
            Err(Status::not_found(""))
        }
    }

    async fn osm_network_around_point(
        &self,
        request: Request<LatLon>,
    ) -> Result<Response<prost::alloc::string::String>, Status> {
        let point = request.into_inner();
        let mut features = Vec::new();

        // Add center point
        features.push(json!({
            "type": "Feature",
            "properties": {
            },
            "geometry": {
                "type": "Point",
                "coordinates": [point.lon, point.lat]
            }
        }));

        let radius_meters = 1000.0;
        let mut segments_in_radius = HashSet::new();
        let mut connections_count = HashMap::new();
        let mut route_matcher = self.route_matcher.lock().await;
        let segments: Vec<String> = route_matcher
            .tile_loader
            .loaded_tiles
            .keys()
            .cloned()
            .collect();

        for tile_id in segments {
            let tile = route_matcher
                .tile_loader
                .load_tile(&tile_id)
                .map_err(|e| Status::from_error(e.into()))?;

            for segment in &tile.road_segments {
                // Check if segment is within radius
                let segment_center = segment.centroid();
                let distance = Haversine.distance(Point::new(point.lon, point.lat), segment_center);

                if distance <= radius_meters {
                    segments_in_radius.insert(segment.id);

                    // Count connection references
                    for &conn_id in &segment.connections {
                        *connections_count.entry(conn_id).or_insert(0) += 1;
                    }
                }
            }
        }

        // Add all segments in radius
        for &seg_id in &segments_in_radius {
            if let Ok(segment) = route_matcher.tile_loader.get_segment(seg_id) {
                let coords: Vec<Vec<f64>> =
                    segment.coordinates.iter().map(|c| vec![c.x, c.y]).collect();

                let incoming_count = connections_count.get(&segment.id).cloned().unwrap_or(0);
                let outgoing_count = segment.connections.len();

                features.push(json!({
                        "type": "Feature",
                        "properties": {
                            "type": "network_segment",
                            "segment_id": segment.id,
                            "highway_type": segment.highway_type,
                            "is_oneway": segment.is_oneway,
                            "incoming_connections": incoming_count,
                            "outgoing_connections": outgoing_count,
                            "connections": segment.connections,
                            "description": format!("Segment: {}, In: {}, Out: {}", segment.id, incoming_count, outgoing_count)
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": coords
                        }
                    }));
            }
        }

        // Create GeoJSON
        let geojson = json!({
            "type": "FeatureCollection",
            "features": features
        });

        Ok(Response::new(serde_json::to_string(&geojson).unwrap()))
    }
}

impl From<&WaySegment> for proto::RoadSegment {
    fn from(value: &WaySegment) -> Self {
        Self {
            id: value.id,
            coordinates: value
                .coordinates
                .iter()
                .map(|v| LatLon { lat: v.y, lon: v.x })
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
