mod debug;
mod io;
mod osm_preprocessing;
mod route_matcher; // New route-based matching implementation
mod tile_loader;

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use chrono::{DateTime, Utc};
use geo::{Coord, Distance, Haversine, Point};
use http::Method;
use io::google_maps_local_timeline::LocationHistoryEntry;
use log::{debug, info, warn};
use osm_preprocessing::WaySegment;
use proto::chronotopia_server::{Chronotopia, ChronotopiaServer};
use proto::{
    ConnectivityRequest, LatLon, MapMatchingStatus, ProcessedTrip, RequestParameters,
    RouteMatchTrace, Trip, TripSummary, Trips, WindowDebugRequest,
};
use route_matcher::{
    MatchedWaySegment, PathfindingDebugInfo, PathfindingResult, RouteMatchJob, RouteMatcher,
    RouteMatcherConfig, TileConfig, WindowTrace,
};
use serde_json::json;
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use tokio::sync::{Mutex, RwLock};
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
        osm_pbf_path: "../sudeste-latest.osm.pbf".to_string(),
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
        split_windows_on_failure: false,
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

    let mut file = File::open("sample/location-history.json").await?;

    let mut contents = vec![];
    file.read_to_end(&mut contents).await?;

    let location_history: Vec<LocationHistoryEntry> = serde_json::from_slice(&contents)?;
    info!("Loaded {} location history entries", location_history.len());

    let addr = "[::1]:10000".parse()?;
    let chronotopia_service = ChronotopiaService::new(location_history, route_matcher).await?;

    // Start trip processing in background
    let service_clone = chronotopia_service.clone();
    tokio::spawn(async move {
        match service_clone.process_all_trips().await {
            Ok(_) => info!("Successfully completed trip processing"),
            Err(e) => warn!("Error during trip processing: {}", e),
        }
    });

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

#[derive(Clone)]
struct ChronotopiaService {
    location_history: Arc<Vec<LocationHistoryEntry>>,
    route_matcher: Arc<Mutex<RouteMatcher>>,
    processed_trips: Arc<RwLock<HashMap<usize, ProcessedTrip>>>,
    processing_status: Arc<RwLock<HashMap<usize, MapMatchingStatus>>>,
}

impl ChronotopiaService {
    async fn new(
        location_history: Vec<LocationHistoryEntry>,
        route_matcher: RouteMatcher,
    ) -> Result<Self> {
        Ok(Self {
            location_history: Arc::new(location_history),
            route_matcher: Arc::new(Mutex::new(route_matcher)),
            processed_trips: Arc::new(RwLock::new(HashMap::new())),
            processing_status: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    async fn process_all_trips(&self) -> Result<()> {
        info!(
            "Processing all trips from {} location history entries using Route-Based Matcher",
            self.location_history.len()
        );

        let trips = self.extract_trips().await?;
        info!("Extracted {} trips for processing", trips.len());

        // Initialize all trips as pending
        {
            let mut status = self.processing_status.write().await;
            for (idx, _) in trips.iter().enumerate() {
                status.insert(idx, MapMatchingStatus::Pending);
            }
        }

        info!("Processing {} trips", trips.len());

        // Process each trip
        for (idx, trip) in trips.into_iter().enumerate() {
            // Update status to processing
            {
                let mut status = self.processing_status.write().await;
                status.insert(idx, MapMatchingStatus::Processing);
            }

            match self.process_trip(trip.clone(), idx).await {
                Ok(_) => {
                    let mut status = self.processing_status.write().await;
                    status.insert(idx, MapMatchingStatus::Completed);
                    info!("Successfully processed trip {}", idx);
                }
                Err(e) => {
                    let mut status = self.processing_status.write().await;
                    status.insert(idx, MapMatchingStatus::Failed);
                    warn!("Failed to process trip {}: {}", idx, e);
                }
            }
        }

        info!("All trips processed");
        Ok(())
    }

    async fn extract_trips(&self) -> Result<Vec<Trip>> {
        let mut trips = Vec::new();
        let mut pending_trip: Option<Trip> = None;

        for entry in self.location_history.iter() {
            if let LocationHistoryEntry::Timeline(tl) = entry {
                let trip: Trip = tl.into();
                if tl.timeline_path.len() < 2 {
                    continue;
                }

                // Include all trips with reasonable data
                if trip.points.len() >= 2 {
                    if let Some(ptrip) = pending_trip.as_mut() {
                        // Check if the trips should be merged (close in time)
                        if trip.points.first().unwrap().timestamp.unwrap().seconds
                            - ptrip.points.last().unwrap().timestamp.unwrap().seconds
                            < chrono::Duration::hours(1).num_seconds()
                        {
                            debug!("Merging timeline entries");
                            ptrip.points.extend(trip.points.clone());
                            ptrip.stop = trip.stop;
                            continue;
                        } else {
                            trips.push(pending_trip.take().unwrap());
                        }
                    }

                    pending_trip = Some(trip);
                }
            }
        }

        // Add the last pending trip if any
        if let Some(trip) = pending_trip {
            trips.push(trip);
        }

        Ok(trips)
    }

    async fn process_trip(&self, trip: Trip, trip_idx: usize) -> Result<()> {
        // Extract points and timestamps
        let mut gps_points = Vec::new();
        let mut timestamps = Vec::new();

        for p in &trip.points {
            if let Some(ts) = p.timestamp {
                if let Some(dt) = DateTime::from_timestamp(ts.seconds, ts.nanos as u32) {
                    gps_points.push(Point::new(
                        p.latlon.as_ref().unwrap().lon,
                        p.latlon.as_ref().unwrap().lat,
                    ));
                    timestamps.push(dt);
                }
            }
        }

        let mut job = RouteMatchJob::new(gps_points, timestamps, None);
        job.activate_tracing();

        // Perform map matching
        info!(
            "Map matching trip {} with {} points",
            trip_idx,
            job.gps_points.len()
        );
        let mut route_matcher = self.route_matcher.lock().await;

        match route_matcher.match_trace(&mut job) {
            Ok(result) => {
                info!(
                    "Map matching successful for trip {}: {} segments matched",
                    trip_idx,
                    result.len()
                );

                // Generate GeoJSON for the matched route
                let matched_geojson = road_segments_to_geojson(&result);

                // Create route match trace from job
                let route_match_trace = {
                    let window_trace = job.window_trace.borrow();
                    let point_candidates = job.point_candidates_geojson.borrow();
                    RouteMatchTrace {
                        trip: Some(trip.clone()),
                        window_traces: window_trace.iter().map(|v| v.into()).collect(),
                        point_candidates: point_candidates
                            .iter()
                            .map(|v| serde_json::to_string(v).unwrap())
                            .collect(),
                    }
                };

                // Store the processed trip
                let processed_trip = ProcessedTrip {
                    trip: Some(trip),
                    matched_segments: result.iter().map(|s| s.into()).collect(),
                    route_match_trace: Some(route_match_trace),
                    geojson: serde_json::to_string(&matched_geojson)?,
                };

                let mut trips = self.processed_trips.write().await;
                trips.insert(trip_idx, processed_trip);

                Ok(())
            }
            Err(e) => {
                warn!(
                    "Route-based map matching failed for trip {}: {}",
                    trip_idx, e
                );
                Err(e.into())
            }
        }
    }

    async fn get_trip_summary(&self, trip_idx: usize) -> Option<TripSummary> {
        let trips = self.processed_trips.read().await;
        let status = self.processing_status.read().await;

        if let Some(processed_trip) = trips.get(&trip_idx) {
            let trip = processed_trip.trip.as_ref()?;

            // Calculate some basic stats
            let start_time = trip.start.as_ref()?;
            let end_time = trip.stop.as_ref()?;
            let duration_seconds = end_time.seconds - start_time.seconds;
            let point_count = trip.points.len() as u32;

            return Some(TripSummary {
                index: trip_idx as u32,
                start: trip.start.clone(),
                end: trip.stop.clone(),
                point_count,
                status: MapMatchingStatus::Completed.into(),
                duration_seconds,
                distance_meters: 0.0, // This would require calculating from the matched segments
                start_point: trip.points.first().and_then(|p| p.latlon.clone()),
                end_point: trip.points.last().and_then(|p| p.latlon.clone()),
                matched_segment_count: processed_trip.matched_segments.len() as u32,
            });
        } else if status.contains_key(&trip_idx) {
            // Trip exists but hasn't been processed yet
            return Some(TripSummary {
                status: (*status.get(&trip_idx).unwrap()).into(),
                ..Default::default()
            });
        }

        None
    }

    async fn load_tiles_for_segments(
        &self,
        route_matcher: &mut RouteMatcher,
        from_segment_id: u64,
        to_segment_id: u64,
    ) -> Result<(), Status> {
        // First try to find the tile containing segment1
        let tile_id1 = match route_matcher.tile_loader.find_segment_tile(from_segment_id) {
            Ok(id) => id,
            Err(_) => {
                return Err(Status::not_found(format!(
                    "Segment {} not found",
                    from_segment_id
                )));
            }
        };

        // Then try to find the tile containing segment2
        let tile_id2 = match route_matcher.tile_loader.find_segment_tile(to_segment_id) {
            Ok(id) => id,
            Err(_) => {
                return Err(Status::not_found(format!(
                    "Segment {} not found",
                    to_segment_id
                )));
            }
        };

        // Load the two tiles
        let _ = route_matcher.tile_loader.load_tile(&tile_id1).unwrap();
        let _ = route_matcher.tile_loader.load_tile(&tile_id2).unwrap();

        // Load segments
        let segment1 = match route_matcher.tile_loader.get_segment(from_segment_id) {
            Ok(s) => s,
            Err(_) => {
                return Err(Status::not_found(format!(
                    "Failed to load segment {}",
                    from_segment_id
                )));
            }
        };

        let segment2 = match route_matcher.tile_loader.get_segment(to_segment_id) {
            Ok(s) => s,
            Err(_) => {
                return Err(Status::not_found(format!(
                    "Failed to load segment {}",
                    to_segment_id
                )));
            }
        };

        // Calculate a bounding box that includes both segments with a buffer
        let min_x = segment1
            .coordinates
            .iter()
            .map(|c| c.x)
            .chain(segment2.coordinates.iter().map(|c| c.x))
            .fold(f64::MAX, |acc, x| acc.min(x));

        let min_y = segment1
            .coordinates
            .iter()
            .map(|c| c.y)
            .chain(segment2.coordinates.iter().map(|c| c.y))
            .fold(f64::MAX, |acc, y| acc.min(y));

        let max_x = segment1
            .coordinates
            .iter()
            .map(|c| c.x)
            .chain(segment2.coordinates.iter().map(|c| c.x))
            .fold(f64::MIN, |acc, x| acc.max(x));

        let max_y = segment1
            .coordinates
            .iter()
            .map(|c| c.y)
            .chain(segment2.coordinates.iter().map(|c| c.y))
            .fold(f64::MIN, |acc, y| acc.max(y));

        // Add a buffer (approximately 1km in degrees)
        let buffer = 0.01;
        let bbox = geo::Rect::new(
            geo::Coord {
                x: min_x - buffer,
                y: min_y - buffer,
            },
            geo::Coord {
                x: max_x + buffer,
                y: max_y + buffer,
            },
        );

        // Load all tiles in the bounding box
        let loaded_tiles = route_matcher
            .tile_loader
            .load_tile_range(
                bbox, buffer, 200, // Allow many tiles to be loaded
            )
            .map_err(|e| Status::internal(format!("Failed to load tiles: {}", e)))?;

        info!("Loaded {} tiles for segment routing", loaded_tiles.len());

        Ok(())
    }
}

#[tonic::async_trait]
impl Chronotopia for ChronotopiaService {
    async fn get_trips(
        &self,
        request: Request<RequestParameters>,
    ) -> Result<Response<Trips>, Status> {
        let req = request.into_inner();
        let mut trips = Vec::new();

        let all_trips = self
            .extract_trips()
            .await
            .map_err(|e| Status::internal(format!("Failed to extract trips: {}", e)))?;

        for trip in all_trips {
            if let Some(from) = req.from.as_ref() {
                if let Some(start) = trip.start.as_ref() {
                    if start.seconds < from.seconds {
                        continue;
                    }
                }
            }

            trips.push(trip);
        }

        Ok(Response::new(Trips { trips }))
    }

    async fn get_route_match_trace(
        &self,
        request: Request<proto::GetRouteMatchTraceRequest>,
    ) -> Result<Response<RouteMatchTrace>, Status> {
        let req = request.into_inner();
        let trip_idx = req.trip_index as usize;

        let trips = self.processed_trips.read().await;
        if let Some(processed_trip) = trips.get(&trip_idx) {
            if let Some(trace) = &processed_trip.route_match_trace {
                return Ok(Response::new(trace.clone()));
            }
        }

        Err(Status::not_found(format!(
            "Route match trace for trip {} not found",
            trip_idx
        )))
    }

    async fn get_trip_summaries(
        &self,
        _request: Request<()>,
    ) -> Result<Response<proto::TripSummaries>, Status> {
        let status = self.processing_status.read().await;
        let mut summaries = Vec::new();

        for idx in status.keys() {
            if let Some(summary) = self.get_trip_summary(*idx).await {
                summaries.push(summary);
            }
        }

        // Sort by start time
        summaries.sort_by_key(|s| s.start.as_ref().map(|t| t.seconds).unwrap_or(0));

        Ok(Response::new(proto::TripSummaries { summaries }))
    }

    async fn get_trip_geo_json(
        &self,
        request: Request<proto::TripRequest>,
    ) -> Result<Response<prost::alloc::string::String>, Status> {
        let trip_idx = request.into_inner().trip_index as usize;

        let trips = self.processed_trips.read().await;
        if let Some(processed_trip) = trips.get(&trip_idx) {
            return Ok(Response::new(processed_trip.geojson.clone()));
        }

        Err(Status::not_found(format!(
            "GeoJSON for trip {} not found",
            trip_idx
        )))
    }

    async fn get_processed_trip(
        &self,
        request: Request<proto::TripRequest>,
    ) -> Result<Response<ProcessedTrip>, Status> {
        let trip_idx = request.into_inner().trip_index as usize;

        let trips = self.processed_trips.read().await;
        if let Some(processed_trip) = trips.get(&trip_idx) {
            return Ok(Response::new(processed_trip.clone()));
        }

        Err(Status::not_found(format!(
            "Processed trip {} not found",
            trip_idx
        )))
    }

    async fn debug_window_path_finding(
        &self,
        request: Request<WindowDebugRequest>,
    ) -> Result<Response<proto::PathfindingDebugInfo>, Status> {
        let req = request.into_inner();
        let trip_idx = req.trip_index as usize;

        // Get the processed trip data
        let processed_trips = self.processed_trips.read().await;
        let processed_trip = processed_trips.get(&trip_idx).ok_or_else(|| {
            Status::not_found(format!("Trip {} not found or not processed yet", trip_idx))
        })?;

        if let (Some(from_segment_id), Some(to_segment_id)) =
            (req.from_segment_id, req.to_segment_id)
        {
            info!(
                "Debugging direct segment pathfinding between segments {} and {}",
                from_segment_id, to_segment_id
            );

            let mut route_matcher = self.route_matcher.lock().await;

            // Create a temporary job for debugging
            let mut job = RouteMatchJob::new(
                vec![Point::new(0.0, 0.0), Point::new(0.0, 0.0)], // Dummy points
                vec![Utc::now(), Utc::now()],                     // Dummy timestamps
                None,
            );
            job.activate_tracing();

            // Load surrounding tiles to make sure we have the necessary data
            self.load_tiles_for_segments(&mut route_matcher, from_segment_id, to_segment_id)
                .await?;

            // Copy graph and segment map from route matcher
            let loaded_tiles = route_matcher
                .tile_loader
                .loaded_tiles
                .keys()
                .cloned()
                .collect();

            let graph = route_matcher.build_road_network(&loaded_tiles).unwrap();
            job.graph.replace(Some(graph.0));
            job.segment_map.replace(graph.1);

            // Generate detailed debug info for direct segment routing
            let debug_info =
                route_matcher.debug_direct_segment_routing(&job, from_segment_id, to_segment_id);

            return Ok(Response::new(debug_info.into()));
        }

        let window_index = req.window_index as usize;
        let start_point = req.start_point as usize;
        let end_point = req.end_point as usize;

        info!(
            "Debugging window pathfinding for trip {}, window {}, points {}-{}",
            trip_idx, window_index, start_point, end_point
        );

        // Get relevant constraints for this window
        let mut route_matcher = self.route_matcher.lock().await;

        // Get route match trace for this trip
        let trace = match &processed_trip.route_match_trace {
            Some(trace) => trace,
            None => {
                return Err(Status::not_found(
                    "Route match trace not found for this trip",
                ));
            }
        };

        // Find the window we're looking for
        let window = trace.window_traces.get(window_index).ok_or_else(|| {
            Status::not_found(format!(
                "Window {} not found in trip {}",
                window_index, trip_idx
            ))
        })?;

        // Extract constraints from the window
        let constraints: Vec<(usize, u64)> = window
            .constraints
            .iter()
            .map(|c| (c.point_idx as usize, c.segment_id))
            .collect();

        // Create a temporary job for debugging
        let points: Vec<Point<f64>> = trace
            .trip
            .as_ref()
            .ok_or_else(|| Status::internal("Trip data not found in route match trace"))?
            .points
            .iter()
            .filter_map(|p| {
                if let Some(latlon) = &p.latlon {
                    Some(Point::new(latlon.lon, latlon.lat))
                } else {
                    None
                }
            })
            .collect();

        let timestamps: Vec<DateTime<Utc>> = trace
            .trip
            .as_ref()
            .unwrap()
            .points
            .iter()
            .filter_map(|p| {
                if let Some(ts) = &p.timestamp {
                    DateTime::from_timestamp(ts.seconds, ts.nanos as u32)
                } else {
                    None
                }
            })
            .collect();

        let mut job = RouteMatchJob::new(points, timestamps, None);
        job.activate_tracing();

        // Copy graph and segment map from route matcher
        let loaded_tiles = route_matcher
            .tile_loader
            .loaded_tiles
            .keys()
            .cloned()
            .collect();
        let graph = route_matcher.build_road_network(&loaded_tiles).unwrap();
        job.graph.replace(Some(graph.0));
        job.segment_map.replace(graph.1);

        // Find candidate segments for all points
        route_matcher
            .find_all_candidate_segments(&job, &loaded_tiles)
            .map_err(|e| Status::internal(format!("Failed to find candidate segments: {}", e)))?;

        // Generate detailed debug info for this window
        let debug_info = route_matcher.debug_constrained_window_failure(
            &job,
            start_point,
            end_point,
            &constraints,
        );

        Ok(Response::new(debug_info.into()))
    }

    /// Analyze segment connectivity
    async fn analyze_segment_connectivity(
        &self,
        request: Request<ConnectivityRequest>,
    ) -> Result<Response<prost::alloc::string::String>, Status> {
        let req = request.into_inner();
        let trip_idx = req.trip_index as usize;

        // Get the processed trip data
        let processed_trips = self.processed_trips.read().await;
        let processed_trip = processed_trips.get(&trip_idx).ok_or_else(|| {
            Status::not_found(format!("Trip {} not found or not processed yet", trip_idx))
        })?;

        // Get route match trace for this trip
        let trace = match &processed_trip.route_match_trace {
            Some(trace) => trace,
            None => {
                return Err(Status::not_found(
                    "Route match trace not found for this trip",
                ));
            }
        };

        if let (Some(start_point_index), Some(end_point_index)) =
            (req.start_point_index, req.end_point_index)
        {
            let start_point = start_point_index as usize;
            let end_point = end_point_index as usize;

            info!(
                "Analyzing segment connectivity for trip {} between points {}-{}",
                trip_idx, start_point, end_point
            );

            // Create a temporary job for connectivity analysis
            let mut route_matcher = self.route_matcher.lock().await;

            // Create a job using the trip data
            let points: Vec<Point<f64>> = trace
                .trip
                .as_ref()
                .ok_or_else(|| Status::internal("Trip data not found in route match trace"))?
                .points
                .iter()
                .filter_map(|p| {
                    if let Some(latlon) = &p.latlon {
                        Some(Point::new(latlon.lon, latlon.lat))
                    } else {
                        None
                    }
                })
                .collect();

            let timestamps: Vec<DateTime<Utc>> = trace
                .trip
                .as_ref()
                .unwrap()
                .points
                .iter()
                .filter_map(|p| {
                    if let Some(ts) = &p.timestamp {
                        DateTime::from_timestamp(ts.seconds, ts.nanos as u32)
                    } else {
                        None
                    }
                })
                .collect();

            let mut job = RouteMatchJob::new(points, timestamps, None);

            // Copy graph and segment map from route matcher
            let loaded_tiles = route_matcher
                .tile_loader
                .loaded_tiles
                .keys()
                .cloned()
                .collect();

            // Copy graph and segment map from route matcher
            let graph = route_matcher.build_road_network(&loaded_tiles).unwrap();
            job.graph.replace(Some(graph.0));
            job.segment_map.replace(graph.1);

            // Find candidate segments for all points
            route_matcher
                .find_all_candidate_segments(&job, &loaded_tiles)
                .map_err(|e| {
                    Status::internal(format!("Failed to find candidate segments: {}", e))
                })?;

            // Generate connectivity visualization
            let geojson = route_matcher
                .analyze_segment_connectivity(&job, start_point, end_point)
                .map_err(|e| Status::internal(format!("Failed to analyze connectivity: {}", e)))?;

            Ok(Response::new(serde_json::to_string(&geojson).unwrap()))
        } else if let (Some(from_segment_id), Some(to_segment_id)) =
            (req.from_segment_id, req.to_segment_id)
        {
            info!(
                "Analyzing segment connectivity for trip {} between segments {}-{}",
                trip_idx, from_segment_id, to_segment_id
            );

            let mut route_matcher = self.route_matcher.lock().await;

            // Create a temporary job for debugging
            let mut job = RouteMatchJob::new(
                vec![Point::new(0.0, 0.0), Point::new(0.0, 0.0)], // Dummy points
                vec![Utc::now(), Utc::now()],                     // Dummy timestamps
                None,
            );
            job.activate_tracing();

            // Load surrounding tiles to make sure we have the necessary data
            self.load_tiles_for_segments(&mut route_matcher, from_segment_id, to_segment_id)
                .await?;

            // Copy graph and segment map from route matcher
            let loaded_tiles = route_matcher
                .tile_loader
                .loaded_tiles
                .keys()
                .cloned()
                .collect();

            // Copy graph and segment map from route matcher
            let graph = route_matcher.build_road_network(&loaded_tiles).unwrap();
            job.graph.replace(Some(graph.0));
            job.segment_map.replace(graph.1);

            // Generate connectivity visualization
            let geojson = route_matcher
                .analyze_segment_connectivity(
                    &job,
                    from_segment_id as usize,
                    to_segment_id as usize,
                )
                .map_err(|e| Status::internal(format!("Failed to analyze connectivity: {}", e)))?;

            Ok(Response::new(serde_json::to_string(&geojson).unwrap()))
        } else {
            Err(Status::invalid_argument("Invalid request parameters"))
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
                "type": "center_point",
                "radius": 2000.0
            },
            "geometry": {
                "type": "Point",
                "coordinates": [point.lon, point.lat]
            }
        }));

        let radius_meters = 2000.0;
        let mut segments_in_radius = HashSet::new();
        let mut connections_count = HashMap::new();
        let mut route_matcher = self.route_matcher.lock().await;
        let segments: Vec<String> = route_matcher
            .tile_loader
            .loaded_tiles
            .keys()
            .cloned()
            .collect();

        // Collect segments in radius
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

        // Group segments by layer for better visualization
        let mut layers = HashSet::new();
        let mut segment_by_layer = HashMap::new();

        // Collect all segments and organize by layer
        for &seg_id in &segments_in_radius {
            if let Ok(segment) = route_matcher.tile_loader.get_segment(seg_id) {
                let layer = segment
                    .metadata
                    .as_ref()
                    .and_then(|m| m.get("layer").map(|l| l.parse::<i8>().unwrap_or(0)))
                    .unwrap_or(0);

                layers.insert(layer);
                segment_by_layer
                    .entry(layer)
                    .or_insert_with(Vec::new)
                    .push(segment);
            }
        }

        // Add segments grouped by layer
        for layer in layers {
            if let Some(segments) = segment_by_layer.get(&layer) {
                for segment in segments {
                    let coords: Vec<Vec<f64>> =
                        segment.coordinates.iter().map(|c| vec![c.x, c.y]).collect();

                    let incoming_count = connections_count.get(&segment.id).cloned().unwrap_or(0);
                    let outgoing_count = segment.connections.len();

                    // Extract additional metadata for visualization
                    let is_bridge = segment
                        .metadata
                        .as_ref()
                        .and_then(|m| m.get("bridge").map(|v| v == "yes"))
                        .unwrap_or(false);

                    let is_tunnel = segment
                        .metadata
                        .as_ref()
                        .and_then(|m| m.get("tunnel").map(|v| v == "yes"))
                        .unwrap_or(false);

                    // Color based on highway type and layer
                    let color = match (segment.highway_type.as_str(), layer) {
                        ("motorway", _) => "#ff0000",
                        ("trunk", _) => "#ff6600",
                        ("primary", _) if is_bridge => "#ffcc00",
                        ("primary", _) => "#ffaa00",
                        ("secondary", _) => "#aaff00",
                        ("tertiary", _) => "#00aaff",
                        (_, l) if l > 0 => "#aa00ff", // Positive layer (bridge)
                        (_, l) if l < 0 => "#006600", // Negative layer (tunnel)
                        _ => "#aaaaaa",
                    };

                    // Line width based on highway type
                    let weight = match segment.highway_type.as_str() {
                        "motorway" => 5,
                        "trunk" => 4,
                        "primary" => 4,
                        "secondary" => 3,
                        "tertiary" => 3,
                        _ => 2,
                    };

                    // Create feature with all the metadata
                    features.push(json!({
                        "type": "Feature",
                        "properties": {
                            "type": "network_segment",
                            "segment_id": segment.id,
                            "osm_way_id": segment.osm_way_id, // Include OSM way ID
                            "highway_type": segment.highway_type,
                            "is_oneway": segment.is_oneway,
                            "incoming_connections": incoming_count,
                            "outgoing_connections": outgoing_count,
                            "connections": segment.connections,
                            "layer": layer,
                            "is_bridge": is_bridge,
                            "is_tunnel": is_tunnel,

                            "color": color,
                            "weight": weight,
                            "opacity": 0.7 ,
                            "description": format!(
                                "ID: {} (OSM: {}), Type: {}, Name: {}, Layer: {}{}{}, In: {}, Out: {}", 
                                segment.id,
                                segment.osm_way_id,
                                segment.highway_type,
                                segment.name.as_deref().unwrap_or("Unnamed"),
                                layer,
                                if is_bridge { ", Bridge" } else { "" },
                                if is_tunnel { ", Tunnel" } else { "" },
                                incoming_count,
                                outgoing_count
                            )
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": coords
                        }
                    }));
                }
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

// Generate a GeoJSON based of the matched route
pub fn road_segments_to_geojson(segments: &[MatchedWaySegment]) -> serde_json::Value {
    let mut features = Vec::new();

    // Helper to convert geo::Coord to GeoJSON format
    let coord_to_json = |c: &Coord<f64>| vec![c.x, c.y];

    // Create the combined route - a simple concatenation of all segment coordinates
    let mut combined_coords = Vec::new();

    // Collect all coordinates from all segments in sequence
    for matched_segment in segments {
        let coords = matched_segment.coordinates();
        if !coords.is_empty() {
            // Simply add all coordinates from this segment
            for coord in &coords {
                combined_coords.push(coord_to_json(coord));
            }
        }
    }

    // Create the main route feature
    let main_route = json!({
        "type": "Feature",
        "properties": {
            "type": "matched_route",
            "color": "#FF0000",
            "weight": 4,
            "description": "Route-Based Matched Route"
        },
        "geometry": {
            "type": "LineString",
            "coordinates": combined_coords
        }
    });

    // Add the combined route as the first feature
    features.insert(0, main_route);

    // Add junction points as separate features for reference
    let mut junction_points = Vec::new();

    if segments.len() > 1 {
        for i in 0..segments.len() - 1 {
            if let (Some(end_node), Some(start_node)) =
                (segments[i].end_node(), segments[i + 1].start_node())
            {
                if end_node == start_node {
                    // This is a junction point
                    let coords = segments[i].coordinates();
                    if let Some(coord) = coords.last() {
                        junction_points.push(json!({
                            "type": "Feature",
                            "properties": {
                                "type": "junction",
                                "node_id": end_node
                            },
                            "geometry": {
                                "type": "Point",
                                "coordinates": [coord.x, coord.y]
                            }
                        }));
                    }
                }
            }
        }
    }

    // Add junction points to features
    features.extend(junction_points);

    // Return the complete GeoJSON
    json!({
        "type": "FeatureCollection",
        "features": features
    })
}

impl From<&MatchedWaySegment> for proto::RoadSegment {
    fn from(matched_segment: &MatchedWaySegment) -> Self {
        let segment = &matched_segment.segment;

        // Get the actual coordinates based on interim indices
        let coords = matched_segment.coordinates();

        // Save the original full coordinates for reference
        let full_coords = segment
            .coordinates
            .iter()
            .map(|v| proto::LatLon { lat: v.y, lon: v.x })
            .collect();

        Self {
            id: segment.id,
            osm_way_id: segment.osm_way_id,
            coordinates: coords
                .iter()
                .map(|v| proto::LatLon { lat: v.y, lon: v.x })
                .collect(),
            is_oneway: segment.is_oneway,
            highway_type: segment.highway_type.clone(),
            connections: segment.connections.clone(),
            name: segment.name.clone(),
            interim_start_idx: matched_segment.entry_node.map(|idx| idx as u32),
            interim_end_idx: matched_segment.exit_node.map(|idx| idx as u32),
            full_coordinates: full_coords,
        }
    }
}

impl From<&WaySegment> for proto::RoadSegment {
    fn from(segment: &WaySegment) -> Self {
        // Get the actual coordinates based on interim indices
        let coords = segment.coordinates.clone();

        // Save the original full coordinates for reference
        let full_coords = segment
            .coordinates
            .iter()
            .map(|v| proto::LatLon { lat: v.y, lon: v.x })
            .collect();

        Self {
            id: segment.id,
            osm_way_id: segment.osm_way_id,
            coordinates: coords
                .iter()
                .map(|v| proto::LatLon { lat: v.y, lon: v.x })
                .collect(),
            is_oneway: segment.is_oneway,
            highway_type: segment.highway_type.clone(),
            connections: segment.connections.clone(),
            name: segment.name.clone(),
            interim_start_idx: None,
            interim_end_idx: None,
            full_coordinates: full_coords,
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
            // Add new debug fields
            constraints: window_trace
                .constraints
                .iter()
                .map(|constraint| proto::PointConstraint {
                    point_idx: constraint.point_idx as u32,
                    segment_id: constraint.segment_id,
                    way_id: constraint.way_id,
                    distance: constraint.distance,
                })
                .collect(),
            used_constraints: window_trace.used_constraints,
            constraint_score: window_trace.constraint_score,
            unconstrained_score: window_trace.unconstrained_score,
            attempted_way_ids: window_trace.attempted_way_ids.clone(),
            debug_notes: window_trace.debug_notes.clone(),
        }
    }
}

impl From<PathfindingDebugInfo> for proto::PathfindingDebugInfo {
    fn from(debug_info: PathfindingDebugInfo) -> Self {
        // Convert start candidates
        let start_candidates: Vec<proto::SegmentCandidate> = debug_info
            .start_candidates
            .iter()
            .map(|candidate| proto::SegmentCandidate {
                segment: Some((&candidate.segment).into()),
                distance: candidate.distance,
                projection: Some(proto::Point {
                    latlon: Some(proto::LatLon {
                        lat: candidate.projection.y(),
                        lon: candidate.projection.x(),
                    }),
                    timestamp: None,
                    label: None,
                    note: None,
                    elevation: None,
                }),
                cost: candidate.cost,
            })
            .collect();

        // Convert end candidates
        let end_candidates: Vec<proto::SegmentCandidate> = debug_info
            .end_candidates
            .iter()
            .map(|candidate| proto::SegmentCandidate {
                segment: Some((&candidate.segment).into()),
                distance: candidate.distance,
                projection: Some(proto::Point {
                    latlon: Some(proto::LatLon {
                        lat: candidate.projection.y(),
                        lon: candidate.projection.x(),
                    }),
                    timestamp: None,
                    label: None,
                    note: None,
                    elevation: None,
                }),
                cost: candidate.cost,
            })
            .collect();

        // Convert constraints
        let constraints: Vec<proto::PointConstraintPair> = debug_info
            .constraints
            .iter()
            .map(|(point_idx, segment_id)| proto::PointConstraintPair {
                point_idx: *point_idx as u32,
                segment_id: *segment_id,
            })
            .collect();

        // Convert constrained candidates map
        let mut constrained_candidates = HashMap::new();
        for (point_idx, seg_ids) in &debug_info.constrained_candidates {
            constrained_candidates.insert(*point_idx as u32, seg_ids.len() as u32);
        }

        // Convert attempted pairs
        let attempted_pairs: Vec<proto::PathfindingAttempt> = debug_info
            .attempted_pairs
            .iter()
            .map(|attempt| {
                let result = match &attempt.result {
                    PathfindingResult::Success(path, cost) => proto::PathfindingResult {
                        r#type: proto::pathfinding_result::ResultType::Success as i32,
                        path: path.iter().map(|segment| segment.into()).collect(),
                        cost: *cost,
                        max_distance: 0.0,    // Not applicable for success
                        actual_distance: 0.0, // Not applicable for success
                        reason: String::new(),
                    },
                    PathfindingResult::TooFar(max_dist, actual_dist) => proto::PathfindingResult {
                        r#type: proto::pathfinding_result::ResultType::TooFar as i32,
                        path: Vec::new(),
                        cost: 0.0,
                        max_distance: *max_dist,
                        actual_distance: *actual_dist,
                        reason: format!(
                            "Distance ({:.2}m) exceeds max limit ({:.2}m)",
                            actual_dist, max_dist
                        ),
                    },
                    PathfindingResult::NoConnection => proto::PathfindingResult {
                        r#type: proto::pathfinding_result::ResultType::NoConnection as i32,
                        path: Vec::new(),
                        cost: 0.0,
                        max_distance: 0.0,
                        actual_distance: 0.0,
                        reason: "Segments are not connected".to_string(),
                    },
                    PathfindingResult::NoPathFound(reason) => proto::PathfindingResult {
                        r#type: proto::pathfinding_result::ResultType::NoPathFound as i32,
                        path: Vec::new(),
                        cost: 0.0,
                        max_distance: 0.0,
                        actual_distance: 0.0,
                        reason: reason.clone(),
                    },
                };

                proto::PathfindingAttempt {
                    from_segment: attempt.from_segment,
                    from_osm_way: attempt.from_osm_way,
                    to_segment: attempt.to_segment,
                    to_osm_way: attempt.to_osm_way,
                    distance: attempt.distance,
                    result: Some(result),
                }
            })
            .collect();

        proto::PathfindingDebugInfo {
            start_point_idx: debug_info.start_point_idx as u32,
            end_point_idx: debug_info.end_point_idx as u32,
            start_candidates,
            end_candidates,
            constraints,
            attempted_pairs,
            constrained_candidates,
            reason: debug_info.reason.clone(),
        }
    }
}

impl From<LatLon> for geo::Point<f64> {
    fn from(value: LatLon) -> Self {
        Point::new(value.lon, value.lat)
    }
}

pub mod proto {
    tonic::include_proto!("chronotopia");
}
