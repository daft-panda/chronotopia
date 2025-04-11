use anyhow::Result;
use chrono::{DateTime, FixedOffset, Utc};
use geo::{Coord, Distance, Haversine, Point};
use log::{debug, error, info, warn};
use sea_orm::QuerySelect;
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, DatabaseConnection, DatabaseTransaction,
    EntityTrait, IntoActiveModel, PaginatorTrait, QueryFilter, QueryOrder, TransactionTrait,
};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::entity::{
    ingest_batches, trips, user_locations_ingest, user_visits_ingest,
};
use crate::proto::{self, Points};
use crate::route_matcher::{MatchedWaySegment, RouteMatchJob, RouteMatcher};

/// Maximum time gap between points to consider them part of the same trip (15 minutes)
const MAX_POINT_GAP_SECONDS: i64 = 15 * 60;

/// Minimum time at the same location to consider it a visit/stop (15 minutes)
const MIN_VISIT_DURATION_SECONDS: i64 = 15 * 60;

/// Maximum distance (in meters) to consider a user at the "same place"
const MAX_VISIT_RADIUS_METERS: f64 = 50.0;

/// Service for processing location batches into trips
#[derive(Debug)]
pub struct TripBuilder {
    db: DatabaseConnection,
    route_matcher: Mutex<RouteMatcher>,
}

/// State of processing for a single user's data
struct TripBuildingContext {
    user_id: Uuid,
    current_trip_points: Vec<user_locations_ingest::Model>,
    last_processed_batch: Option<i32>,
}

impl TripBuilder {
    /// Create a new TripBuilder service
    pub fn new(db: DatabaseConnection, route_matcher: RouteMatcher) -> Self {
        Self {
            db,
            route_matcher: Mutex::new(route_matcher),
        }
    }

    /// Process all pending batches for all users
    pub async fn process_all_pending_batches(&self) -> Result<()> {
        // Get distinct user IDs with pending batches
        let users_with_pending_batches = ingest_batches::Entity::find()
            .filter(
                ingest_batches::Column::Processed
                    .is_null()
                    .or(ingest_batches::Column::Processed.eq(false)),
            )
            .select_only()
            .column(ingest_batches::Column::UserId)
            .distinct()
            .into_tuple()
            .all(&self.db)
            .await?;

        info!(
            "Found {} users with pending batches to process",
            users_with_pending_batches.len()
        );

        // Process each user's batches
        for (user_id,) in users_with_pending_batches {
            info!("Processing batches for user {}", user_id);
            match self.process_user_batches(user_id).await {
                Ok(count) => info!("Processed {} batches for user {}", count, user_id),
                Err(e) => error!("Error processing batches for user {}: {}", user_id, e),
            }
        }

        Ok(())
    }

    /// Process pending batches for a specific user
    pub async fn process_user_batches(&self, user_id: Uuid) -> Result<usize> {
        let mut processed_count = 0;
        let mut context = TripBuildingContext {
            user_id,
            current_trip_points: Vec::new(),
            last_processed_batch: None,
        };

        // Find the most recent processed batch for continuity
        let last_processed = ingest_batches::Entity::find()
            .filter(ingest_batches::Column::UserId.eq(user_id))
            .filter(ingest_batches::Column::Processed.eq(true))
            .order_by_desc(ingest_batches::Column::BatchDateTime)
            .one(&self.db)
            .await?;

        if let Some(batch) = last_processed {
            // Get the timestamp of the latest point in the last processed batch
            let latest_point = user_locations_ingest::Entity::find()
                .filter(user_locations_ingest::Column::BatchId.eq(batch.id))
                .order_by_desc(user_locations_ingest::Column::DateTime)
                .one(&self.db)
                .await?;

            if let Some(point) = latest_point {
                // Check if there's an ongoing trip by looking at the most recent trip
                let latest_trip = trips::Entity::find()
                    .filter(trips::Column::UserId.eq(user_id))
                    .order_by_desc(trips::Column::EndTime)
                    .one(&self.db)
                    .await?;

                if let Some(trip) = latest_trip {
                    let trip_end = trip.end_time;
                    let point_time = point.date_time;

                    // If the last processed point is close to the last trip end,
                    // we might need to continue that trip
                    if (point_time.timestamp() - trip_end.timestamp()).abs() < MAX_POINT_GAP_SECONDS
                    {
                        debug!("Potential ongoing trip detected, checking for continuity");
                        // We'll leave this empty for now - when we process the next batch,
                        // the time gap logic will determine if it should continue the last trip
                    }
                }
            }
        }

        // Get all pending batches for this user, ordered by datetime
        let pending_batches = ingest_batches::Entity::find()
            .filter(ingest_batches::Column::UserId.eq(user_id))
            .filter(
                ingest_batches::Column::Processed
                    .is_null()
                    .or(ingest_batches::Column::Processed.eq(false)),
            )
            .order_by(ingest_batches::Column::BatchDateTime, sea_orm::Order::Asc)
            .paginate(&self.db, 100)
            .fetch_and_next()
            .await?
            .unwrap_or_default();

        for batch in pending_batches {
            let result = self.process_batch(&mut context, batch.id).await;
            match result {
                Ok(_) => {
                    processed_count += 1;
                    // Mark batch as processed
                    let mut batch_model = batch.into_active_model();
                    batch_model.processed = Set(Some(true));
                    batch_model.update(&self.db).await?;
                }
                Err(e) => {
                    error!("Error processing batch {}: {}", batch.id, e);
                    // Don't mark as processed if there was an error
                }
            }
        }

        // Finalize any in-progress trip
        if !context.current_trip_points.is_empty() {
            self.finalize_current_trip(&mut context).await?;
        }

        Ok(processed_count)
    }

    /// Process a single batch of location data
    async fn process_batch(&self, context: &mut TripBuildingContext, batch_id: i32) -> Result<()> {
        info!("Processing batch {} for user {}", batch_id, context.user_id);

        // Start a transaction for this batch processing
        let txn = self.db.begin().await?;

        // Get all location points from this batch, ordered by time
        let location_points = user_locations_ingest::Entity::find()
            .filter(user_locations_ingest::Column::BatchId.eq(batch_id))
            .order_by(user_locations_ingest::Column::DateTime, sea_orm::Order::Asc)
            .all(&txn)
            .await?;

        if location_points.is_empty() {
            debug!(
                "No location points in batch {}, marking as processed",
                batch_id
            );
            self.mark_batch_processed(&txn, batch_id).await?;
            txn.commit().await?;
            return Ok(());
        }

        // Process each location point
        for point in location_points {
            self.process_location_point(context, point, &txn).await?;
        }

        // Mark this batch as the last processed
        context.last_processed_batch = Some(batch_id);

        // Mark the batch as processed
        self.mark_batch_processed(&txn, batch_id).await?;

        // Commit the transaction
        txn.commit().await?;

        Ok(())
    }

    /// Process a single location point
    async fn process_location_point(
        &self,
        context: &mut TripBuildingContext,
        point: user_locations_ingest::Model,
        txn: &DatabaseTransaction,
    ) -> Result<()> {
        // If this is the first point, just add it to the current trip
        if context.current_trip_points.is_empty() {
            context.current_trip_points.push(point);
            return Ok(());
        }

        // Get the last point in the current trip
        let last_point = context.current_trip_points.last().unwrap().clone();

        // Calculate time difference between this point and the last one
        let time_diff = (point.date_time.timestamp() - last_point.date_time.timestamp()).abs();

        // Calculate distance between this point and the last one
        let p1 = Point::new(last_point.longitude, last_point.latitude);
        let p2 = Point::new(point.longitude, point.latitude);
        let distance = Haversine.distance(p1, p2);

        debug!(
            "Point time diff: {}s, distance: {:.2}m",
            time_diff, distance
        );

        // If the time gap is too large, end the current trip and start a new one
        if time_diff > MAX_POINT_GAP_SECONDS {
            debug!("Time gap too large ({}s), ending current trip", time_diff);

            if !context.current_trip_points.is_empty() {
                self.finalize_current_trip(context).await?;
            }

            // Start a new trip with this point
            context.current_trip_points.push(point);
            return Ok(());
        }

        // Check if the user is stationary (detection for visit)
        if distance < MAX_VISIT_RADIUS_METERS {
            // User is still in the same area
            // Check if time spent here is significant enough to consider it a visit
            let first_stationary_point = context.current_trip_points.last().unwrap().clone();
            let stationary_time =
                point.date_time.timestamp() - first_stationary_point.date_time.timestamp();

            if stationary_time >= MIN_VISIT_DURATION_SECONDS {
                debug!(
                    "User stationary for {}s at location ({}, {}), ending trip and recording visit",
                    stationary_time, point.latitude, point.longitude
                );

                // End the current trip
                if !context.current_trip_points.is_empty() {
                    self.finalize_current_trip(context).await?;
                }

                // Record a visit
                self.record_visit(
                    context.user_id,
                    first_stationary_point.date_time,
                    point.date_time,
                    last_point.latitude,
                    last_point.longitude,
                    txn,
                )
                .await?;

                // Start a new trip with this point
                context.current_trip_points.push(point);
                return Ok(());
            }
        }

        // Otherwise, add this point to the current trip
        context.current_trip_points.push(point);
        Ok(())
    }

    /// Finalize the current trip by storing it in the database
    async fn finalize_current_trip(&self, context: &mut TripBuildingContext) -> Result<()> {
        if context.current_trip_points.len() < 2 {
            debug!(
                "Not enough points ({}) to create a trip, skipping",
                context.current_trip_points.len()
            );
            context.current_trip_points.clear();
            return Ok(());
        }

        // Create a transaction for storing the trip
        let txn = self.db.begin().await?;

        // Create Trip model
        let first_point = context.current_trip_points.first().unwrap();
        let last_point = context.current_trip_points.last().unwrap();

        // Calculate total distance
        let mut total_distance = 0.0;
        for i in 1..context.current_trip_points.len() {
            let p1 = Point::new(
                context.current_trip_points[i - 1].longitude,
                context.current_trip_points[i - 1].latitude,
            );
            let p2 = Point::new(
                context.current_trip_points[i].longitude,
                context.current_trip_points[i].latitude,
            );
            total_distance += Haversine.distance(p1, p2);
        }

        // Check if a similar trip already exists to prevent duplicates
        let existing_trip = trips::Entity::find()
            .filter(trips::Column::UserId.eq(context.user_id))
            .filter(trips::Column::StartTime.eq(first_point.date_time))
            .filter(trips::Column::EndTime.eq(last_point.date_time))
            .one(&txn)
            .await?;

        if existing_trip.is_some() {
            debug!(
                "Trip already exists for user {} from {} to {}",
                context.user_id, first_point.date_time, last_point.date_time
            );
            context.current_trip_points.clear();
            txn.rollback().await?;
            return Ok(());
        }

        // Generate linestring from points
        let linestring = self.generate_linestring(&context.current_trip_points)?;

        // Generate a simple initial bounding box
        let bounding_box = self.generate_bounding_box(&context.current_trip_points)?;

        let points = Points {
            points: context
                .current_trip_points
                .iter()
                .map(|p| p.into())
                .collect(),
        };

        let mut points_buf: Vec<u8> = Vec::new();
        prost::Message::encode(&points, &mut points_buf)?;

        // Create the trip record
        let trip_id = Uuid::new_v4();
        let trip = trips::ActiveModel {
            id: Set(trip_id),
            user_id: Set(context.user_id),
            geometry: Set(linestring),
            bounding_box: Set(bounding_box),
            start_time: Set(first_point.date_time),
            end_time: Set(last_point.date_time),
            distance_meters: Set(total_distance),
            points: Set(Some(points_buf)),
            processed: Set(false),
            osm_way_ids: Set(serde_json::Value::Array(Vec::new())), // Empty array initially
            route_match_trace: Set(None),
            geo_json: Set(None),
            label: Set(None),
            notes: Set(None),
            last_modified: Set(Utc::now().into()),
        };

        // Save the trip
        let trip_model = trip.insert(&txn).await?;
        info!(
            "Created new trip {} for user {} with {} points, distance: {:.2}m",
            trip_model.id,
            context.user_id,
            context.current_trip_points.len(),
            total_distance
        );

        // Process with route matcher in background
        let mut route_matcher = self.route_matcher.lock().await;
        match self
            .process_trip_route_matching(&mut route_matcher, context, trip_model.id, &txn)
            .await
        {
            Ok(_) => {
                // Trip is updated within the processing function
                info!(
                    "Successfully processed route matching for trip {}",
                    trip_model.id
                );
            }
            Err(e) => {
                warn!(
                    "Failed to process route matching for trip {}: {}",
                    trip_model.id, e
                );
                // We'll still commit the trip, it's just not route-matched
            }
        }

        // Commit transaction
        txn.commit().await?;

        // Clear current trip points
        context.current_trip_points.clear();
        Ok(())
    }

    /// Record a visit in the database
    async fn record_visit(
        &self,
        user_id: Uuid,
        arrival_time: DateTime<FixedOffset>,
        departure_time: DateTime<FixedOffset>,
        latitude: f64,
        longitude: f64,
        txn: &DatabaseTransaction,
    ) -> Result<()> {
        // Check if a visit already exists with the same times
        let existing_visit = user_visits_ingest::Entity::find()
            .filter(user_visits_ingest::Column::ArrivalDateTime.eq(arrival_time))
            .filter(user_visits_ingest::Column::DepartureDateTime.eq(departure_time))
            .one(txn)
            .await?;

        if existing_visit.is_some() {
            debug!(
                "Visit already exists from {} to {}",
                arrival_time, departure_time
            );
            return Ok(());
        }

        debug!(
            "Recording visit for user {} at ({}, {}) from {} to {}",
            user_id, latitude, longitude, arrival_time, departure_time
        );

        // Try to identify what this place is (in a real implementation, this would
        // use reverse geocoding or a places API)
        let canonical_label = None; // Placeholder

        // Create the visit record - we're reusing the user_visits_ingest table,
        // but with a NULL batch_id to indicate it's derived rather than imported
        let visit = user_visits_ingest::ActiveModel {
            id: sea_orm::ActiveValue::NotSet,
            batch_id: Set(-1), // Special value to indicate derived visit
            latitude: Set(latitude),
            longitude: Set(longitude),
            horizontal_accuracy: Set(Some(MAX_VISIT_RADIUS_METERS)),
            arrival_date_time: Set(arrival_time),
            departure_date_time: Set(departure_time),
            canonical_label: Set(canonical_label),
            external_place_id: Set(None),
        };

        visit.insert(txn).await?;
        Ok(())
    }

    /// Process route matching for a trip
    async fn process_trip_route_matching(
        &self,
        route_matcher: &mut RouteMatcher,
        context: &TripBuildingContext,
        trip_id: Uuid,
        txn: &DatabaseTransaction,
    ) -> Result<()> {
        // Extract points and timestamps for route matching
        let mut gps_points = Vec::new();
        let mut timestamps = Vec::new();

        for point in &context.current_trip_points {
            gps_points.push(Point::new(point.longitude, point.latitude));
            timestamps.push(point.date_time.into());
        }

        let mut job = RouteMatchJob::new(gps_points, timestamps, None);
        job.activate_tracing();

        // Perform map matching
        info!(
            "Map matching trip {} with {} points",
            trip_id,
            job.gps_points.len()
        );

        match route_matcher.match_trace(&mut job) {
            Ok(result) => {
                info!(
                    "Map matching successful for trip {}: {} segments matched",
                    trip_id,
                    result.len()
                );

                // Collect OSM way IDs from the matched segments
                let way_ids: Vec<i64> = result
                    .iter()
                    .map(|segment| segment.segment.osm_way_id as i64)
                    .collect();

                // Create a JSON array of OSM way IDs
                let way_ids_json = serde_json::to_value(&way_ids)?;

                // Calculate the bounding box of the trip
                let bounding_box = self.calculate_trip_bounding_box(&result)?;

                // Serialize the route match trace to protobuf binary format
                let window_trace = job.window_trace.take();
                // Convert to proto RouteMatchTrace
                let mut proto_trace = proto::RouteMatchTrace {
                    window_traces: Vec::new(),
                    point_candidates: Vec::new(),
                };

                // Convert window traces
                for window in window_trace.iter() {
                    proto_trace.window_traces.push(window.into());
                }

                // Convert point candidates GeoJSON
                let point_candidates = job.point_candidates_geojson.take();

                for candidate in point_candidates.iter() {
                    proto_trace
                        .point_candidates
                        .push(serde_json::to_string(candidate).unwrap());
                }

                // Serialize to protobuf binary
                let mut route_match_trace: Vec<u8> = Vec::new();
                prost::Message::encode(&proto_trace, &mut route_match_trace)?;

                // Generate GeoJSON for the matched route
                let matched_geojson = self.road_segments_to_geojson(&result);

                // Update the trip record with the matching results
                let trip_update = trips::ActiveModel {
                    id: Set(trip_id),
                    processed: Set(true),
                    osm_way_ids: Set(way_ids_json),
                    bounding_box: Set(bounding_box),
                    route_match_trace: Set(Some(route_match_trace)),
                    geo_json: Set(Some(serde_json::to_string(&matched_geojson)?)),
                    ..Default::default()
                };

                trip_update.update(txn).await?;

                Ok(())
            }
            Err(e) => {
                warn!(
                    "Route-based map matching failed for trip {}: {}",
                    trip_id, e
                );
                Err(e)
            }
        }
    }

    /// Calculate the bounding box of a trip as a PostGIS polygon
    fn calculate_trip_bounding_box(&self, segments: &[MatchedWaySegment]) -> Result<String> {
        // Find min/max coordinates from all segments
        let mut min_lon = f64::MAX;
        let mut min_lat = f64::MAX;
        let mut max_lon = f64::MIN;
        let mut max_lat = f64::MIN;

        for segment in segments {
            for coord in segment.coordinates() {
                min_lon = min_lon.min(coord.x);
                min_lat = min_lat.min(coord.y);
                max_lon = max_lon.max(coord.x);
                max_lat = max_lat.max(coord.y);
            }
        }

        // Add a small buffer to the bounding box (approximately 100m)
        let buffer = 0.001; // roughly 100m in degrees
        min_lon -= buffer;
        min_lat -= buffer;
        max_lon += buffer;
        max_lat += buffer;

        // Create a PostGIS polygon in WKT format
        let polygon = format!(
            "SRID=4326;POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
            min_lon,
            min_lat, // bottom-left
            max_lon,
            min_lat, // bottom-right
            max_lon,
            max_lat, // top-right
            min_lon,
            max_lat, // top-left
            min_lon,
            min_lat // close the polygon by repeating first point
        );

        Ok(polygon)
    }

    /// Generate an initial bounding box from location points
    fn generate_bounding_box(&self, points: &[user_locations_ingest::Model]) -> Result<String> {
        // Find min/max coordinates from all points
        let mut min_lon = f64::MAX;
        let mut min_lat = f64::MAX;
        let mut max_lon = f64::MIN;
        let mut max_lat = f64::MIN;

        for point in points {
            min_lon = min_lon.min(point.longitude);
            min_lat = min_lat.min(point.latitude);
            max_lon = max_lon.max(point.longitude);
            max_lat = max_lat.max(point.latitude);
        }

        // Add a small buffer to the bounding box (approximately 100m)
        let buffer = 0.001; // roughly 100m in degrees
        min_lon -= buffer;
        min_lat -= buffer;
        max_lon += buffer;
        max_lat += buffer;

        // Create a PostGIS polygon in WKT format
        let polygon = format!(
            "SRID=4326;POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
            min_lon,
            min_lat, // bottom-left
            max_lon,
            min_lat, // bottom-right
            max_lon,
            max_lat, // top-right
            min_lon,
            max_lat, // top-left
            min_lon,
            min_lat // close the polygon by repeating first point
        );

        Ok(polygon)
    }

    /// Mark a batch as processed
    async fn mark_batch_processed(&self, txn: &DatabaseTransaction, batch_id: i32) -> Result<()> {
        let batch = ingest_batches::Entity::find_by_id(batch_id)
            .one(txn)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Batch not found"))?;

        let mut batch_model = batch.into_active_model();
        batch_model.processed = Set(Some(true));
        batch_model.update(txn).await?;

        Ok(())
    }

    /// Generate a PostGIS linestring from points
    fn generate_linestring(&self, points: &[user_locations_ingest::Model]) -> Result<String> {
        // Create a PostGIS-compatible linestring in WKT format
        let coords: Vec<String> = points
            .iter()
            .map(|p| format!("{} {}", p.longitude, p.latitude))
            .collect();

        let linestring = format!("SRID=4326;LINESTRING({})", coords.join(","));
        Ok(linestring)
    }

    /// Generate a GeoJSON based of the matched route
    fn road_segments_to_geojson(&self, segments: &[MatchedWaySegment]) -> serde_json::Value {
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
        let main_route = serde_json::json!({
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
                            junction_points.push(serde_json::json!({
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
        serde_json::json!({
            "type": "FeatureCollection",
            "features": features
        })
    }
}
