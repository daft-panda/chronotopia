use anyhow::Result;
use chrono::{DateTime, Utc};
use geo::Point;
use log::{error, info, warn};
use sea_orm::{
    ActiveModelTrait, ColumnTrait, DatabaseConnection, EntityTrait, IntoActiveModel,
    PaginatorTrait, QueryFilter, QueryOrder, QuerySelect, Set, TransactionTrait,
};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{self, Duration};
use uuid::Uuid;

use crate::entity::{import_summary, ingest_batches, trips, user_processing_state};
use crate::proto::{self, Points};
use crate::route_matcher::RouteMatchJob;
use crate::trip_builder::TripBuilder;

/// Service for scheduling and managing trip processing jobs
#[derive(Debug, Clone)]
pub struct TripProcessor {
    db: DatabaseConnection,
    trip_builder: Arc<TripBuilder>,
    processing: Arc<Mutex<bool>>,
}

impl TripProcessor {
    /// Create a new TripProcessor
    pub fn new(db: DatabaseConnection, trip_builder: Arc<TripBuilder>) -> Self {
        Self {
            db,
            trip_builder,
            processing: Arc::new(Mutex::new(false)),
        }
    }

    /// Start the periodic trip processing scheduler
    pub async fn start_scheduler(&self) {
        let interval = Duration::from_secs(60 * 5); // Run every 5 minutes
        let mut interval = time::interval(interval);

        info!("Starting trip processor scheduler");

        loop {
            interval.tick().await;
            if let Err(e) = self.process_pending_batches().await {
                error!("Error processing pending batches: {}", e);
            }
        }
    }

    /// Process all pending batches for all users
    pub async fn process_pending_batches(&self) -> Result<()> {
        // Check if processing is already running
        let mut processing = self.processing.lock().await;
        if *processing {
            info!("Trip processing already in progress, skipping");
            return Ok(());
        }

        // Set the processing flag
        *processing = true;
        drop(processing); // Release the lock

        info!("Starting batch processing run");

        // Process all pending batches
        match self.trip_builder.process_all_pending_batches().await {
            Ok(_) => {
                info!("Batch processing completed successfully");
                // Update all import summaries
                self.update_import_summaries().await?;
                // Update user processing states
                self.update_user_processing_states().await?;
            }
            Err(e) => {
                error!("Error during batch processing: {}", e);
            }
        }

        // Reset the processing flag
        let mut processing = self.processing.lock().await;
        *processing = false;

        Ok(())
    }

    /// Process batches for a specific user
    pub async fn process_user_batches(&self, user_id: Uuid) -> Result<()> {
        info!("Processing batches for user {}", user_id);

        // Process the user's batches
        let processed_count = self.trip_builder.process_user_batches(user_id).await?;

        if processed_count > 0 {
            // Update import summaries for this user
            self.update_user_import_summaries(user_id).await?;
            // Update user processing state
            self.update_single_user_processing_state(user_id).await?;
        }

        Ok(())
    }

    /// Update import summaries based on processed batches
    async fn update_import_summaries(&self) -> Result<()> {
        info!("Updating import summaries");

        // Get all import summaries that are not marked as complete
        let pending_imports = import_summary::Entity::find()
            .filter(import_summary::Column::ProcessingComplete.eq(false))
            .all(&self.db)
            .await?;

        for import in pending_imports {
            // Check if all batches related to this import have been processed
            // This requires matching on source_info JSON which contains import ID
            // For simplicity, we'll count the trips generated from this import

            let trip_count: u64 = trips::Entity::find()
                .filter(trips::Column::UserId.eq(import.user_id))
                .filter(trips::Column::StartTime.gte(import.import_date_time))
                .count(&self.db)
                .await?;

            let mut import_model = import.clone().into_active_model();
            import_model.processing_complete = Set(true);
            import_model.processed_date_time = Set(Some(chrono::Utc::now().into()));
            import_model.generated_trips = Set(trip_count as i32);

            // Update the import summary
            import_model.update(&self.db).await?;
        }

        Ok(())
    }

    /// Update import summaries for a specific user
    async fn update_user_import_summaries(&self, user_id: Uuid) -> Result<()> {
        info!("Updating import summaries for user {}", user_id);

        // Same logic as above but filtered by user_id
        let pending_imports = import_summary::Entity::find()
            .filter(import_summary::Column::UserId.eq(user_id))
            .filter(import_summary::Column::ProcessingComplete.eq(false))
            .all(&self.db)
            .await?;

        for import in pending_imports {
            let trip_count: u64 = trips::Entity::find()
                .filter(trips::Column::UserId.eq(user_id))
                .filter(trips::Column::StartTime.gte(import.import_date_time))
                .count(&self.db)
                .await?;

            let mut import_model = import.clone().into_active_model();
            import_model.processing_complete = Set(true);
            import_model.processed_date_time = Set(Some(chrono::Utc::now().into()));
            import_model.generated_trips = Set(trip_count as i32);

            import_model.update(&self.db).await?;
        }

        Ok(())
    }

    /// Update processing state for all users
    async fn update_user_processing_states(&self) -> Result<()> {
        info!("Updating user processing states");

        // Get distinct user IDs with processed batches
        let users_with_batches = ingest_batches::Entity::find()
            .filter(ingest_batches::Column::Processed.eq(true))
            .select_only()
            .column(ingest_batches::Column::UserId)
            .distinct()
            .into_tuple()
            .all(&self.db)
            .await?;

        for (user_id,) in users_with_batches {
            self.update_single_user_processing_state(user_id).await?;
        }

        Ok(())
    }

    /// Update processing state for a single user
    async fn update_single_user_processing_state(&self, user_id: Uuid) -> Result<()> {
        // Get the latest processed batch time for the user
        let latest_batch = ingest_batches::Entity::find()
            .filter(ingest_batches::Column::UserId.eq(user_id))
            .filter(ingest_batches::Column::Processed.eq(true))
            .order_by_desc(ingest_batches::Column::BatchDateTime)
            .one(&self.db)
            .await?;

        // Get the total number of trips generated for the user
        let trip_count: u64 = trips::Entity::find()
            .filter(trips::Column::UserId.eq(user_id))
            .count(&self.db)
            .await?;

        // Get the total number of visits detected for the user
        let visit_count: u64 = crate::entity::user_visits_ingest::Entity::find()
            .filter(crate::entity::user_visits_ingest::Column::BatchId.eq(-1)) // The special value for derived visits
            .count(&self.db)
            .await?;

        if let Some(batch) = latest_batch {
            // Check if a processing state exists for this user
            let existing_state = user_processing_state::Entity::find_by_id(user_id)
                .one(&self.db)
                .await?;

            if let Some(state) = existing_state {
                // Update existing state
                let mut state_model = state.into_active_model();
                state_model.last_processed_time = Set(batch.batch_date_time);
                state_model.total_trips_generated = Set(trip_count as i32);
                state_model.total_visits_detected = Set(visit_count as i32);
                state_model.last_updated = Set(chrono::Utc::now().into());
                state_model.update(&self.db).await?;
            } else {
                // Create new state
                let state_model = user_processing_state::ActiveModel {
                    user_id: Set(user_id),
                    last_processed_time: Set(batch.batch_date_time),
                    total_trips_generated: Set(trip_count as i32),
                    total_visits_detected: Set(visit_count as i32),
                    last_updated: Set(chrono::Utc::now().into()),
                };
                user_processing_state::Entity::insert(state_model)
                    .exec(&self.db)
                    .await?;
            }
        }

        Ok(())
    }

    /// Process a specific trip
    pub async fn process_trip(&self, trip_id: Uuid) -> Result<()> {
        info!("Processing trip {}", trip_id);

        // Get the trip
        let trip = trips::Entity::find_by_id(trip_id)
            .one(&self.db)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Trip not found"))?;

        // Check if the trip is already processed
        if trip.processed {
            info!(
                "Trip {} is already processed, marking as unprocessed first",
                trip_id
            );

            // Mark as unprocessed to ensure it gets reprocessed
            let mut trip_model = trip.clone().into_active_model();
            trip_model.processed = Set(false);
            trip_model.update(&self.db).await?;
        }

        // Create a transaction for our processing
        let txn = self.db.begin().await?;

        // Extract GPS points from the trip
        let points: Vec<Point<f64>> = match &trip.points {
            Some(points_data) => {
                let points: Points = prost::Message::decode(points_data.as_slice())?;
                points
                    .points
                    .iter()
                    .filter_map(|p| p.latlon.as_ref().map(|ll| Point::new(ll.lon, ll.lat)))
                    .collect()
            }
            None => {
                warn!("Trip {} has no points", trip_id);
                return Err(anyhow::anyhow!("Trip has no points"));
            }
        };

        // Extract timestamps
        let timestamps: Vec<DateTime<Utc>> = match &trip.points {
            Some(points_data) => {
                let points: Points = prost::Message::decode(points_data.as_slice())?;
                points
                    .points
                    .iter()
                    .map(|p| p.date_time.as_ref().unwrap().into())
                    .collect()
            }
            None => vec![Utc::now(); points.len()], // Fallback
        };

        // Setup route matching job
        let mut job = RouteMatchJob::new(points, timestamps, None);
        job.activate_tracing(); // Enable trace collection for debugging

        // Lock the route matcher
        let mut route_matcher = self.trip_builder.route_matcher.lock().await;

        // Perform map matching
        match route_matcher.match_trace(&mut job) {
            Ok(matched_segments) => {
                info!(
                    "Map matching successful for trip {}: {} segments matched",
                    trip_id,
                    matched_segments.len()
                );

                // Collect OSM way IDs from the matched segments
                let way_ids: Vec<i64> = matched_segments
                    .iter()
                    .map(|segment| segment.segment.osm_way_id as i64)
                    .collect();

                // Create a JSON array of OSM way IDs
                let way_ids_json = serde_json::to_value(&way_ids)?;

                // Calculate the bounding box of the trip
                let bounding_box = self
                    .trip_builder
                    .calculate_trip_bounding_box(&matched_segments)?;

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
                let matched_geojson = self
                    .trip_builder
                    .road_segments_to_geojson(&matched_segments);

                // Update the trip record with the matching results
                let trip_update = trips::ActiveModel {
                    id: Set(trip_id),
                    processed: Set(true),
                    osm_way_ids: Set(way_ids_json),
                    bounding_box: Set(postgis::ewkb::GeometryT::Polygon(bounding_box)),
                    route_match_trace: Set(Some(route_match_trace)),
                    geo_json: Set(Some(serde_json::to_string(&matched_geojson)?)),
                    last_modified: Set(Utc::now().into()),
                    ..Default::default()
                };

                trip_update.update(&txn).await?;
                info!("Updated trip {} with reprocessed route match data", trip_id);
            }
            Err(e) => {
                // Mark the trip as processed but with an error
                let trip_update = trips::ActiveModel {
                    id: Set(trip_id),
                    processed: Set(true), // Mark as processed even with error
                    notes: Set(Some(format!("Route matching failed: {}", e))),
                    last_modified: Set(Utc::now().into()),
                    ..Default::default()
                };

                trip_update.update(&txn).await?;
                warn!(
                    "Route-based map matching failed for trip {}: {}",
                    trip_id, e
                );
            }
        }

        // Commit the transaction
        txn.commit().await?;

        // Update the user processing state
        self.update_single_user_processing_state(trip.user_id)
            .await?;

        info!("Trip {} reprocessing completed", trip_id);
        Ok(())
    }
}
