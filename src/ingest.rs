use std::collections::HashSet;

use anyhow::Result;
use chrono::{DateTime, FixedOffset, Utc};
use log::{debug, info};
use sea_orm::ActiveValue::NotSet;
use sea_orm::prelude::DateTimeWithTimeZone;
use sea_orm::{ActiveValue::Set, DatabaseConnection, EntityTrait};
use sea_orm::{ColumnTrait, Condition, QueryFilter as _, QuerySelect as _};
use tonic::{Request, Response, Status};
use uuid::Uuid;

use crate::entity::sea_orm_active_enums;
use crate::entity::{
    device_metadata, ingest_batches, user_activity_ingest, user_locations_ingest,
    user_visits_ingest,
};
use crate::proto::activity_event::ActivityType as ProtoActivityType;
use crate::proto::ingest_server::Ingest;
use crate::proto::{
    ActivityEvent as ProtoActivityEvent, DeviceMetadata as ProtoDeviceMetadata,
    GoogleMapsTimelineExport, IngestBatch as ProtoIngestBatch, IngestResponse,
    LocationPoint as ProtoLocationPoint, VisitEvent as ProtoVisitEvent,
};

#[derive(Clone)]
pub struct IngestService {
    db: DatabaseConnection,
}

impl IngestService {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }

    // Helper to get the latest device metadata for a user
    async fn get_latest_device_metadata(&self, user_id: Uuid) -> Result<Option<Uuid>, Status> {
        use device_metadata::Entity as DeviceMetadata;
        use sea_orm::query::*;

        let maybe_metadata = DeviceMetadata::find()
            .filter(device_metadata::Column::UserId.eq(user_id))
            .order_by_desc(device_metadata::Column::DateCreated)
            .one(&self.db)
            .await
            .map_err(|e| Status::internal(format!("Database error: {}", e)))?;

        Ok(maybe_metadata.map(|m| m.id))
    }

    // Helper method to convert IngestBatch.source to JSONB
    fn extract_source_info(batch: &ProtoIngestBatch) -> Option<serde_json::Value> {
        use crate::proto::ingest_batch::Source;

        let source = batch.source.as_ref()?;

        let json_value = match source {
            Source::ChronotopiaApp(app) => {
                serde_json::json!({
                    "type": "chronotopia_app",
                    "app_version": app.app_version
                })
            }
            Source::ChronotopiaWeb(web) => {
                serde_json::json!({
                    "type": "chronotopia_web",
                    "web_version": web.web_version
                })
            }
            Source::ChronotopiaApi(api) => {
                serde_json::json!({
                    "type": "chronotopia_api",
                    "api_version": api.api_version,
                    "used_api_key": api.used_api_key
                })
            }
            Source::GoogleMaps(maps) => {
                serde_json::json!({
                    "type": "google_maps",
                    "version": maps.version
                })
            }
            Source::AppleHealth(health) => {
                serde_json::json!({
                    "type": "apple_health",
                    "version": health.version
                })
            }
            Source::FitnessApp(app) => {
                serde_json::json!({
                    "type": "fitness_app",
                    "app_name": app.app_name,
                    "app_version": app.app_version
                })
            }
        };

        Some(json_value)
    }

    // Process and store location points
    async fn store_location_points(
        &self,
        batch_id: i32,
        locations: Vec<ProtoLocationPoint>,
    ) -> Result<(), Status> {
        let mut location_models = Vec::with_capacity(locations.len());

        for point in locations {
            let datetime: DateTime<FixedOffset> = match point.date_time.as_ref() {
                Some(dt) => dt.into(),
                None => {
                    return Err(Status::invalid_argument(
                        "Missing datetime in location point",
                    ));
                }
            };

            let location_model = user_locations_ingest::ActiveModel {
                id: sea_orm::ActiveValue::NotSet,
                batch_id: Set(batch_id),
                latitude: Set(point.latitude),
                longitude: Set(point.longitude),
                altitude: Set(if point.altitude != 0.0 {
                    Some(point.altitude)
                } else {
                    None
                }),
                horizontal_accuracy: Set(if point.horizontal_accuracy != 0.0 {
                    Some(point.horizontal_accuracy)
                } else {
                    None
                }),
                vertical_accuracy: Set(if point.vertical_accuracy != 0.0 {
                    Some(point.vertical_accuracy)
                } else {
                    None
                }),
                speed_accuracy: Set(if point.speed_accuracy != 0.0 {
                    Some(point.speed_accuracy)
                } else {
                    None
                }),
                bearing_accuracy: Set(if point.bearing_accuracy != 0.0 {
                    Some(point.bearing_accuracy)
                } else {
                    None
                }),
                speed: Set(if point.speed != 0.0 {
                    Some(point.speed)
                } else {
                    None
                }),
                bearing: Set(if point.bearing != 0.0 {
                    Some(point.bearing)
                } else {
                    None
                }),
                date_time: Set(datetime),
                is_mock_location: Set(Some(point.is_mock_location)),
                floor_level: Set(point.floor_level),
                battery_level: Set(if point.battery_level != 0 {
                    Some(point.battery_level as i32)
                } else {
                    None
                }),
                network_type: Set(if point.network_type.is_some() {
                    Some(point.network_type.unwrap())
                } else {
                    None
                }),
            };

            location_models.push(location_model);
        }

        let location_models = &mut location_models.into_iter();

        // Use bulk insert for better performance
        loop {
            let chunk: Vec<user_locations_ingest::ActiveModel> =
                location_models.take(1000).collect();
            if chunk.is_empty() {
                break;
            }

            user_locations_ingest::Entity::insert_many(chunk)
                .exec(&self.db)
                .await
                .map_err(|e| {
                    Status::internal(format!("Failed to insert location points: {}", e))
                })?;
        }

        Ok(())
    }

    // Process and store activity events
    async fn store_activity_events(
        &self,
        batch_id: i32,
        activity_events: Vec<ProtoActivityEvent>,
    ) -> Result<(), Status> {
        let mut activity_models = Vec::with_capacity(activity_events.len());

        for event in activity_events {
            let start_datetime: DateTime<FixedOffset> = match event.start.as_ref() {
                Some(dt) => dt.into(),
                None => {
                    return Err(Status::invalid_argument(
                        "Missing start datetime in activity event",
                    ));
                }
            };

            let end_datetime: Option<DateTime<FixedOffset>> =
                event.end.as_ref().map(|dt| dt.into());

            // Map activity type from proto enum to database enum
            let activity_type = match ProtoActivityType::from_i32(event.r#type) {
                Some(ProtoActivityType::Unknown) => sea_orm_active_enums::ActivityType::Unknown,
                Some(ProtoActivityType::Still) => sea_orm_active_enums::ActivityType::Still,
                Some(ProtoActivityType::Walking) => sea_orm_active_enums::ActivityType::Walking,
                Some(ProtoActivityType::Running) => sea_orm_active_enums::ActivityType::Running,
                Some(ProtoActivityType::InVehicle) => sea_orm_active_enums::ActivityType::InVehicle,
                Some(ProtoActivityType::OnBicycle) => sea_orm_active_enums::ActivityType::OnBicycle,
                Some(ProtoActivityType::OnFoot) => sea_orm_active_enums::ActivityType::OnFoot,
                Some(ProtoActivityType::Tilting) => sea_orm_active_enums::ActivityType::Tilting,
                None => sea_orm_active_enums::ActivityType::Unknown,
            };

            let activity_model = user_activity_ingest::ActiveModel {
                id: sea_orm::ActiveValue::NotSet,
                batch_id: Set(batch_id),
                r#type: Set(activity_type),
                confidence: Set(event.confidence),
                start_date_time: Set(start_datetime),
                end_date_time: Set(end_datetime),
                step_count: Set(event.step_count),
                distance: Set(event.distance),
            };

            activity_models.push(activity_model);
        }

        let activity_models = &mut activity_models.into_iter();

        // Use bulk insert for better performance
        loop {
            let chunk: Vec<user_activity_ingest::ActiveModel> =
                activity_models.take(1000).collect();
            if chunk.is_empty() {
                break;
            }

            user_activity_ingest::Entity::insert_many(chunk)
                .exec(&self.db)
                .await
                .map_err(|e| Status::internal(format!("Failed to insert activity: {}", e)))?;
        }

        Ok(())
    }

    // Process and store visit events
    async fn store_visit_events(
        &self,
        batch_id: i32,
        visit_events: Vec<ProtoVisitEvent>,
    ) -> Result<(), Status> {
        let mut visit_models = Vec::with_capacity(visit_events.len());

        for event in visit_events {
            let arrival_datetime: DateTime<FixedOffset> = match event.arrival.as_ref() {
                Some(dt) => dt.into(),
                None => {
                    return Err(Status::invalid_argument(
                        "Missing arrival datetime in visit event",
                    ));
                }
            };

            let departure_datetime: DateTime<FixedOffset> = match event.departure.as_ref() {
                Some(dt) => dt.into(),
                None => {
                    return Err(Status::invalid_argument(
                        "Missing departure datetime in visit event",
                    ));
                }
            };

            let visit_model = user_visits_ingest::ActiveModel {
                id: sea_orm::ActiveValue::NotSet,
                batch_id: Set(Some(batch_id)),
                latitude: Set(event.latitude),
                longitude: Set(event.longitude),
                horizontal_accuracy: Set(if event.horizontal_accuracy != 0.0 {
                    Some(event.horizontal_accuracy)
                } else {
                    None
                }),
                arrival_date_time: Set(arrival_datetime.fixed_offset()),
                departure_date_time: Set(departure_datetime.fixed_offset()),
                canonical_label: Set(if !event.canonical_label.is_empty() {
                    Some(event.canonical_label)
                } else {
                    None
                }),
                external_place_id: Set(event.external_place_id),
            };

            visit_models.push(visit_model);
        }

        let visit_models = &mut visit_models.into_iter();

        // Use bulk insert for better performance
        loop {
            let chunk: Vec<user_visits_ingest::ActiveModel> = visit_models.take(1000).collect();
            if chunk.is_empty() {
                break;
            }

            user_visits_ingest::Entity::insert_many(chunk)
                .exec(&self.db)
                .await
                .map_err(|e| Status::internal(format!("Failed to insert visits: {}", e)))?;
        }

        Ok(())
    }

    // Helper method to parse place location from Google's format
    fn parse_place_location(place_location: &str) -> Option<(f64, f64)> {
        if !place_location.starts_with("geo:") {
            return None;
        }

        let coords = place_location.replace("geo:", "");
        let mut parts = coords.split(',');

        let lat = parts.next()?.parse::<f64>().ok()?;
        let lon = parts.next()?.parse::<f64>().ok()?;

        Some((lat, lon))
    }

    // Helper method to map Google activity types to our enum
    fn map_google_activity_type(google_type: &str) -> ProtoActivityType {
        match google_type.to_lowercase().as_str() {
            "still" => ProtoActivityType::Still,
            "walking" => ProtoActivityType::Walking,
            "running" => ProtoActivityType::Running,
            "on_bicycle" | "cycling" => ProtoActivityType::OnBicycle,
            "in_vehicle" | "driving" => ProtoActivityType::InVehicle,
            "on_foot" => ProtoActivityType::OnFoot,
            _ => ProtoActivityType::Unknown,
        }
    }

    // Helper method to parse probability strings to confidence integers
    fn parse_probability(probability: &str) -> i32 {
        match probability.parse::<f64>() {
            Ok(value) => (value * 100.0).round() as i32,
            Err(_) => 0,
        }
    }

    // New function to filter out overlapping data from a batch
    async fn filter_batch_overlap(
        &self,
        user_id: Uuid,
        batch: &mut ProtoIngestBatch,
    ) -> Result<(), Status> {
        // Early return if batch is empty
        if batch.locations.is_empty() && batch.activities.is_empty() && batch.visits.is_empty() {
            return Ok(());
        }

        // Determine the time range of the current batch
        let mut min_time: Option<DateTime<Utc>> = None;
        let mut max_time: Option<DateTime<Utc>> = None;

        // Check location points
        for loc in &batch.locations {
            if let Some(dt) = &loc.date_time {
                let time: DateTime<Utc> = dt.into();
                min_time = Some(min_time.map_or(time, |t| if time < t { time } else { t }));
                max_time = Some(max_time.map_or(time, |t| if time > t { time } else { t }));
            }
        }

        // Check activity events
        for act in &batch.activities {
            if let Some(dt) = &act.start {
                let time: DateTime<Utc> = dt.into();
                min_time = Some(min_time.map_or(time, |t| if time < t { time } else { t }));
            }
            if let Some(dt) = &act.end {
                let time: DateTime<Utc> = dt.into();
                max_time = Some(max_time.map_or(time, |t| if time > t { time } else { t }));
            }
        }

        // Check visit events
        for visit in &batch.visits {
            if let Some(dt) = &visit.arrival {
                let time: DateTime<Utc> = dt.into();
                min_time = Some(min_time.map_or(time, |t| if time < t { time } else { t }));
            }
            if let Some(dt) = &visit.departure {
                let time: DateTime<Utc> = dt.into();
                max_time = Some(max_time.map_or(time, |t| if time > t { time } else { t }));
            }
        }

        // If we couldn't determine a time range, just return the original batch
        if min_time.is_none() || max_time.is_none() {
            debug!("No time range could be determined for batch, skipping overlap check");
            return Ok(());
        }

        let min_time = min_time.unwrap();
        let max_time = max_time.unwrap();

        debug!(
            "Checking for overlapping data for user {} between {} and {}",
            user_id, min_time, max_time
        );

        // Find existing batches in this time range
        let overlapping_batches = ingest_batches::Entity::find()
            .filter(
                Condition::all()
                    .add(ingest_batches::Column::UserId.eq(user_id))
                    .add(
                        Condition::any()
                            // Check if batch overlaps with the time range
                            .add(
                                Condition::all()
                                    .add(ingest_batches::Column::BatchDateTime.lte(max_time))
                                    .add(ingest_batches::Column::BatchDateTime.gte(min_time)),
                            )
                            // Also include batches that might contain data in this range
                            .add(
                                Condition::all()
                                    .add(ingest_batches::Column::BatchDateTime.lte(min_time))
                                    .add(ingest_batches::Column::Processed.eq(true)),
                            ),
                    ),
            )
            .all(&self.db)
            .await
            .map_err(|e| Status::internal(format!("Failed to query existing batches: {}", e)))?;

        if overlapping_batches.is_empty() {
            debug!("No overlapping batches found, proceeding with full import");
            return Ok(());
        }

        debug!(
            "Found {} potentially overlapping batches, filtering data",
            overlapping_batches.len()
        );

        // Get all the batch IDs
        let batch_ids: Vec<i32> = overlapping_batches.iter().map(|b| b.id).collect();

        // Filter out overlapping location points
        if !batch.locations.is_empty() {
            // Get existing location timestamps in the time range
            let existing_locations = user_locations_ingest::Entity::find()
                .select_only()
                .column(user_locations_ingest::Column::DateTime)
                .filter(
                    Condition::all()
                        .add(user_locations_ingest::Column::BatchId.is_in(batch_ids.clone()))
                        .add(user_locations_ingest::Column::DateTime.gte(min_time))
                        .add(user_locations_ingest::Column::DateTime.lte(max_time)),
                )
                .into_tuple()
                .all(&self.db)
                .await
                .map_err(|e| {
                    Status::internal(format!("Failed to query existing locations: {}", e))
                })?;

            // Create a set of existing timestamps for fast lookup
            let existing_timestamps: HashSet<i64> = existing_locations
                .iter()
                .map(|dt: &(DateTimeWithTimeZone,)| dt.0.timestamp())
                .collect();

            // Filter locations that don't exist in the database yet
            batch.locations = batch
                .locations
                .iter()
                .filter(|loc| {
                    if let Some(dt) = &loc.date_time {
                        let time: DateTime<Utc> = dt.into();
                        !existing_timestamps.contains(&time.timestamp())
                    } else {
                        // Keep points without a timestamp (though these should be rare/none)
                        true
                    }
                })
                .cloned()
                .collect();

            debug!(
                "Filtered locations: {} new out of {} total",
                batch.locations.len(),
                batch.locations.len()
            );
        }

        // Filter out overlapping activities
        if !batch.activities.is_empty() {
            // For activities, we need to check both start and end times
            let existing_activities: Vec<(
                Option<DateTime<FixedOffset>>,
                Option<DateTime<FixedOffset>>,
            )> = user_activity_ingest::Entity::find()
                .select_only()
                .column(user_activity_ingest::Column::StartDateTime)
                .column(user_activity_ingest::Column::EndDateTime)
                .filter(
                    Condition::all()
                        .add(user_activity_ingest::Column::BatchId.is_in(batch_ids.clone()))
                        .add(
                            Condition::any()
                                .add(
                                    user_activity_ingest::Column::StartDateTime
                                        .between(min_time, max_time),
                                )
                                .add(
                                    user_activity_ingest::Column::EndDateTime
                                        .between(min_time, max_time),
                                )
                                .add(
                                    Condition::all()
                                        .add(
                                            user_activity_ingest::Column::StartDateTime
                                                .lte(min_time),
                                        )
                                        .add(
                                            user_activity_ingest::Column::EndDateTime.gte(max_time),
                                        ),
                                ),
                        ),
                )
                .into_tuple()
                .all(&self.db)
                .await
                .map_err(|e| {
                    Status::internal(format!("Failed to query existing activities: {}", e))
                })?;

            // Create a lookup structure for existing activities based on start/end times
            let mut existing_activities_set = std::collections::HashSet::new();
            for (start, end) in existing_activities {
                let key = (start.map(|dt| dt.timestamp()), end.map(|dt| dt.timestamp()));
                existing_activities_set.insert(key);
            }

            // Filter activities
            batch.activities = batch
                .activities
                .iter()
                .filter(|act| {
                    let start_time = act.start.as_ref().map(|dt| {
                        let time: DateTime<Utc> = dt.into();
                        time.timestamp()
                    });

                    let end_time = act.end.as_ref().map(|dt| {
                        let time: DateTime<Utc> = dt.into();
                        time.timestamp()
                    });

                    let key = (start_time, end_time);
                    !existing_activities_set.contains(&key)
                })
                .cloned()
                .collect();

            debug!(
                "Filtered activities: {} new out of {} total",
                batch.activities.len(),
                batch.activities.len()
            );
        }

        // Filter out overlapping visits
        if !batch.visits.is_empty() {
            // For visits, we'll check by arrival and departure times
            let existing_visits: Vec<(
                Option<DateTime<FixedOffset>>,
                Option<DateTime<FixedOffset>>,
            )> = user_visits_ingest::Entity::find()
                .select_only()
                .column(user_visits_ingest::Column::ArrivalDateTime)
                .column(user_visits_ingest::Column::DepartureDateTime)
                .filter(
                    Condition::all()
                        .add(user_visits_ingest::Column::BatchId.is_in(batch_ids))
                        .add(
                            Condition::any()
                                .add(
                                    user_visits_ingest::Column::ArrivalDateTime
                                        .between(min_time, max_time),
                                )
                                .add(
                                    user_visits_ingest::Column::DepartureDateTime
                                        .between(min_time, max_time),
                                )
                                .add(
                                    Condition::all()
                                        .add(
                                            user_visits_ingest::Column::ArrivalDateTime
                                                .lte(min_time),
                                        )
                                        .add(
                                            user_visits_ingest::Column::DepartureDateTime
                                                .gte(max_time),
                                        ),
                                ),
                        ),
                )
                .into_tuple()
                .all(&self.db)
                .await
                .map_err(|e| Status::internal(format!("Failed to query existing visits: {}", e)))?;

            // Create a lookup structure for existing visits based on time periods
            let mut existing_visits_set = std::collections::HashSet::new();
            for (arrival, departure) in existing_visits {
                let key = (arrival.unwrap().timestamp(), departure.unwrap().timestamp());
                existing_visits_set.insert(key);
            }

            // Filter visits
            batch.visits = batch
                .visits
                .iter()
                .filter(|visit| {
                    let arrival = visit
                        .arrival
                        .as_ref()
                        .map(|dt| {
                            let time: DateTime<Utc> = dt.into();
                            time.timestamp()
                        })
                        .unwrap_or(0);

                    let departure = visit
                        .departure
                        .as_ref()
                        .map(|dt| {
                            let time: DateTime<Utc> = dt.into();
                            time.timestamp()
                        })
                        .unwrap_or(0);

                    let key = (arrival, departure);
                    !existing_visits_set.contains(&key)
                })
                .cloned()
                .collect();

            debug!(
                "Filtered visits: {} new out of {} total",
                batch.visits.len(),
                batch.visits.len()
            );
        }

        Ok(())
    }
}

#[tonic::async_trait]
impl Ingest for IngestService {
    async fn submit_batch(
        &self,
        request: Request<ProtoIngestBatch>,
    ) -> Result<Response<IngestResponse>, Status> {
        // Extract authentication context and user id from request
        let user_id = match request.extensions().get::<Uuid>() {
            Some(user_id) => *user_id,
            None => return Err(Status::unauthenticated("User not authenticated")),
        };

        let mut batch = request.into_inner();
        debug!(
            "Received ingest batch from user {}: {} locations, {} activities, {} visits",
            user_id,
            batch.locations.len(),
            batch.activities.len(),
            batch.visits.len()
        );

        // Filter out any data that overlaps with existing records
        self.filter_batch_overlap(user_id, &mut batch).await?;

        // If batch is empty after filtering, return early with success
        if batch.locations.is_empty() && batch.activities.is_empty() && batch.visits.is_empty() {
            info!(
                "All data in batch already exists for user {}, skipping insertion",
                user_id
            );
            return Ok(Response::new(IngestResponse {
                success: true,
                alert_message: "No new data to import, all records already exist.".into(),
                pause_tracking: false,
                recommended_upload_interval: 300,
                processed_locations: Some(0),
                processed_activities: Some(0),
                processed_visits: Some(0),
            }));
        }

        // Get the latest device metadata ID for this user
        let device_metadata_id = self.get_latest_device_metadata(user_id).await?;

        // Get batch timestamp
        let batch_datetime = match batch.date_time.as_ref() {
            Some(dt) => dt.into(),
            None => Utc::now(), // Use current time if not provided
        };

        // Extract source information
        let source_info = Self::extract_source_info(&batch);

        // Create and save batch record
        let batch_model = ingest_batches::ActiveModel {
            id: NotSet,
            user_id: Set(user_id),
            device_metadata_id: Set(device_metadata_id),
            batch_date_time: Set(batch_datetime.fixed_offset()),
            received_date_time: Set(Utc::now().fixed_offset()),
            ready_for_processing: Set(false),
            processed: Set(Some(false)),
            source_info: Set(source_info
                .map(|v| serde_json::to_value(&v).unwrap_or_default())
                .unwrap()),
        };

        let batch_result = ingest_batches::Entity::insert(batch_model)
            .exec(&self.db)
            .await
            .map_err(|e| Status::internal(format!("Failed to insert ingest batch: {}", e)))?;

        let batch_id = batch_result.last_insert_id;

        // Process and store the location points
        if !batch.locations.is_empty() {
            self.store_location_points(batch_id, batch.locations)
                .await?;
        }

        // Process and store the activity events
        if !batch.activities.is_empty() {
            self.store_activity_events(batch_id, batch.activities)
                .await?;
        }

        // Process and store the visit events
        if !batch.visits.is_empty() {
            self.store_visit_events(batch_id, batch.visits).await?;
        }

        // Mark the batch as processed successfully
        let update_batch = ingest_batches::ActiveModel {
            id: Set(batch_id),
            ready_for_processing: Set(true),
            ..Default::default()
        };

        ingest_batches::Entity::update(update_batch)
            .exec(&self.db)
            .await
            .map_err(|e| Status::internal(format!("Failed to update batch status: {}", e)))?;

        info!(
            "Successfully stored ingest batch for user {} with ID {}",
            user_id, batch_id
        );

        // Return success response
        Ok(Response::new(IngestResponse {
            success: true,
            alert_message: String::new(),
            pause_tracking: false,
            recommended_upload_interval: 300, // 5 minutes default
            processed_locations: None,
            processed_activities: None,
            processed_visits: None,
        }))
    }

    async fn submit_device_metadata(
        &self,
        request: Request<ProtoDeviceMetadata>,
    ) -> Result<Response<IngestResponse>, Status> {
        // Extract authentication context and user id from request
        let user_id = match request.extensions().get::<Uuid>() {
            Some(user_id) => *user_id,
            None => return Err(Status::unauthenticated("User not authenticated")),
        };

        let metadata = request.into_inner();
        debug!(
            "Received device metadata from user {}: {} {}",
            user_id, metadata.platform, metadata.device_model
        );

        // Create and store device metadata
        let metadata_model = device_metadata::ActiveModel {
            id: Set(Uuid::new_v4()),
            user_id: Set(user_id),
            platform: Set(metadata.platform),
            os_version: Set(metadata.os_version),
            app_version: Set(metadata.app_version),
            device_model: Set(metadata.device_model),
            device_language: Set(metadata.device_language),
            date_created: Set(Utc::now().fixed_offset()),
        };

        device_metadata::Entity::insert(metadata_model)
            .exec(&self.db)
            .await
            .map_err(|e| Status::internal(format!("Failed to insert device metadata: {}", e)))?;

        // Return success response
        Ok(Response::new(IngestResponse {
            success: true,
            alert_message: String::new(),
            pause_tracking: false,
            recommended_upload_interval: 300, // 5 minutes default
            processed_locations: None,
            processed_activities: None,
            processed_visits: None,
        }))
    }

    async fn submit_google_maps_timeline_export(
        &self,
        request: Request<GoogleMapsTimelineExport>,
    ) -> Result<Response<IngestResponse>, Status> {
        // Extract authentication context and user id from request
        let user_id = match request.extensions().get::<Uuid>() {
            Some(user_id) => *user_id,
            None => return Err(Status::unauthenticated("User not authenticated")),
        };

        let export = request.into_inner();

        if export.json_content.is_empty() {
            return Err(Status::invalid_argument("Empty JSON content"));
        }

        debug!(
            "Received Google Maps Timeline export from user {}: {}",
            user_id, export.export_name
        );

        // Get export timestamp or use current time
        let export_datetime = match export.export_date.as_ref() {
            Some(dt) => dt.into(),
            None => Utc::now(), // Use current time if not provided
        };

        // Parse the JSON content to extract data
        let timeline_entries: Vec<crate::io::google_maps_local_timeline::LocationHistoryEntry> =
            serde_json::from_str(&export.json_content)
                .map_err(|e| Status::invalid_argument(format!("Invalid JSON format: {}", e)))?;

        let mut locations = Vec::new();
        let mut activities = Vec::new();
        let mut visits = Vec::new();

        // Process each entry based on its type
        for entry in timeline_entries {
            match entry {
                crate::io::google_maps_local_timeline::LocationHistoryEntry::Visit(visit) => {
                    // Convert VisitEntry to VisitEvent
                    if let Some(place_location) =
                        Self::parse_place_location(&visit.visit.top_candidate.place_location)
                    {
                        let visit_event = ProtoVisitEvent {
                            latitude: place_location.0,
                            longitude: place_location.1,
                            horizontal_accuracy: 0.0, // Not provided in Google data
                            arrival: Some((&visit.start_time).into()),
                            departure: Some((&visit.end_time).into()),
                            canonical_label: visit.visit.top_candidate.semantic_type.clone(),
                            external_place_id: Some(visit.visit.top_candidate.place_id.clone()),
                        };
                        visits.push(visit_event);
                    }
                }
                crate::io::google_maps_local_timeline::LocationHistoryEntry::Activity(activity) => {
                    // Convert ActivityEntry to ActivityEvent
                    let activity_type = Self::map_google_activity_type(
                        &activity.activity.top_candidate.activity_type,
                    );

                    // Try to parse distance if available
                    let distance = activity.activity.distance_meters.parse::<f64>().ok();

                    let activity_event = ProtoActivityEvent {
                        r#type: activity_type as i32,
                        confidence: Self::parse_probability(
                            &activity.activity.top_candidate.probability,
                        ),
                        start: Some((&activity.start_time).into()),
                        end: Some((&activity.end_time).into()),
                        step_count: None, // Not provided in Google data
                        distance,
                    };
                    activities.push(activity_event);
                }
                crate::io::google_maps_local_timeline::LocationHistoryEntry::Timeline(timeline) => {
                    // Convert TimelineEntry points to LocationPoints
                    for point in &timeline.timeline_path {
                        if let Some(point_data) = point.clone().into_point_with_entry(&timeline) {
                            if let (Some(latlon), Some(date_time)) =
                                (point_data.latlon, point_data.date_time)
                            {
                                let location_point = ProtoLocationPoint {
                                    latitude: latlon.lat,
                                    longitude: latlon.lon,
                                    altitude: 0.0, // Not provided in Google data
                                    horizontal_accuracy: 0.0, // Not provided
                                    vertical_accuracy: 0.0, // Not provided
                                    speed_accuracy: 0.0, // Not provided
                                    bearing_accuracy: 0.0, // Not provided
                                    speed: 0.0,    // Not provided
                                    bearing: 0.0,  // Not provided
                                    date_time: Some(date_time),
                                    is_mock_location: false, // Assume not mocked
                                    floor_level: None,       // Not provided
                                    battery_level: 0,        // Not provided
                                    network_type: None,      // Not provided
                                };
                                locations.push(location_point);
                            }
                        }
                    }
                }
                crate::io::google_maps_local_timeline::LocationHistoryEntry::TimelineMemory(_) => {
                    // TimelineMemory entries are not currently mapped to our schema
                    debug!("Skipping TimelineMemory entry (not supported in current schema)");
                }
            }
        }

        // Create a synthetic batch with the parsed data
        let mut batch = ProtoIngestBatch {
            date_time: Some((&export_datetime).into()),
            locations,
            activities,
            visits,
            source: Some(crate::proto::ingest_batch::Source::GoogleMaps(
                crate::proto::GoogleMapsTimelineSource {
                    version: "import".to_string(),
                },
            )),
        };

        // Filter out any data that overlaps with existing records
        self.filter_batch_overlap(user_id, &mut batch).await?;

        // If batch is empty after filtering, return early with success
        if batch.locations.is_empty() && batch.activities.is_empty() && batch.visits.is_empty() {
            info!(
                "All data in Google Maps Timeline export already exists for user {}, skipping insertion",
                user_id
            );
            return Ok(Response::new(IngestResponse {
                success: true,
                alert_message: "No new data to import, all records already exist.".into(),
                pause_tracking: false,
                recommended_upload_interval: 300,
                processed_locations: Some(0),
                processed_activities: Some(0),
                processed_visits: Some(0),
            }));
        }

        // Get the latest device metadata ID for this user
        let device_metadata_id = self.get_latest_device_metadata(user_id).await?;

        // Create source info JSON for Google Maps Timeline
        let source_info = serde_json::json!({
            "type": "google_maps_timeline",
            "export_name": export.export_name,
            "export_date": export_datetime.to_rfc3339()
        });

        // Create a batch record for this import
        let batch_model = ingest_batches::ActiveModel {
            id: NotSet,
            user_id: Set(user_id),
            device_metadata_id: Set(device_metadata_id),
            batch_date_time: Set(export_datetime.fixed_offset()),
            received_date_time: Set(Utc::now().fixed_offset()),
            ready_for_processing: Set(false),
            processed: Set(Some(false)),
            source_info: Set(serde_json::to_value(&source_info).unwrap_or_default()),
        };

        let batch_result = ingest_batches::Entity::insert(batch_model)
            .exec(&self.db)
            .await
            .map_err(|e| Status::internal(format!("Failed to insert export batch: {}", e)))?;

        let batch_id = batch_result.last_insert_id;

        // Process and store the filtered location points
        if !batch.locations.is_empty() {
            self.store_location_points(batch_id, batch.locations.clone())
                .await?;
        }

        // Process and store the filtered activity events
        if !batch.activities.is_empty() {
            self.store_activity_events(batch_id, batch.activities.clone())
                .await?;
        }

        // Process and store the filtered visit events
        if !batch.visits.is_empty() {
            self.store_visit_events(batch_id, batch.visits.clone())
                .await?;
        }

        // Mark the batch as processed successfully
        let update_batch = ingest_batches::ActiveModel {
            id: Set(batch_id),
            ready_for_processing: Set(true),
            ..Default::default()
        };

        ingest_batches::Entity::update(update_batch)
            .exec(&self.db)
            .await
            .map_err(|e| Status::internal(format!("Failed to update batch status: {}", e)))?;

        info!(
            "Successfully processed Google Maps Timeline export for user {} with batch ID {}: {} locations, {} activities, {} visits",
            user_id,
            batch_id,
            batch.locations.len(),
            batch.activities.len(),
            batch.visits.len()
        );

        // Return success response with processing statistics
        Ok(Response::new(IngestResponse {
            success: true,
            alert_message: format!(
                "Successfully processed Google Maps Timeline export with {} locations, {} activities, and {} visits",
                batch.locations.len(),
                batch.activities.len(),
                batch.visits.len()
            ),
            pause_tracking: false,
            recommended_upload_interval: 300, // 5 minutes default
            processed_locations: Some(batch.locations.len() as i32),
            processed_activities: Some(batch.activities.len() as i32),
            processed_visits: Some(batch.visits.len() as i32),
        }))
    }
}
