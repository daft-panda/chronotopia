use anyhow::Result;
use chrono::{DateTime, FixedOffset, Utc};
use log::{debug, info};
use sea_orm::ColumnTrait;
use sea_orm::{ActiveValue::Set, DatabaseConnection, EntityTrait};
use tonic::{Request, Response, Status};
use uuid::Uuid;

use crate::entity::{
    device_metadata, ingest_batches, user_activity_ingest, user_locations_ingest,
    user_visits_ingest,
};
use crate::proto::ingest_server::Ingest;
use crate::proto::{
    ActivityEvent as ProtoActivityEvent, DeviceMetadata as ProtoDeviceMetadata,
    IngestBatch as ProtoIngestBatch, IngestResponse, LocationPoint as ProtoLocationPoint,
    VisitEvent as ProtoVisitEvent,
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

    // Process and store location points
    async fn store_location_points(
        &self,
        batch_id: Uuid,
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

        // Use bulk insert for better performance
        if !location_models.is_empty() {
            use user_locations_ingest::Entity as LocationPoints;
            LocationPoints::insert_many(location_models)
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
        batch_id: Uuid,
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
            let activity_type = match event.r#type() {
                crate::proto::activity_event::ActivityType::Unknown => {
                    crate::entity::sea_orm_active_enums::ActivityType::Unknown
                }
                crate::proto::activity_event::ActivityType::Still => {
                    crate::entity::sea_orm_active_enums::ActivityType::Still
                }
                crate::proto::activity_event::ActivityType::Walking => {
                    crate::entity::sea_orm_active_enums::ActivityType::Walking
                }
                crate::proto::activity_event::ActivityType::Running => {
                    crate::entity::sea_orm_active_enums::ActivityType::Running
                }
                crate::proto::activity_event::ActivityType::InVehicle => {
                    crate::entity::sea_orm_active_enums::ActivityType::InVehicle
                }
                crate::proto::activity_event::ActivityType::OnBicycle => {
                    crate::entity::sea_orm_active_enums::ActivityType::OnBicycle
                }
                crate::proto::activity_event::ActivityType::OnFoot => {
                    crate::entity::sea_orm_active_enums::ActivityType::OnFoot
                }
                crate::proto::activity_event::ActivityType::Tilting => {
                    crate::entity::sea_orm_active_enums::ActivityType::Tilting
                }
            };

            let activity_model = user_activity_ingest::ActiveModel {
                id: sea_orm::ActiveValue::NotSet,
                batch_id: Set(batch_id),
                r#type: Set(activity_type),
                confidence: Set(event.confidence),
                start_date_time: Set(start_datetime),
                end_date_time: Set(end_datetime),
                step_count: Set(event.step_count),
                distance: Set(if event.distance.is_some() {
                    Some(event.distance.unwrap())
                } else {
                    None
                }),
            };

            activity_models.push(activity_model);
        }

        // Use bulk insert for better performance
        if !activity_models.is_empty() {
            use user_activity_ingest::Entity as ActivityEvents;
            ActivityEvents::insert_many(activity_models)
                .exec(&self.db)
                .await
                .map_err(|e| {
                    Status::internal(format!("Failed to insert activity events: {}", e))
                })?;
        }

        Ok(())
    }

    // Process and store visit events
    async fn store_visit_events(
        &self,
        batch_id: Uuid,
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
                batch_id: Set(batch_id),
                latitude: Set(event.latitude),
                longitude: Set(event.longitude),
                horizontal_accuracy: Set(if event.horizontal_accuracy != 0.0 {
                    Some(event.horizontal_accuracy)
                } else {
                    None
                }),
                arrival_date_time: Set(arrival_datetime.fixed_offset()),
                departure_date_time: Set(departure_datetime.fixed_offset()),
            };

            visit_models.push(visit_model);
        }

        // Use bulk insert for better performance
        if !visit_models.is_empty() {
            use user_visits_ingest::Entity as VisitEvents;
            VisitEvents::insert_many(visit_models)
                .exec(&self.db)
                .await
                .map_err(|e| Status::internal(format!("Failed to insert visit events: {}", e)))?;
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

        let batch = request.into_inner();
        debug!(
            "Received ingest batch from user {}: {} locations, {} activities, {} visits",
            user_id,
            batch.locations.len(),
            batch.activities.len(),
            batch.visits.len()
        );

        // Get the latest device metadata ID for this user
        let device_metadata_id = self.get_latest_device_metadata(user_id).await?;

        // Get batch timestamp
        let batch_datetime = match batch.date_time.as_ref() {
            Some(dt) => dt.into(),
            None => Utc::now(), // Use current time if not provided
        };

        // Create and save batch record
        let batch_model = ingest_batches::ActiveModel {
            id: Set(Uuid::new_v4()),
            user_id: Set(user_id),
            device_metadata_id: Set(device_metadata_id),
            batch_date_time: Set(batch_datetime.fixed_offset()),
            received_date_time: Set(Utc::now().fixed_offset()),
            processed: Set(Some(false)),
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
        }))
    }
}
