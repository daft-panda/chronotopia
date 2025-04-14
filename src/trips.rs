use chrono::FixedOffset;
use geo::{BoundingRect, Polygon};
use geo_postgis::FromPostgis;
use log::{debug, error, info, warn};
use sea_orm::{ColumnTrait, DatabaseConnection, EntityTrait, QueryFilter, QueryOrder, QuerySelect};
use std::sync::Arc;
use tonic::{Request, Response, Status};
use uuid::Uuid;

use crate::ROUTE_MATCHER_CONFIG;
use crate::entity::sea_orm_active_enums::ActivityType;
use crate::entity::{
    import_summary, trips, user_activity_ingest, user_processing_state, user_visits_ingest,
};
use crate::proto::trips_server::Trips;
use crate::proto::{self, DateTime, Point, Points};
use crate::route_matcher::RouteMatcher;
use crate::trip_processor::TripProcessor;

/// Service for handling trip-related API requests
pub struct TripsService {
    db: DatabaseConnection,
    trip_processor: Arc<TripProcessor>,
}

impl TripsService {
    /// Create a new TripsService
    pub fn new(db: DatabaseConnection, trip_processor: Arc<TripProcessor>) -> Self {
        Self { db, trip_processor }
    }

    /// Convert a database trip model to a proto trip model
    async fn db_trip_to_proto(&self, db_trip: trips::Model) -> Result<proto::Trip, Status> {
        let points: Vec<Point> = match db_trip.points {
            Some(v) => {
                let points: Points = prost::Message::decode(v.as_slice())
                    .map_err(|e| Status::internal("invalid points in db trip"))?;
                points.points
            }
            None => vec![],
        };

        // Convert basic trip fields
        let mut proto_trip = proto::Trip {
            id: Some(proto::Uuid {
                value: db_trip.id.to_string(),
            }),
            start_time: Some(DateTime::from(&db_trip.start_time)),
            end_time: Some(DateTime::from(&db_trip.end_time)),
            distance_meters: db_trip.distance_meters,
            points,
            processed: db_trip.processed,
            matched_segments: Vec::new(),
            geojson: db_trip.geo_json.unwrap_or_default(),
            visits: Vec::new(),
            activities: Vec::new(),
            label: db_trip.label,
            notes: db_trip.notes,
            route_match_trace: None, // We'll load this separately if needed
        };

        // Load visits that occurred during this trip's time range
        let visits = user_visits_ingest::Entity::find()
            .filter(
                user_visits_ingest::Column::ArrivalDateTime
                    .gte(db_trip.start_time)
                    .and(user_visits_ingest::Column::DepartureDateTime.lte(db_trip.end_time)),
            )
            .all(&self.db)
            .await
            .map_err(|e| {
                error!("Error loading trip visits: {}", e);
                Status::internal("Failed to load trip data")
            })?;

        // Convert visits to proto format
        for visit in visits {
            let proto_visit = proto::VisitEvent {
                latitude: visit.latitude,
                longitude: visit.longitude,
                horizontal_accuracy: visit.horizontal_accuracy.unwrap_or(0.0),
                arrival: Some(DateTime::from(&visit.arrival_date_time)),
                departure: Some(DateTime::from(&visit.departure_date_time)),
                canonical_label: visit.canonical_label.unwrap_or_default(),
                external_place_id: visit.external_place_id,
            };
            proto_trip.visits.push(proto_visit);
        }

        // Load activities that occurred during this trip's time range
        let activities = user_activity_ingest::Entity::find()
            .filter(
                user_activity_ingest::Column::StartDateTime
                    .gte(db_trip.start_time)
                    .and(user_activity_ingest::Column::StartDateTime.lte(db_trip.end_time)),
            )
            .all(&self.db)
            .await
            .map_err(|e| {
                error!("Error loading trip activities: {}", e);
                Status::internal("Failed to load trip data")
            })?;

        // Convert activities to proto format
        for activity in activities {
            let activity_type = match activity.r#type {
                ActivityType::Unknown => proto::activity_event::ActivityType::Unknown,
                ActivityType::Still => proto::activity_event::ActivityType::Still,
                ActivityType::Walking => proto::activity_event::ActivityType::Walking,
                ActivityType::Running => proto::activity_event::ActivityType::Running,
                ActivityType::InVehicle => proto::activity_event::ActivityType::InVehicle,
                ActivityType::OnBicycle => proto::activity_event::ActivityType::OnBicycle,
                ActivityType::OnFoot => proto::activity_event::ActivityType::OnFoot,
                ActivityType::Tilting => proto::activity_event::ActivityType::Tilting,
                _ => proto::activity_event::ActivityType::Unknown,
            };

            let proto_activity = proto::ActivityEvent {
                r#type: activity_type as i32,
                confidence: activity.confidence,
                start: Some(DateTime::from(&activity.start_date_time)),
                end: activity.end_date_time.map(|dt| DateTime::from(&dt)),
                step_count: activity.step_count,
                distance: activity.distance,
            };
            proto_trip.activities.push(proto_activity);
        }

        Ok(proto_trip)
    }
}

#[tonic::async_trait]
impl Trips for TripsService {
    async fn get_trips_for_user(
        &self,
        request: Request<proto::GetTripsForUserRequest>,
    ) -> Result<Response<proto::GetTripsForUserResponse>, Status> {
        // Extract authentication context and user id from request
        let user_id = match request.extensions().get::<Uuid>() {
            Some(user_id) => *user_id,
            None => return Err(Status::unauthenticated("User not authenticated")),
        };

        let req = request.into_inner();

        debug!("Getting trips for user {}", user_id);

        // Query trips from the database
        let mut trip_query = trips::Entity::find().filter(trips::Column::UserId.eq(user_id));

        // Apply optional filters from the request
        if let Some(from_date) = req.from_date {
            // Convert proto DateTime to database datetime
            let from_datetime = chrono::DateTime::<FixedOffset>::from(&from_date);
            trip_query = trip_query.filter(trips::Column::StartTime.gte(from_datetime));
        }

        if let Some(to_date) = req.to_date {
            // Convert proto DateTime to database datetime
            let to_datetime = chrono::DateTime::<FixedOffset>::from(&to_date);
            trip_query = trip_query.filter(trips::Column::EndTime.lte(to_datetime));
        }

        // Apply ordering
        trip_query = trip_query.order_by_desc(trips::Column::StartTime);

        // Apply limit if provided
        if req.limit > 0 {
            trip_query = trip_query.limit(req.limit as u64);
        }

        // Execute the query
        let trips_result = trip_query.all(&self.db).await.map_err(|e| {
            error!("Database error when fetching trips: {}", e);
            Status::internal("Failed to fetch trips")
        })?;

        // Convert to proto models
        let mut proto_trips = Vec::new();
        for db_trip in trips_result {
            let proto_trip = self.db_trip_to_proto(db_trip).await?;
            proto_trips.push(proto_trip);
        }

        info!("Found {} trips for user {}", proto_trips.len(), user_id);

        // Return the response
        Ok(Response::new(proto::GetTripsForUserResponse {
            trips: Some(proto::TripList { trips: proto_trips }),
        }))
    }

    async fn get_user_processing_state(
        &self,
        request: Request<()>,
    ) -> Result<Response<proto::GetUserProcessingStateResponse>, Status> {
        // Extract authentication context and user id from request
        let user_id = match request.extensions().get::<Uuid>() {
            Some(user_id) => *user_id,
            None => return Err(Status::unauthenticated("User not authenticated")),
        };

        debug!("Getting processing state for user {}", user_id);

        // Get the user's processing state
        let processing_state = user_processing_state::Entity::find_by_id(user_id)
            .one(&self.db)
            .await
            .map_err(|e| {
                error!("Database error when fetching processing state: {}", e);
                Status::internal("Failed to fetch processing state")
            })?;

        // Build the response
        let state = if let Some(state) = processing_state {
            proto::TripProcessingState {
                last_processed_time: Some(DateTime::from(&state.last_processed_time)),
                total_trips_generated: state.total_trips_generated as u32,
                total_visits_detected: state.total_visits_detected as u32,
                last_updated: Some(DateTime::from(&state.last_updated)),
            }
        } else {
            // Default state if not found
            proto::TripProcessingState {
                last_processed_time: Some(DateTime::from(&chrono::Utc::now())),
                total_trips_generated: 0,
                total_visits_detected: 0,
                last_updated: Some(DateTime::from(&chrono::Utc::now())),
            }
        };

        // Get the import summaries for this user
        let import_summaries = import_summary::Entity::find()
            .filter(import_summary::Column::UserId.eq(user_id))
            .order_by_desc(import_summary::Column::ImportDateTime)
            .all(&self.db)
            .await
            .map_err(|e| {
                error!("Database error when fetching import summaries: {}", e);
                Status::internal("Failed to fetch import summaries")
            })?;

        // Convert to proto models
        let imports = import_summaries
            .into_iter()
            .map(|i| proto::ImportSummary {
                id: Some(proto::Uuid {
                    value: i.id.to_string(),
                }),
                import_type: i.import_type,
                import_date_time: Some(DateTime::from(&i.import_date_time)),
                location_count: i.location_count as u32,
                activity_count: i.activity_count as u32,
                visit_count: i.visit_count as u32,
                generated_trips: i.generated_trips as u32,
                processing_complete: i.processing_complete,
                create_date_time: Some(DateTime::from(&i.create_date_time)),
                processed_date_time: i.processed_date_time.map(|dt| DateTime::from(&dt)),
                import_name: i.import_name.unwrap_or_default(),
            })
            .collect();

        // Return the response
        Ok(Response::new(proto::GetUserProcessingStateResponse {
            state: Some(state),
            imports,
        }))
    }

    async fn trigger_processing(
        &self,
        request: Request<()>,
    ) -> Result<Response<proto::TriggerProcessingResponse>, Status> {
        // Extract authentication context and user id from request
        let user_id = match request.extensions().get::<Uuid>() {
            Some(user_id) => *user_id,
            None => return Err(Status::unauthenticated("User not authenticated")),
        };

        info!("Triggering processing for user {}", user_id);

        // Spawn a task to process the user's batches
        let trip_processor = self.trip_processor.clone();
        let user_id_clone = user_id;
        tokio::spawn(async move {
            if let Err(e) = trip_processor.process_user_batches(user_id_clone).await {
                error!("Error processing batches for user {}: {}", user_id_clone, e);
            }
        });

        // Return success response
        Ok(Response::new(proto::TriggerProcessingResponse {
            success: true,
            message: "Processing triggered successfully. It may take a few minutes to complete."
                .to_string(),
        }))
    }

    async fn get_trip_details(
        &self,
        request: Request<proto::GetTripDetailsRequest>,
    ) -> Result<Response<proto::GetTripDetailsResponse>, Status> {
        // Extract authentication context and user id from request
        let user_id = match request.extensions().get::<Uuid>() {
            Some(user_id) => *user_id,
            None => return Err(Status::unauthenticated("User not authenticated")),
        };

        let req = request.into_inner();
        let trip_uuid = match req.trip_id {
            Some(uuid) => Uuid::parse_str(&uuid.value)
                .map_err(|_| Status::invalid_argument("Invalid trip UUID"))?,
            None => return Err(Status::invalid_argument("Trip ID is required")),
        };

        debug!(
            "Getting trip details for trip {} by user {}",
            trip_uuid, user_id
        );

        // Get the trip
        let trip = trips::Entity::find_by_id(trip_uuid)
            .one(&self.db)
            .await
            .map_err(|e| {
                error!("Database error when fetching trip: {}", e);
                Status::internal("Failed to fetch trip")
            })?;

        // Check if the trip exists and belongs to the user
        let trip = match trip {
            Some(t) if t.user_id == user_id => t,
            Some(_) => {
                return Err(Status::permission_denied(
                    "Trip does not belong to the authenticated user",
                ));
            }
            None => return Err(Status::not_found("Trip not found")),
        };

        // Parse OSM way IDs
        let osm_way_ids: Vec<u64> = match serde_json::from_value(trip.osm_way_ids.clone()) {
            Ok(ids) => ids,
            Err(e) => {
                warn!("Error parsing OSM way IDs: {}", e);
                Vec::new()
            }
        };

        let mut route_matcher: RouteMatcher =
            RouteMatcher::new(ROUTE_MATCHER_CONFIG.clone()).unwrap();
        let bbox: Option<Polygon<f64>> = match trip.bounding_box {
            postgis::ewkb::GeometryT::<postgis::ewkb::Point>::Polygon(p) => {
                std::option::Option::<geo::Polygon>::from_postgis(&p)
            }
            _ => None,
        };

        // Fetch OSM way metadata
        let osm_metadata = match route_matcher
            .tile_loader
            .get_way_metadata_from_bbox(bbox.unwrap().bounding_rect().unwrap(), &osm_way_ids)
        {
            Ok(metadata) => metadata,
            Err(e) => {
                warn!("Error fetching OSM metadata: {}", e);
                Vec::new() // Continue even if metadata fetch fails
            }
        };

        let points: Vec<Point> = match trip.points {
            Some(v) => {
                let points: Points = prost::Message::decode(v.as_slice())
                    .map_err(|e| Status::internal("invalid points in db trip"))?;
                points.points
            }
            None => vec![],
        };

        // Convert basic trip fields
        let mut proto_trip = proto::Trip {
            id: Some(proto::Uuid {
                value: trip.id.to_string(),
            }),
            start_time: Some(DateTime::from(&trip.start_time)),
            end_time: Some(DateTime::from(&trip.end_time)),
            distance_meters: trip.distance_meters,
            points,
            processed: trip.processed,
            matched_segments: Vec::new(),
            geojson: trip.geo_json.unwrap_or_default(),
            visits: Vec::new(),
            activities: Vec::new(),
            label: trip.label,
            notes: trip.notes,
            route_match_trace: None,
        };

        // Deserialize route match trace if available
        if let Some(trace_data) = trip.route_match_trace {
            match prost::Message::decode(trace_data.as_slice()) {
                Ok(trace) => {
                    let trace: proto::RouteMatchTrace = trace;
                    proto_trip.route_match_trace = Some(trace);
                }
                Err(e) => {
                    warn!("Error decoding route match trace: {}", e);
                    // Continue without the trace
                }
            }
        }

        // Create matched segments from OSM way IDs and metadata
        for way_id in osm_way_ids {
            // Find metadata for this way ID
            let metadata = osm_metadata.iter().find(|m| m.osm_way_id == way_id);

            // Create a basic segment
            let segment = proto::RoadSegment {
                id: 0, // We don't have segment IDs anymore, just way IDs
                osm_way_id: way_id,
                coordinates: Vec::new(), // Not including detailed coordinates
                is_oneway: metadata.map(|m| m.is_oneway).unwrap_or(false),
                highway_type: metadata.map(|m| m.highway_type.clone()).unwrap_or_default(),
                connections: Vec::new(), // Not including connections
                name: metadata.and_then(|m| m.name.clone()),
                interim_start_idx: None,
                interim_end_idx: None,
                full_coordinates: Vec::new(),
            };

            proto_trip.matched_segments.push(segment);
        }

        // Load visits that occurred during this trip's time range
        let visits = user_visits_ingest::Entity::find()
            .filter(
                user_visits_ingest::Column::ArrivalDateTime
                    .gte(trip.start_time)
                    .and(user_visits_ingest::Column::DepartureDateTime.lte(trip.end_time)),
            )
            .all(&self.db)
            .await
            .map_err(|e| {
                error!("Error loading trip visits: {}", e);
                Status::internal("Failed to load trip data")
            })?;

        // Convert visits to proto format
        for visit in visits {
            let proto_visit = proto::VisitEvent {
                latitude: visit.latitude,
                longitude: visit.longitude,
                horizontal_accuracy: visit.horizontal_accuracy.unwrap_or(0.0),
                arrival: Some(DateTime::from(&visit.arrival_date_time)),
                departure: Some(DateTime::from(&visit.departure_date_time)),
                canonical_label: visit.canonical_label.unwrap_or_default(),
                external_place_id: visit.external_place_id,
            };
            proto_trip.visits.push(proto_visit);
        }

        // Load activities that occurred during this trip's time range
        let activities = user_activity_ingest::Entity::find()
            .filter(
                user_activity_ingest::Column::StartDateTime
                    .gte(trip.start_time)
                    .and(user_activity_ingest::Column::StartDateTime.lte(trip.end_time)),
            )
            .all(&self.db)
            .await
            .map_err(|e| {
                error!("Error loading trip activities: {}", e);
                Status::internal("Failed to load trip data")
            })?;

        // Convert activities to proto format
        for activity in activities {
            let activity_type = match activity.r#type {
                ActivityType::Unknown => proto::activity_event::ActivityType::Unknown,
                ActivityType::Still => proto::activity_event::ActivityType::Still,
                ActivityType::Walking => proto::activity_event::ActivityType::Walking,
                ActivityType::Running => proto::activity_event::ActivityType::Running,
                ActivityType::InVehicle => proto::activity_event::ActivityType::InVehicle,
                ActivityType::OnBicycle => proto::activity_event::ActivityType::OnBicycle,
                ActivityType::OnFoot => proto::activity_event::ActivityType::OnFoot,
                ActivityType::Tilting => proto::activity_event::ActivityType::Tilting,
                _ => proto::activity_event::ActivityType::Unknown,
            };

            let proto_activity = proto::ActivityEvent {
                r#type: activity_type as i32,
                confidence: activity.confidence,
                start: Some(DateTime::from(&activity.start_date_time)),
                end: activity.end_date_time.map(|dt| DateTime::from(&dt)),
                step_count: activity.step_count,
                distance: activity.distance,
            };
            proto_trip.activities.push(proto_activity);
        }

        // Return the response
        Ok(Response::new(proto::GetTripDetailsResponse {
            trip: Some(proto_trip),
        }))
    }
}
