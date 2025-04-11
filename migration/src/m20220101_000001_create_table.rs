use sea_orm_migration::{
    prelude::{extension::postgres::Type, *},
    schema::*,
    sea_orm::{EnumIter, Iterable as _},
};

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .get_connection()
            .execute_unprepared("CREATE EXTENSION IF NOT EXISTS postgis;")
            .await?;

        manager
            .create_type(
                Type::create()
                    .as_enum(UserStatus::Enum)
                    .values(UserStatus::iter())
                    .to_owned(),
            )
            .await?;

        manager
            .create_type(
                Type::create()
                    .as_enum(ActivityType::Enum)
                    .values(ActivityType::iter())
                    .to_owned(),
            )
            .await?;

        manager
            .create_table(
                Table::create()
                    .table(Users::Table)
                    .if_not_exists()
                    .col(pk_uuid(Users::Id))
                    .col(string(Users::FirstName))
                    .col(string(Users::LastName))
                    .col(string_null(Users::UserName))
                    .col(string(Users::Email))
                    .col(binary(Users::Password))
                    .col(binary(Users::Salt))
                    .col(timestamp_with_time_zone(Users::DateCreated))
                    .col(timestamp_with_time_zone(Users::LastModified))
                    .col(enumeration(
                        Users::Status,
                        Alias::new("user_status"),
                        UserStatus::iter(),
                    ))
                    .to_owned(),
            )
            .await?;

        manager
            .create_table(
                Table::create()
                    .table(Trips::Table)
                    .if_not_exists()
                    .col(pk_uuid(Trips::Id))
                    .col(uuid(Trips::UserId))
                    .col(
                        ColumnDef::new(Trips::Geometry)
                            .custom(Alias::new("GEOMETRY(LINESTRING, 4326)"))
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(Trips::BoundingBox)
                            .custom(Alias::new("GEOMETRY(POLYGON, 4326)"))
                            .not_null(),
                    )
                    .col(timestamp_with_time_zone(Trips::StartTime))
                    .col(timestamp_with_time_zone(Trips::EndTime))
                    .col(double(Trips::DistanceMeters))
                    .col(binary_null(Trips::Points))
                    .col(boolean(Trips::Processed))
                    .col(json_binary(Trips::OsmWayIds)) // JSON array of OSM way IDs
                    .col(binary_null(Trips::RouteMatchTrace)) // Binary protobuf data
                    .col(string_null(Trips::Label))
                    .col(string_null(Trips::Notes))
                    .col(string_null(Trips::GeoJson))
                    .col(timestamp_with_time_zone(Trips::LastModified))
                    .foreign_key(
                        ForeignKey::create()
                            .from(Trips::Table, Trips::UserId)
                            .to(Users::Table, Users::Id)
                            .on_update(ForeignKeyAction::Cascade)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        // Add UserProcessingState table to track processing state
        manager
            .create_table(
                Table::create()
                    .table(UserProcessingState::Table)
                    .if_not_exists()
                    .col(pk_uuid(UserProcessingState::UserId))
                    .col(timestamp_with_time_zone(
                        UserProcessingState::LastProcessedTime,
                    ))
                    .col(integer(UserProcessingState::TotalTripsGenerated))
                    .col(integer(UserProcessingState::TotalVisitsDetected))
                    .col(timestamp_with_time_zone(UserProcessingState::LastUpdated))
                    .foreign_key(
                        ForeignKey::create()
                            .from(UserProcessingState::Table, UserProcessingState::UserId)
                            .to(Users::Table, Users::Id)
                            .on_update(ForeignKeyAction::Cascade)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        // Add ImportSummary table to track processing of imports
        manager
            .create_table(
                Table::create()
                    .table(ImportSummary::Table)
                    .if_not_exists()
                    .col(pk_uuid(ImportSummary::Id))
                    .col(uuid(ImportSummary::UserId))
                    .col(string(ImportSummary::ImportType))
                    .col(string_null(ImportSummary::ImportName))
                    .col(timestamp_with_time_zone(ImportSummary::ImportDateTime))
                    .col(integer(ImportSummary::LocationCount))
                    .col(integer(ImportSummary::ActivityCount))
                    .col(integer(ImportSummary::VisitCount))
                    .col(integer(ImportSummary::GeneratedTrips))
                    .col(boolean(ImportSummary::ProcessingComplete))
                    .col(timestamp_with_time_zone(ImportSummary::CreateDateTime))
                    .col(timestamp_with_time_zone_null(
                        ImportSummary::ProcessedDateTime,
                    ))
                    .foreign_key(
                        ForeignKey::create()
                            .from(ImportSummary::Table, ImportSummary::UserId)
                            .to(Users::Table, Users::Id)
                            .on_update(ForeignKeyAction::Cascade)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        // Add Device Metadata table
        manager
            .create_table(
                Table::create()
                    .table(DeviceMetadata::Table)
                    .if_not_exists()
                    .col(pk_uuid(DeviceMetadata::Id))
                    .col(uuid(DeviceMetadata::UserId))
                    .col(string(DeviceMetadata::Platform))
                    .col(string(DeviceMetadata::OsVersion))
                    .col(string(DeviceMetadata::AppVersion))
                    .col(string(DeviceMetadata::DeviceModel))
                    .col(string(DeviceMetadata::DeviceLanguage))
                    .col(timestamp_with_time_zone(DeviceMetadata::DateCreated))
                    .foreign_key(
                        ForeignKey::create()
                            .from(DeviceMetadata::Table, DeviceMetadata::UserId)
                            .to(Users::Table, Users::Id)
                            .on_update(ForeignKeyAction::Cascade)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        // Add Ingest Batch table
        manager
            .create_table(
                Table::create()
                    .table(IngestBatches::Table)
                    .if_not_exists()
                    .col(pk_auto(IngestBatches::Id))
                    .col(uuid(IngestBatches::UserId))
                    .col(uuid_null(IngestBatches::DeviceMetadataId))
                    .col(timestamp_with_time_zone(IngestBatches::BatchDateTime))
                    .col(timestamp_with_time_zone(IngestBatches::ReceivedDateTime))
                    .col(boolean_null(IngestBatches::Processed))
                    .col(json_binary(IngestBatches::SourceInfo))
                    .foreign_key(
                        ForeignKey::create()
                            .from(IngestBatches::Table, IngestBatches::UserId)
                            .to(Users::Table, Users::Id)
                            .on_update(ForeignKeyAction::Cascade)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .foreign_key(
                        ForeignKey::create()
                            .from(IngestBatches::Table, IngestBatches::DeviceMetadataId)
                            .to(DeviceMetadata::Table, DeviceMetadata::Id)
                            .on_update(ForeignKeyAction::Cascade)
                            .on_delete(ForeignKeyAction::SetNull),
                    )
                    .to_owned(),
            )
            .await?;

        // Add Location Point table - optimized for bulk inserts
        manager
            .create_table(
                Table::create()
                    .table(UserLocationsIngest::Table)
                    .if_not_exists()
                    .col(pk_auto(UserLocationsIngest::Id)) // Use auto-increment for high-speed inserts
                    .col(integer(UserLocationsIngest::BatchId))
                    .col(double(UserLocationsIngest::Latitude))
                    .col(double(UserLocationsIngest::Longitude))
                    .col(double_null(UserLocationsIngest::Altitude))
                    .col(double_null(UserLocationsIngest::HorizontalAccuracy))
                    .col(double_null(UserLocationsIngest::VerticalAccuracy))
                    .col(double_null(UserLocationsIngest::SpeedAccuracy))
                    .col(double_null(UserLocationsIngest::BearingAccuracy))
                    .col(double_null(UserLocationsIngest::Speed))
                    .col(double_null(UserLocationsIngest::Bearing))
                    .col(timestamp_with_time_zone(UserLocationsIngest::DateTime))
                    .col(boolean_null(UserLocationsIngest::IsMockLocation))
                    .col(integer_null(UserLocationsIngest::FloorLevel))
                    .col(integer_null(UserLocationsIngest::BatteryLevel))
                    .col(string_null(UserLocationsIngest::NetworkType))
                    .foreign_key(
                        ForeignKey::create()
                            .from(UserLocationsIngest::Table, UserLocationsIngest::BatchId)
                            .to(IngestBatches::Table, IngestBatches::Id)
                            .on_update(ForeignKeyAction::Cascade)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        // Add Activity Event table
        manager
            .create_table(
                Table::create()
                    .table(UserActivityIngest::Table)
                    .if_not_exists()
                    .col(pk_auto(UserActivityIngest::Id)) // Use auto-increment for high-speed inserts
                    .col(integer(UserActivityIngest::BatchId))
                    .col(enumeration(
                        UserActivityIngest::Type,
                        Alias::new("activity_type"),
                        ActivityType::iter(),
                    ))
                    .col(integer(UserActivityIngest::Confidence))
                    .col(timestamp_with_time_zone(UserActivityIngest::StartDateTime))
                    .col(timestamp_with_time_zone_null(
                        UserActivityIngest::EndDateTime,
                    ))
                    .col(big_integer_null(UserActivityIngest::StepCount))
                    .col(double_null(UserActivityIngest::Distance))
                    .foreign_key(
                        ForeignKey::create()
                            .from(UserActivityIngest::Table, UserActivityIngest::BatchId)
                            .to(IngestBatches::Table, IngestBatches::Id)
                            .on_update(ForeignKeyAction::Cascade)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        // Add Visit Event table
        manager
            .create_table(
                Table::create()
                    .table(UserVisitsIngest::Table)
                    .if_not_exists()
                    .col(pk_auto(UserVisitsIngest::Id)) // Use auto-increment for high-speed inserts
                    .col(integer(UserVisitsIngest::BatchId))
                    .col(double(UserVisitsIngest::Latitude))
                    .col(double(UserVisitsIngest::Longitude))
                    .col(double_null(UserVisitsIngest::HorizontalAccuracy))
                    .col(timestamp_with_time_zone(UserVisitsIngest::ArrivalDateTime))
                    .col(timestamp_with_time_zone(
                        UserVisitsIngest::DepartureDateTime,
                    ))
                    .col(string_null(UserVisitsIngest::CanonicalLabel))
                    .col(string_null(UserVisitsIngest::ExternalPlaceId))
                    .foreign_key(
                        ForeignKey::create()
                            .from(UserVisitsIngest::Table, UserVisitsIngest::BatchId)
                            .to(IngestBatches::Table, IngestBatches::Id)
                            .on_update(ForeignKeyAction::Cascade)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        // Create indices for better query performance
        manager
            .create_index(
                Index::create()
                    .name("idx_ingest_batch_user_id")
                    .table(IngestBatches::Table)
                    .col(IngestBatches::UserId)
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_ingest_batch_processed")
                    .table(IngestBatches::Table)
                    .col(IngestBatches::Processed)
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_location_point_batch_id")
                    .table(UserLocationsIngest::Table)
                    .col(UserLocationsIngest::BatchId)
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_location_point_datetime")
                    .table(UserLocationsIngest::Table)
                    .col(UserLocationsIngest::DateTime)
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_activity_event_batch_id")
                    .table(UserActivityIngest::Table)
                    .col(UserActivityIngest::BatchId)
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_visit_event_batch_id")
                    .table(UserVisitsIngest::Table)
                    .col(UserVisitsIngest::BatchId)
                    .to_owned(),
            )
            .await?;

        // Add index for the external place ID
        manager
            .create_index(
                Index::create()
                    .name("idx_visit_external_place_id")
                    .table(UserVisitsIngest::Table)
                    .col(UserVisitsIngest::ExternalPlaceId)
                    .to_owned(),
            )
            .await?;

        // Add indices for trips
        manager
            .create_index(
                Index::create()
                    .name("idx_trips_user_id")
                    .table(Trips::Table)
                    .col(Trips::UserId)
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_trips_start_time")
                    .table(Trips::Table)
                    .col(Trips::StartTime)
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_trips_end_time")
                    .table(Trips::Table)
                    .col(Trips::EndTime)
                    .to_owned(),
            )
            .await?;

        // Create spatial indices
        manager
            .get_connection()
            .execute_unprepared(&format!(
                "CREATE INDEX idx_trips_geometry ON {} USING GIST ({})",
                Trips::Table.to_string(),
                Trips::Geometry.to_string()
            ))
            .await?;

        manager
            .get_connection()
            .execute_unprepared(&format!(
                "CREATE INDEX idx_trips_bbox ON {} USING GIST ({})",
                Trips::Table.to_string(),
                Trips::BoundingBox.to_string()
            ))
            .await?;

        // Add indices for import summary
        manager
            .create_index(
                Index::create()
                    .name("idx_import_summary_user_id")
                    .table(ImportSummary::Table)
                    .col(ImportSummary::UserId)
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        // Drop in reverse order (child tables first)
        manager
            .drop_table(Table::drop().table(UserVisitsIngest::Table).to_owned())
            .await?;

        manager
            .drop_table(Table::drop().table(UserActivityIngest::Table).to_owned())
            .await?;

        manager
            .drop_table(Table::drop().table(UserLocationsIngest::Table).to_owned())
            .await?;

        manager
            .drop_table(Table::drop().table(ImportSummary::Table).to_owned())
            .await?;

        manager
            .drop_table(Table::drop().table(UserProcessingState::Table).to_owned())
            .await?;

        manager
            .drop_table(Table::drop().table(IngestBatches::Table).to_owned())
            .await?;

        manager
            .drop_table(Table::drop().table(DeviceMetadata::Table).to_owned())
            .await?;

        manager
            .drop_table(Table::drop().table(Trips::Table).to_owned())
            .await?;

        manager
            .drop_table(Table::drop().table(Users::Table).to_owned())
            .await?;

        manager
            .drop_type(Type::drop().if_exists().name(ActivityType::Enum).to_owned())
            .await?;

        manager
            .drop_type(Type::drop().if_exists().name(UserStatus::Enum).to_owned())
            .await
    }
}

#[derive(DeriveIden)]
enum Users {
    Table,
    Id,
    FirstName,
    LastName,
    UserName,
    Email,
    Password,
    Salt,
    DateCreated,
    LastModified,
    Status,
}

#[derive(DeriveIden, EnumIter)]
#[sea_orm(enum_name = "user_status")]
pub enum UserStatus {
    #[sea_orm(iden = "user_status")]
    Enum,
    Active,
    Expired,
    Disabled,
    Tombstone,
}

#[derive(DeriveIden)]
enum Trips {
    Table,
    Id,
    UserId,
    Geometry,
    BoundingBox,
    StartTime,
    EndTime,
    DistanceMeters,
    Points,
    Processed,
    OsmWayIds,
    RouteMatchTrace,
    Label,
    Notes,
    GeoJson,
    LastModified,
}

#[derive(DeriveIden)]
enum DeviceMetadata {
    Table,
    Id,
    UserId,
    Platform,
    OsVersion,
    AppVersion,
    DeviceModel,
    DeviceLanguage,
    DateCreated,
}

#[derive(DeriveIden)]
enum IngestBatches {
    Table,
    Id,
    UserId,
    DeviceMetadataId,
    BatchDateTime,
    ReceivedDateTime,
    Processed,
    SourceInfo,
}

#[derive(DeriveIden)]
enum UserLocationsIngest {
    Table,
    Id,
    BatchId,
    Latitude,
    Longitude,
    Altitude,
    HorizontalAccuracy,
    VerticalAccuracy,
    SpeedAccuracy,
    BearingAccuracy,
    Speed,
    Bearing,
    DateTime,
    IsMockLocation,
    FloorLevel,
    BatteryLevel,
    NetworkType,
}

#[derive(DeriveIden)]
enum UserActivityIngest {
    Table,
    Id,
    BatchId,
    Type,
    Confidence,
    StartDateTime,
    EndDateTime,
    StepCount,
    Distance,
}

#[derive(DeriveIden, EnumIter)]
#[sea_orm(enum_name = "activity_type")]
pub enum ActivityType {
    #[sea_orm(iden = "activity_type")]
    Enum,
    Unknown,
    Still,
    Walking,
    Running,
    InVehicle,
    OnBicycle,
    OnFoot,
    Tilting,
}

#[derive(DeriveIden)]
enum UserVisitsIngest {
    Table,
    Id,
    BatchId,
    Latitude,
    Longitude,
    HorizontalAccuracy,
    ArrivalDateTime,
    DepartureDateTime,
    CanonicalLabel,
    ExternalPlaceId,
}

#[derive(DeriveIden)]
enum UserProcessingState {
    Table,
    UserId,
    LastProcessedTime,
    TotalTripsGenerated,
    TotalVisitsDetected,
    LastUpdated,
}

#[derive(DeriveIden)]
enum ImportSummary {
    Table,
    Id,
    UserId,
    ImportType,
    ImportName,
    ImportDateTime,
    LocationCount,
    ActivityCount,
    VisitCount,
    GeneratedTrips,
    ProcessingComplete,
    CreateDateTime,
    ProcessedDateTime,
}
