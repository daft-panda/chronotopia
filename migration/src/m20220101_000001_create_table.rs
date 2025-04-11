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
                    .col(date_time(Trips::Date))
                    .col(string_null(Trips::Label))
                    .col(string_null(Trips::Notes))
                    .col(date_time(Trips::LastModified))
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
                    .col(pk_uuid(IngestBatches::Id))
                    .col(uuid(IngestBatches::UserId))
                    .col(uuid_null(IngestBatches::DeviceMetadataId))
                    .col(timestamp_with_time_zone(IngestBatches::BatchDateTime))
                    .col(timestamp_with_time_zone(IngestBatches::ReceivedDateTime))
                    .col(boolean_null(IngestBatches::Processed))
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
                    .col(uuid(UserLocationsIngest::BatchId))
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
                    .col(uuid(UserActivityIngest::BatchId))
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
                    .col(uuid(UserVisitsIngest::BatchId))
                    .col(double(UserVisitsIngest::Latitude))
                    .col(double(UserVisitsIngest::Longitude))
                    .col(double_null(UserVisitsIngest::HorizontalAccuracy))
                    .col(timestamp_with_time_zone(UserVisitsIngest::ArrivalDateTime))
                    .col(timestamp_with_time_zone(
                        UserVisitsIngest::DepartureDateTime,
                    ))
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
    Date,
    Label,
    Notes,
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
}
