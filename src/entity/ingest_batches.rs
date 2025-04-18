//! `SeaORM` Entity, @generated by sea-orm-codegen 1.1.8

use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Eq)]
#[sea_orm(table_name = "ingest_batches")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i32,
    pub user_id: Uuid,
    pub device_metadata_id: Option<Uuid>,
    pub batch_date_time: DateTimeWithTimeZone,
    pub received_date_time: DateTimeWithTimeZone,
    pub ready_for_processing: bool,
    pub processed: Option<bool>,
    #[sea_orm(column_type = "JsonBinary")]
    pub source_info: Json,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::device_metadata::Entity",
        from = "Column::DeviceMetadataId",
        to = "super::device_metadata::Column::Id",
        on_update = "Cascade",
        on_delete = "SetNull"
    )]
    DeviceMetadata,
    #[sea_orm(has_many = "super::user_activity_ingest::Entity")]
    UserActivityIngest,
    #[sea_orm(has_many = "super::user_locations_ingest::Entity")]
    UserLocationsIngest,
    #[sea_orm(has_many = "super::user_visits_ingest::Entity")]
    UserVisitsIngest,
    #[sea_orm(
        belongs_to = "super::users::Entity",
        from = "Column::UserId",
        to = "super::users::Column::Id",
        on_update = "Cascade",
        on_delete = "Cascade"
    )]
    Users,
}

impl Related<super::device_metadata::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::DeviceMetadata.def()
    }
}

impl Related<super::user_activity_ingest::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::UserActivityIngest.def()
    }
}

impl Related<super::user_locations_ingest::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::UserLocationsIngest.def()
    }
}

impl Related<super::user_visits_ingest::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::UserVisitsIngest.def()
    }
}

impl Related<super::users::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Users.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
