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
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(Trips::Table).to_owned())
            .await?;

        manager
            .drop_table(Table::drop().table(Users::Table).to_owned())
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
