use anyhow::Result;
use argon2::{
    Argon2,
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString, rand_core::OsRng},
};
use chrono::{DateTime, Utc};
use email_address::EmailAddress;
use log::error;
use rand::{Rng, distr::Alphanumeric};
use sea_orm::{
    ActiveModelTrait, ColumnTrait, Condition, DatabaseConnection, DeleteResult, EntityTrait,
    PaginatorTrait, QueryFilter, QueryOrder, Set,
};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};
use uuid::Uuid;

use crate::entity::{users, users::Entity as Users};
use crate::proto::{
    ChangePasswordRequest, CreateUserRequest, ListUsersRequest, RequestPasswordResetRequest,
    ResetPasswordRequest, UpdateUserRequest, User, UserRequest, Users as ProtoUsers,
    ValidateTokenRequest, user_management_server::UserManagement,
};

// Structure to store password reset tokens
#[derive(Debug, Clone)]
struct PasswordResetToken {
    token: String,
    created_at: DateTime<Utc>,
}

// Cache for password reset tokens
type TokenCache = Arc<RwLock<std::collections::HashMap<String, PasswordResetToken>>>;

#[derive(Clone)]
pub struct UserManagementService {
    db: DatabaseConnection,
    tokens: TokenCache,
    token_expiry: Duration,
}

impl UserManagementService {
    pub fn new(db: DatabaseConnection) -> Self {
        Self {
            db,
            tokens: Arc::new(RwLock::new(std::collections::HashMap::new())),
            token_expiry: Duration::from_secs(24 * 60 * 60), // 24 hours
        }
    }

    // Helper to convert DB model to Proto user
    fn model_to_proto(model: users::Model) -> User {
        let created = SystemTime::from(DateTime::<Utc>::from_naive_utc_and_offset(
            model.date_created.naive_utc(),
            Utc,
        ));

        let modified = SystemTime::from(DateTime::<Utc>::from_naive_utc_and_offset(
            model.last_modified.naive_utc(),
            Utc,
        ));

        User {
            id: model.id.to_string(),
            first_name: model.first_name,
            last_name: model.last_name,
            user_name: model.user_name,
            email: model.email,
            date_created: Some(prost_types::Timestamp::from(created)),
            last_modified: Some(prost_types::Timestamp::from(modified)),
            active: true, // Assuming all users are active by default
        }
    }

    // Helper to hash a password with Argon2
    fn hash_password(password: &str) -> Result<(Vec<u8>, Vec<u8>), Status> {
        let salt = SaltString::generate(&mut OsRng);

        // Initialize Argon2 with default parameters
        let argon2 = Argon2::default();

        // Hash the password
        let password_hash = argon2
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| Status::internal(format!("Failed to hash password: {}", e)))?;

        // Extract the salt and hash
        let salt_bytes = salt.as_str().as_bytes().to_vec();
        let hash_bytes = password_hash.to_string().as_bytes().to_vec();

        Ok((hash_bytes, salt_bytes))
    }

    // Helper to verify a password
    fn verify_password(password: &str, hash: &[u8], salt: &[u8]) -> Result<bool, Status> {
        let salt_str = std::str::from_utf8(salt)
            .map_err(|e| Status::internal(format!("Failed to convert salt: {}", e)))?;

        let hash_str = std::str::from_utf8(hash)
            .map_err(|e| Status::internal(format!("Failed to convert hash: {}", e)))?;

        let parsed_hash = PasswordHash::new(hash_str)
            .map_err(|e| Status::internal(format!("Failed to parse hash: {}", e)))?;

        let argon2 = Argon2::default();

        Ok(argon2
            .verify_password(password.as_bytes(), &parsed_hash)
            .is_ok())
    }

    // Generate a random token for password reset
    fn generate_reset_token(&self) -> String {
        rand::rng()
            .sample_iter(&Alphanumeric)
            .take(32)
            .map(char::from)
            .collect()
    }

    // Helper to validate email
    fn validate_email(email: &str) -> Result<(), Status> {
        if !EmailAddress::is_valid(email) {
            return Err(Status::invalid_argument("Invalid email address"));
        }
        Ok(())
    }

    // Helper to validate password strength
    fn validate_password(password: &str) -> Result<(), Status> {
        if password.len() < 8 {
            return Err(Status::invalid_argument(
                "Password must be at least 8 characters long",
            ));
        }

        // Check for at least one number and one letter
        let has_number = password.chars().any(|c| c.is_numeric());
        let has_letter = password.chars().any(|c| c.is_alphabetic());

        if !has_number || !has_letter {
            return Err(Status::invalid_argument(
                "Password must contain at least one letter and one number",
            ));
        }

        Ok(())
    }

    // Helper to send password reset email (you would implement actual email sending)
    async fn send_reset_email(
        &self,
        email: &str,
        user_id: &str,
        token: &str,
    ) -> Result<(), Status> {
        unimplemented!();
    }
}

#[tonic::async_trait]
impl UserManagement for UserManagementService {
    async fn create_user(
        &self,
        request: Request<CreateUserRequest>,
    ) -> Result<Response<User>, Status> {
        let req = request.into_inner();

        // Validate email
        UserManagementService::validate_email(&req.email)?;

        // Validate password
        UserManagementService::validate_password(&req.password)?;

        // Check if email already exists
        let email_exists = Users::find()
            .filter(users::Column::Email.eq(&req.email))
            .count(&self.db)
            .await
            .map_err(|e| Status::internal(format!("Database error: {}", e)))?
            > 0;

        if email_exists {
            return Err(Status::already_exists("Email already in use"));
        }

        // Hash the password
        let (password_hash, salt) = UserManagementService::hash_password(&req.password)?;

        // Create new user
        let now = chrono::Utc::now().fixed_offset();
        let user_id = Uuid::new_v4();

        let user = users::ActiveModel {
            id: Set(user_id),
            first_name: Set(req.first_name),
            last_name: Set(req.last_name),
            user_name: Set(req.user_name),
            email: Set(req.email),
            password: Set(password_hash),
            salt: Set(salt),
            date_created: Set(now),
            last_modified: Set(now),
            status: Set(crate::entity::sea_orm_active_enums::UserStatus::Active),
        };

        // Insert into database
        let result = user
            .insert(&self.db)
            .await
            .map_err(|e| Status::internal(format!("Failed to create user: {}", e)))?;

        // Return the created user
        Ok(Response::new(Self::model_to_proto(result)))
    }

    async fn get_user(&self, request: Request<UserRequest>) -> Result<Response<User>, Status> {
        let user_id = request.into_inner().id;

        // Parse UUID
        let uuid = Uuid::parse_str(&user_id)
            .map_err(|_| Status::invalid_argument("Invalid user ID format"))?;

        // Find user in database
        let user = Users::find_by_id(uuid)
            .one(&self.db)
            .await
            .map_err(|e| Status::internal(format!("Database error: {}", e)))?
            .ok_or_else(|| Status::not_found("User not found"))?;

        // Return user
        Ok(Response::new(Self::model_to_proto(user)))
    }

    async fn list_users(
        &self,
        request: Request<ListUsersRequest>,
    ) -> Result<Response<ProtoUsers>, Status> {
        let req = request.into_inner();

        // Start building the query
        let mut query = Users::find();

        // Apply search filter if provided
        if let Some(search_term) = req.search_term {
            if !search_term.is_empty() {
                query = query.filter(
                    Condition::any()
                        .add(users::Column::FirstName.contains(&search_term))
                        .add(users::Column::LastName.contains(&search_term))
                        .add(users::Column::Email.contains(&search_term)),
                );
            }
        }

        // Apply sorting
        if let Some(sort_by) = req.sort_by {
            let column = match sort_by.as_str() {
                "firstName" => users::Column::FirstName,
                "lastName" => users::Column::LastName,
                "email" => users::Column::Email,
                "dateCreated" => users::Column::DateCreated,
                _ => users::Column::LastModified,
            };

            if req.sort_desc.unwrap_or(false) {
                query = query.order_by_desc(column);
            } else {
                query = query.order_by_asc(column);
            }
        } else {
            // Default sorting by last_modified
            query = query.order_by_desc(users::Column::LastModified);
        }

        // Get total count for pagination
        let total_count = query
            .clone()
            .count(&self.db)
            .await
            .map_err(|e| Status::internal(format!("Database error: {}", e)))?;

        // Apply pagination
        let page = req.page.unwrap_or(1);
        let page_size: u64 = req.page_size.unwrap_or(10).into();
        let total_pages = (total_count as f64 / page_size as f64).ceil() as u32;

        // Get paginated results
        let paginator = query.paginate(&self.db, page_size);
        let users = paginator
            .fetch_page((page - 1) as u64)
            .await
            .map_err(|e| Status::internal(format!("Database error: {}", e)))?;

        // Convert to proto users
        let proto_users = users.into_iter().map(Self::model_to_proto).collect();

        // Return response
        Ok(Response::new(ProtoUsers {
            users: proto_users,
            total_count: total_count as u32,
            page,
            total_pages,
        }))
    }

    async fn update_user(
        &self,
        request: Request<UpdateUserRequest>,
    ) -> Result<Response<User>, Status> {
        let req = request.into_inner();

        // Parse UUID
        let uuid = Uuid::parse_str(&req.id)
            .map_err(|_| Status::invalid_argument("Invalid user ID format"))?;

        // Find user in database
        let user = Users::find_by_id(uuid)
            .one(&self.db)
            .await
            .map_err(|e| Status::internal(format!("Database error: {}", e)))?
            .ok_or_else(|| Status::not_found("User not found"))?;

        // Prepare update model
        let mut user_model: users::ActiveModel = user.clone().into();

        // Update fields that were provided
        if let Some(first_name) = req.first_name {
            user_model.first_name = Set(first_name);
        }

        if let Some(last_name) = req.last_name {
            user_model.last_name = Set(last_name);
        }

        if let Some(user_name) = req.user_name {
            user_model.user_name = Set(Some(user_name));
        }

        if let Some(email) = req.email {
            // Validate email
            UserManagementService::validate_email(&email)?;

            // Check if email already exists for another user
            let email_exists = Users::find()
                .filter(users::Column::Email.eq(&email))
                .filter(users::Column::Id.ne(uuid))
                .count(&self.db)
                .await
                .map_err(|e| Status::internal(format!("Database error: {}", e)))?
                > 0;

            if email_exists {
                return Err(Status::already_exists(
                    "Email already in use by another user",
                ));
            }

            user_model.email = Set(email);
        }

        // Always update last_modified
        user_model.last_modified = Set(chrono::Utc::now().fixed_offset());

        // Save changes
        let updated_user = user_model
            .update(&self.db)
            .await
            .map_err(|e| Status::internal(format!("Failed to update user: {}", e)))?;

        // Return updated user
        Ok(Response::new(Self::model_to_proto(updated_user)))
    }

    async fn delete_user(&self, request: Request<UserRequest>) -> Result<Response<()>, Status> {
        let user_id = request.into_inner().id;

        // Parse UUID
        let uuid = Uuid::parse_str(&user_id)
            .map_err(|_| Status::invalid_argument("Invalid user ID format"))?;

        // Delete user
        let result: DeleteResult = Users::delete_by_id(uuid)
            .exec(&self.db)
            .await
            .map_err(|e| Status::internal(format!("Failed to delete user: {}", e)))?;

        if result.rows_affected == 0 {
            return Err(Status::not_found("User not found"));
        }

        Ok(Response::new(()))
    }

    async fn change_password(
        &self,
        request: Request<ChangePasswordRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();

        // Parse UUID
        let uuid = Uuid::parse_str(&req.id)
            .map_err(|_| Status::invalid_argument("Invalid user ID format"))?;

        // Find user
        let user = Users::find_by_id(uuid)
            .one(&self.db)
            .await
            .map_err(|e| Status::internal(format!("Database error: {}", e)))?
            .ok_or_else(|| Status::not_found("User not found"))?;

        // Verify current password
        let password_valid = UserManagementService::verify_password(
            &req.current_password,
            &user.password,
            &user.salt,
        )?;

        if !password_valid {
            return Err(Status::permission_denied("Current password is incorrect"));
        }

        // Validate new password
        UserManagementService::validate_password(&req.new_password)?;

        // Hash the new password
        let (password_hash, salt) = UserManagementService::hash_password(&req.new_password)?;

        // Update password
        let mut user_model: users::ActiveModel = user.into();
        user_model.password = Set(password_hash);
        user_model.salt = Set(salt);
        user_model.last_modified = Set(chrono::Utc::now().fixed_offset());

        // Save changes
        user_model
            .update(&self.db)
            .await
            .map_err(|e| Status::internal(format!("Failed to update password: {}", e)))?;

        Ok(Response::new(()))
    }

    async fn reset_password(
        &self,
        request: Request<ResetPasswordRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();

        // Parse UUID
        let uuid = Uuid::parse_str(&req.id)
            .map_err(|_| Status::invalid_argument("Invalid user ID format"))?;

        // Find user
        let user = Users::find_by_id(uuid)
            .one(&self.db)
            .await
            .map_err(|e| Status::internal(format!("Database error: {}", e)))?
            .ok_or_else(|| Status::not_found("User not found"))?;

        // If token is provided, verify it
        if let Some(token) = &req.token {
            let tokens = self.tokens.read().await;

            let stored_token = tokens.get(&req.id).ok_or_else(|| {
                Status::invalid_argument("No password reset token found for this user")
            })?;

            // Check if token is expired
            let now = chrono::Utc::now();
            if now.timestamp() - stored_token.created_at.timestamp()
                > self.token_expiry.as_secs() as i64
            {
                return Err(Status::deadline_exceeded(
                    "Password reset token has expired",
                ));
            }

            // Check if token matches
            if *token != stored_token.token {
                return Err(Status::invalid_argument("Invalid password reset token"));
            }
        }

        // Validate new password
        UserManagementService::validate_password(&req.new_password)?;

        // Hash the new password
        let (password_hash, salt) = UserManagementService::hash_password(&req.new_password)?;

        // Update password
        let mut user_model: users::ActiveModel = user.into();
        user_model.password = Set(password_hash);
        user_model.salt = Set(salt);
        user_model.last_modified = Set(chrono::Utc::now().fixed_offset());

        // Save changes
        user_model
            .update(&self.db)
            .await
            .map_err(|e| Status::internal(format!("Failed to update password: {}", e)))?;

        // If token was used, remove it
        if req.token.is_some() {
            let mut tokens = self.tokens.write().await;
            tokens.remove(&req.id);
        }

        Ok(Response::new(()))
    }

    async fn request_password_reset(
        &self,
        request: Request<RequestPasswordResetRequest>,
    ) -> Result<Response<()>, Status> {
        let email = request.into_inner().email;

        // Validate email
        UserManagementService::validate_email(&email)?;

        // Find user by email
        let user = Users::find()
            .filter(users::Column::Email.eq(&email))
            .one(&self.db)
            .await
            .map_err(|e| Status::internal(format!("Database error: {}", e)))?;

        // If user not found, don't inform the client (security)
        if let Some(user) = user {
            // Generate reset token
            let token = self.generate_reset_token();
            let user_id = user.id.to_string();

            // Store token
            {
                let mut tokens = self.tokens.write().await;
                tokens.insert(
                    user_id.clone(),
                    PasswordResetToken {
                        token: token.clone(),
                        created_at: chrono::Utc::now(),
                    },
                );
            }

            // Send reset email
            if let Err(e) = self.send_reset_email(&email, &user_id, &token).await {
                error!("Failed to send reset email: {}", e);
                // Don't return the error to the client for security reasons
            }
        }

        // Always return success to avoid email enumeration
        Ok(Response::new(()))
    }

    async fn validate_reset_token(
        &self,
        request: Request<ValidateTokenRequest>,
    ) -> Result<Response<bool>, Status> {
        let req = request.into_inner();

        let tokens = self.tokens.read().await;

        let stored_token = tokens.get(&req.id);

        if let Some(stored_token) = stored_token {
            // Check if token is expired
            let now = chrono::Utc::now();
            if now.timestamp() - stored_token.created_at.timestamp()
                > self.token_expiry.as_secs() as i64
            {
                return Ok(Response::new(false));
            }

            // Check if token matches
            let valid = req.token == stored_token.token;
            Ok(Response::new(valid))
        } else {
            Ok(Response::new(false))
        }
    }
}
