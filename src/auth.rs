use log::{trace, warn};
use tonic::{Request, Status};
use uuid::Uuid;

/// Authentication interceptor for gRPC services
pub fn auth_interceptor(mut req: Request<()>) -> Result<Request<()>, Status> {
    // Extract the authorization header
    if let Some(auth_header) = req.metadata().get("authorization") {
        if let Ok(auth_str) = auth_header.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                // Skip "Bearer "

                // In a real application, you would validate the token
                // and extract the user ID. For this example, we'll just
                // extract a user UUID from the token directly.
                // In production, consider using JWT or other secure token methods.

                match Uuid::parse_str(token) {
                    Ok(user_id) => {
                        // Add the user ID to the request extensions
                        req.extensions_mut().insert(user_id);
                        trace!("Authenticated user: {}", user_id);
                        Ok(req)
                    }
                    Err(e) => {
                        warn!("Invalid token format: {}", e);
                        Err(Status::unauthenticated("Invalid authorization token"))
                    }
                }
            } else {
                Err(Status::unauthenticated("Invalid authorization format"))
            }
        } else {
            Err(Status::unauthenticated("Invalid authorization header"))
        }
    } else {
        Err(Status::unauthenticated("Missing authorization"))
    }
}
