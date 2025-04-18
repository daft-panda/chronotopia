fn main() {
    tonic_build::configure()
        .compile_protos(
            &[
                "proto/chronotopia.proto",
                "proto/common.proto",
                "proto/user_management.proto",
                "proto/ingest.proto",
                "proto/trips.proto",
            ],
            &["proto"],
        )
        .unwrap_or_else(|e| panic!("Failed to compile protos {:?}", e));
}
