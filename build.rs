fn main() {
    tonic_build::configure()
        .compile_protos(
            &[
                "proto/chronotopia.proto",
                "proto/datetime.proto",
                "proto/user_management.proto",
            ],
            &["proto"],
        )
        .unwrap_or_else(|e| panic!("Failed to compile protos {:?}", e));
}
