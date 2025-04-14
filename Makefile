.PHONY: proto

proto:
	cd proto/ && npx --prefix ../web buf generate

proto-swift:
	cd proto && protoc --swift_out=. ingest.proto && protoc --grpc-swift_out=. --plugin=/Users/wannes/Downloads/grpc-swift-protobuf/.build/arm64-apple-macosx/debug/protoc-gen-grpc-swift ingest.proto && protoc --swift_out=. common.proto

clean-sample:
	rm -Rf sample/temp
	rm -Rf sample/tiles
	rm sample/*.geojson

clippy:
	cargo clippy --fix --allow-dirty

migrate-up:
	sea-orm-cli migrate up

migrate-down:
	sea-orm-cli migrate down

generate-db-entity:
	echo 'Generating entities into tmp-entity, copy over the ones you need'
	mkdir -p tmp-entity
	sea-orm-cli generate entity -o tmp-entity/