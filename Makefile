.PHONY: proto

proto:
	cd proto/ && npx --prefix ../web buf generate

proto-swift:
	cd proto && protoc --swift_out=.  ingest.proto && --swift_out=. --plugin=/Users/wannes/Downloads/grpc-swift-protobuf/.build/arm64-apple-macosx/debug/protoc-gen-swift-tool  ingest.proto && \
	protoc --swift_out=. datetime.proto

clean-sample:
	rm -Rf sample/temp
	rm -Rf sample/tiles
	rm sample/*.geojson

