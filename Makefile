.PHONY: proto

proto:
	cd proto/ && npx --prefix ../web buf generate

clean-sample:
	rm -Rf sample/temp
	rm -Rf sample/tiles
	rm sample/*.geojson