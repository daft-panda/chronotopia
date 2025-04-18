// @generated by protoc-gen-es v1.10.0 with parameter "target=ts"
// @generated from file ingest.proto (package chronotopia, syntax proto3)
/* eslint-disable */
// @ts-nocheck

import type { BinaryReadOptions, FieldList, JsonReadOptions, JsonValue, PartialMessage, PlainMessage } from "@bufbuild/protobuf";
import { Message, proto3 } from "@bufbuild/protobuf";
import { DateTime } from "./common_pb.js";

/**
 * Source information for Chronotopia App
 *
 * @generated from message chronotopia.ChronotopiaAppSource
 */
export class ChronotopiaAppSource extends Message<ChronotopiaAppSource> {
  /**
   * @generated from field: string app_version = 1;
   */
  appVersion = "";

  constructor(data?: PartialMessage<ChronotopiaAppSource>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.ChronotopiaAppSource";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "app_version", kind: "scalar", T: 9 /* ScalarType.STRING */ },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): ChronotopiaAppSource {
    return new ChronotopiaAppSource().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): ChronotopiaAppSource {
    return new ChronotopiaAppSource().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): ChronotopiaAppSource {
    return new ChronotopiaAppSource().fromJsonString(jsonString, options);
  }

  static equals(a: ChronotopiaAppSource | PlainMessage<ChronotopiaAppSource> | undefined, b: ChronotopiaAppSource | PlainMessage<ChronotopiaAppSource> | undefined): boolean {
    return proto3.util.equals(ChronotopiaAppSource, a, b);
  }
}

/**
 * Source information for Chronotopia Web
 *
 * @generated from message chronotopia.ChronotopiaWebSource
 */
export class ChronotopiaWebSource extends Message<ChronotopiaWebSource> {
  /**
   * @generated from field: string web_version = 1;
   */
  webVersion = "";

  constructor(data?: PartialMessage<ChronotopiaWebSource>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.ChronotopiaWebSource";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "web_version", kind: "scalar", T: 9 /* ScalarType.STRING */ },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): ChronotopiaWebSource {
    return new ChronotopiaWebSource().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): ChronotopiaWebSource {
    return new ChronotopiaWebSource().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): ChronotopiaWebSource {
    return new ChronotopiaWebSource().fromJsonString(jsonString, options);
  }

  static equals(a: ChronotopiaWebSource | PlainMessage<ChronotopiaWebSource> | undefined, b: ChronotopiaWebSource | PlainMessage<ChronotopiaWebSource> | undefined): boolean {
    return proto3.util.equals(ChronotopiaWebSource, a, b);
  }
}

/**
 * Source information for Chronotopia API
 *
 * @generated from message chronotopia.ChronotopiaAPISource
 */
export class ChronotopiaAPISource extends Message<ChronotopiaAPISource> {
  /**
   * @generated from field: string api_version = 1;
   */
  apiVersion = "";

  /**
   * @generated from field: string used_api_key = 2;
   */
  usedApiKey = "";

  constructor(data?: PartialMessage<ChronotopiaAPISource>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.ChronotopiaAPISource";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "api_version", kind: "scalar", T: 9 /* ScalarType.STRING */ },
    { no: 2, name: "used_api_key", kind: "scalar", T: 9 /* ScalarType.STRING */ },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): ChronotopiaAPISource {
    return new ChronotopiaAPISource().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): ChronotopiaAPISource {
    return new ChronotopiaAPISource().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): ChronotopiaAPISource {
    return new ChronotopiaAPISource().fromJsonString(jsonString, options);
  }

  static equals(a: ChronotopiaAPISource | PlainMessage<ChronotopiaAPISource> | undefined, b: ChronotopiaAPISource | PlainMessage<ChronotopiaAPISource> | undefined): boolean {
    return proto3.util.equals(ChronotopiaAPISource, a, b);
  }
}

/**
 * Source information for Google Maps Timeline export
 *
 * @generated from message chronotopia.GoogleMapsTimelineSource
 */
export class GoogleMapsTimelineSource extends Message<GoogleMapsTimelineSource> {
  /**
   * @generated from field: string version = 1;
   */
  version = "";

  constructor(data?: PartialMessage<GoogleMapsTimelineSource>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.GoogleMapsTimelineSource";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "version", kind: "scalar", T: 9 /* ScalarType.STRING */ },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): GoogleMapsTimelineSource {
    return new GoogleMapsTimelineSource().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): GoogleMapsTimelineSource {
    return new GoogleMapsTimelineSource().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): GoogleMapsTimelineSource {
    return new GoogleMapsTimelineSource().fromJsonString(jsonString, options);
  }

  static equals(a: GoogleMapsTimelineSource | PlainMessage<GoogleMapsTimelineSource> | undefined, b: GoogleMapsTimelineSource | PlainMessage<GoogleMapsTimelineSource> | undefined): boolean {
    return proto3.util.equals(GoogleMapsTimelineSource, a, b);
  }
}

/**
 * Source information for Apple Health export
 *
 * @generated from message chronotopia.AppleHealthSource
 */
export class AppleHealthSource extends Message<AppleHealthSource> {
  /**
   * @generated from field: string version = 1;
   */
  version = "";

  constructor(data?: PartialMessage<AppleHealthSource>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.AppleHealthSource";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "version", kind: "scalar", T: 9 /* ScalarType.STRING */ },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): AppleHealthSource {
    return new AppleHealthSource().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): AppleHealthSource {
    return new AppleHealthSource().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): AppleHealthSource {
    return new AppleHealthSource().fromJsonString(jsonString, options);
  }

  static equals(a: AppleHealthSource | PlainMessage<AppleHealthSource> | undefined, b: AppleHealthSource | PlainMessage<AppleHealthSource> | undefined): boolean {
    return proto3.util.equals(AppleHealthSource, a, b);
  }
}

/**
 * Source information for external fitness apps
 *
 * @generated from message chronotopia.FitnessAppSource
 */
export class FitnessAppSource extends Message<FitnessAppSource> {
  /**
   * @generated from field: string app_name = 1;
   */
  appName = "";

  /**
   * @generated from field: string app_version = 2;
   */
  appVersion = "";

  constructor(data?: PartialMessage<FitnessAppSource>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.FitnessAppSource";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "app_name", kind: "scalar", T: 9 /* ScalarType.STRING */ },
    { no: 2, name: "app_version", kind: "scalar", T: 9 /* ScalarType.STRING */ },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): FitnessAppSource {
    return new FitnessAppSource().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): FitnessAppSource {
    return new FitnessAppSource().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): FitnessAppSource {
    return new FitnessAppSource().fromJsonString(jsonString, options);
  }

  static equals(a: FitnessAppSource | PlainMessage<FitnessAppSource> | undefined, b: FitnessAppSource | PlainMessage<FitnessAppSource> | undefined): boolean {
    return proto3.util.equals(FitnessAppSource, a, b);
  }
}

/**
 * @generated from message chronotopia.IngestBatch
 */
export class IngestBatch extends Message<IngestBatch> {
  /**
   * DateTime when the packet was created
   *
   * @generated from field: chronotopia.DateTime dateTime = 3;
   */
  dateTime?: DateTime;

  /**
   * Batch of location points
   *
   * @generated from field: repeated chronotopia.LocationPoint locations = 4;
   */
  locations: LocationPoint[] = [];

  /**
   * Batch of activity events
   *
   * @generated from field: repeated chronotopia.ActivityEvent activities = 5;
   */
  activities: ActivityEvent[] = [];

  /**
   * Batch of visit events
   *
   * @generated from field: repeated chronotopia.VisitEvent visits = 6;
   */
  visits: VisitEvent[] = [];

  /**
   * Source information about where this data came from
   *
   * @generated from oneof chronotopia.IngestBatch.source
   */
  source: {
    /**
     * @generated from field: chronotopia.ChronotopiaAppSource chronotopia_app = 7;
     */
    value: ChronotopiaAppSource;
    case: "chronotopiaApp";
  } | {
    /**
     * @generated from field: chronotopia.ChronotopiaWebSource chronotopia_web = 8;
     */
    value: ChronotopiaWebSource;
    case: "chronotopiaWeb";
  } | {
    /**
     * @generated from field: chronotopia.ChronotopiaAPISource chronotopia_api = 9;
     */
    value: ChronotopiaAPISource;
    case: "chronotopiaApi";
  } | {
    /**
     * @generated from field: chronotopia.GoogleMapsTimelineSource google_maps = 10;
     */
    value: GoogleMapsTimelineSource;
    case: "googleMaps";
  } | {
    /**
     * @generated from field: chronotopia.AppleHealthSource apple_health = 11;
     */
    value: AppleHealthSource;
    case: "appleHealth";
  } | {
    /**
     * @generated from field: chronotopia.FitnessAppSource fitness_app = 12;
     */
    value: FitnessAppSource;
    case: "fitnessApp";
  } | { case: undefined; value?: undefined } = { case: undefined };

  constructor(data?: PartialMessage<IngestBatch>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.IngestBatch";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 3, name: "dateTime", kind: "message", T: DateTime },
    { no: 4, name: "locations", kind: "message", T: LocationPoint, repeated: true },
    { no: 5, name: "activities", kind: "message", T: ActivityEvent, repeated: true },
    { no: 6, name: "visits", kind: "message", T: VisitEvent, repeated: true },
    { no: 7, name: "chronotopia_app", kind: "message", T: ChronotopiaAppSource, oneof: "source" },
    { no: 8, name: "chronotopia_web", kind: "message", T: ChronotopiaWebSource, oneof: "source" },
    { no: 9, name: "chronotopia_api", kind: "message", T: ChronotopiaAPISource, oneof: "source" },
    { no: 10, name: "google_maps", kind: "message", T: GoogleMapsTimelineSource, oneof: "source" },
    { no: 11, name: "apple_health", kind: "message", T: AppleHealthSource, oneof: "source" },
    { no: 12, name: "fitness_app", kind: "message", T: FitnessAppSource, oneof: "source" },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): IngestBatch {
    return new IngestBatch().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): IngestBatch {
    return new IngestBatch().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): IngestBatch {
    return new IngestBatch().fromJsonString(jsonString, options);
  }

  static equals(a: IngestBatch | PlainMessage<IngestBatch> | undefined, b: IngestBatch | PlainMessage<IngestBatch> | undefined): boolean {
    return proto3.util.equals(IngestBatch, a, b);
  }
}

/**
 * Location point data
 *
 * @generated from message chronotopia.LocationPoint
 */
export class LocationPoint extends Message<LocationPoint> {
  /**
   * Basic coordinates
   *
   * @generated from field: double latitude = 1;
   */
  latitude = 0;

  /**
   * @generated from field: double longitude = 2;
   */
  longitude = 0;

  /**
   * @generated from field: double altitude = 3;
   */
  altitude = 0;

  /**
   * Accuracy information
   *
   * @generated from field: double horizontal_accuracy = 4;
   */
  horizontalAccuracy = 0;

  /**
   * @generated from field: double vertical_accuracy = 5;
   */
  verticalAccuracy = 0;

  /**
   * @generated from field: double speed_accuracy = 6;
   */
  speedAccuracy = 0;

  /**
   * @generated from field: double bearing_accuracy = 7;
   */
  bearingAccuracy = 0;

  /**
   * Motion information
   *
   * @generated from field: double speed = 8;
   */
  speed = 0;

  /**
   * @generated from field: double bearing = 9;
   */
  bearing = 0;

  /**
   * Timing
   *
   * @generated from field: chronotopia.DateTime dateTime = 10;
   */
  dateTime?: DateTime;

  /**
   * Location provider information
   *
   * @generated from field: bool is_mock_location = 11;
   */
  isMockLocation = false;

  /**
   * For indoor positioning (iOS)
   *
   * @generated from field: optional int32 floor_level = 12;
   */
  floorLevel?: number;

  /**
   * Device state information
   *
   * @generated from field: uint32 battery_level = 13;
   */
  batteryLevel = 0;

  /**
   * @generated from field: optional string network_type = 14;
   */
  networkType?: string;

  constructor(data?: PartialMessage<LocationPoint>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.LocationPoint";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "latitude", kind: "scalar", T: 1 /* ScalarType.DOUBLE */ },
    { no: 2, name: "longitude", kind: "scalar", T: 1 /* ScalarType.DOUBLE */ },
    { no: 3, name: "altitude", kind: "scalar", T: 1 /* ScalarType.DOUBLE */ },
    { no: 4, name: "horizontal_accuracy", kind: "scalar", T: 1 /* ScalarType.DOUBLE */ },
    { no: 5, name: "vertical_accuracy", kind: "scalar", T: 1 /* ScalarType.DOUBLE */ },
    { no: 6, name: "speed_accuracy", kind: "scalar", T: 1 /* ScalarType.DOUBLE */ },
    { no: 7, name: "bearing_accuracy", kind: "scalar", T: 1 /* ScalarType.DOUBLE */ },
    { no: 8, name: "speed", kind: "scalar", T: 1 /* ScalarType.DOUBLE */ },
    { no: 9, name: "bearing", kind: "scalar", T: 1 /* ScalarType.DOUBLE */ },
    { no: 10, name: "dateTime", kind: "message", T: DateTime },
    { no: 11, name: "is_mock_location", kind: "scalar", T: 8 /* ScalarType.BOOL */ },
    { no: 12, name: "floor_level", kind: "scalar", T: 5 /* ScalarType.INT32 */, opt: true },
    { no: 13, name: "battery_level", kind: "scalar", T: 13 /* ScalarType.UINT32 */ },
    { no: 14, name: "network_type", kind: "scalar", T: 9 /* ScalarType.STRING */, opt: true },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): LocationPoint {
    return new LocationPoint().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): LocationPoint {
    return new LocationPoint().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): LocationPoint {
    return new LocationPoint().fromJsonString(jsonString, options);
  }

  static equals(a: LocationPoint | PlainMessage<LocationPoint> | undefined, b: LocationPoint | PlainMessage<LocationPoint> | undefined): boolean {
    return proto3.util.equals(LocationPoint, a, b);
  }
}

/**
 * Activity recognition event
 *
 * @generated from message chronotopia.ActivityEvent
 */
export class ActivityEvent extends Message<ActivityEvent> {
  /**
   * @generated from field: chronotopia.ActivityEvent.ActivityType type = 1;
   */
  type = ActivityEvent_ActivityType.UNKNOWN;

  /**
   * 0-100
   *
   * @generated from field: int32 confidence = 2;
   */
  confidence = 0;

  /**
   * Timestamps
   *
   * @generated from field: chronotopia.DateTime start = 3;
   */
  start?: DateTime;

  /**
   * @generated from field: optional chronotopia.DateTime end = 4;
   */
  end?: DateTime;

  /**
   * Movement data
   *
   * @generated from field: optional int64 step_count = 5;
   */
  stepCount?: bigint;

  /**
   * @generated from field: optional double distance = 6;
   */
  distance?: number;

  constructor(data?: PartialMessage<ActivityEvent>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.ActivityEvent";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "type", kind: "enum", T: proto3.getEnumType(ActivityEvent_ActivityType) },
    { no: 2, name: "confidence", kind: "scalar", T: 5 /* ScalarType.INT32 */ },
    { no: 3, name: "start", kind: "message", T: DateTime },
    { no: 4, name: "end", kind: "message", T: DateTime, opt: true },
    { no: 5, name: "step_count", kind: "scalar", T: 3 /* ScalarType.INT64 */, opt: true },
    { no: 6, name: "distance", kind: "scalar", T: 1 /* ScalarType.DOUBLE */, opt: true },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): ActivityEvent {
    return new ActivityEvent().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): ActivityEvent {
    return new ActivityEvent().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): ActivityEvent {
    return new ActivityEvent().fromJsonString(jsonString, options);
  }

  static equals(a: ActivityEvent | PlainMessage<ActivityEvent> | undefined, b: ActivityEvent | PlainMessage<ActivityEvent> | undefined): boolean {
    return proto3.util.equals(ActivityEvent, a, b);
  }
}

/**
 * @generated from enum chronotopia.ActivityEvent.ActivityType
 */
export enum ActivityEvent_ActivityType {
  /**
   * @generated from enum value: UNKNOWN = 0;
   */
  UNKNOWN = 0,

  /**
   * @generated from enum value: STILL = 1;
   */
  STILL = 1,

  /**
   * @generated from enum value: WALKING = 2;
   */
  WALKING = 2,

  /**
   * @generated from enum value: RUNNING = 3;
   */
  RUNNING = 3,

  /**
   * @generated from enum value: IN_VEHICLE = 4;
   */
  IN_VEHICLE = 4,

  /**
   * @generated from enum value: ON_BICYCLE = 5;
   */
  ON_BICYCLE = 5,

  /**
   * @generated from enum value: ON_FOOT = 6;
   */
  ON_FOOT = 6,

  /**
   * @generated from enum value: TILTING = 7;
   */
  TILTING = 7,
}
// Retrieve enum metadata with: proto3.getEnumType(ActivityEvent_ActivityType)
proto3.util.setEnumType(ActivityEvent_ActivityType, "chronotopia.ActivityEvent.ActivityType", [
  { no: 0, name: "UNKNOWN" },
  { no: 1, name: "STILL" },
  { no: 2, name: "WALKING" },
  { no: 3, name: "RUNNING" },
  { no: 4, name: "IN_VEHICLE" },
  { no: 5, name: "ON_BICYCLE" },
  { no: 6, name: "ON_FOOT" },
  { no: 7, name: "TILTING" },
]);

/**
 * Visit events (iOS-specific but we'll handle on Android too)
 *
 * @generated from message chronotopia.VisitEvent
 */
export class VisitEvent extends Message<VisitEvent> {
  /**
   * @generated from field: double latitude = 1;
   */
  latitude = 0;

  /**
   * @generated from field: double longitude = 2;
   */
  longitude = 0;

  /**
   * @generated from field: double horizontal_accuracy = 3;
   */
  horizontalAccuracy = 0;

  /**
   * @generated from field: chronotopia.DateTime arrival = 4;
   */
  arrival?: DateTime;

  /**
   * @generated from field: chronotopia.DateTime departure = 5;
   */
  departure?: DateTime;

  /**
   * Semantic information
   *
   * e.g., "home", "work", "restaurant"
   *
   * @generated from field: string canonical_label = 6;
   */
  canonicalLabel = "";

  /**
   * External place identifier
   *
   * @generated from field: optional string external_place_id = 8;
   */
  externalPlaceId?: string;

  constructor(data?: PartialMessage<VisitEvent>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.VisitEvent";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "latitude", kind: "scalar", T: 1 /* ScalarType.DOUBLE */ },
    { no: 2, name: "longitude", kind: "scalar", T: 1 /* ScalarType.DOUBLE */ },
    { no: 3, name: "horizontal_accuracy", kind: "scalar", T: 1 /* ScalarType.DOUBLE */ },
    { no: 4, name: "arrival", kind: "message", T: DateTime },
    { no: 5, name: "departure", kind: "message", T: DateTime },
    { no: 6, name: "canonical_label", kind: "scalar", T: 9 /* ScalarType.STRING */ },
    { no: 8, name: "external_place_id", kind: "scalar", T: 9 /* ScalarType.STRING */, opt: true },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): VisitEvent {
    return new VisitEvent().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): VisitEvent {
    return new VisitEvent().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): VisitEvent {
    return new VisitEvent().fromJsonString(jsonString, options);
  }

  static equals(a: VisitEvent | PlainMessage<VisitEvent> | undefined, b: VisitEvent | PlainMessage<VisitEvent> | undefined): boolean {
    return proto3.util.equals(VisitEvent, a, b);
  }
}

/**
 * Device metadata
 *
 * @generated from message chronotopia.DeviceMetadata
 */
export class DeviceMetadata extends Message<DeviceMetadata> {
  /**
   * "android" or "ios"
   *
   * @generated from field: string platform = 1;
   */
  platform = "";

  /**
   * @generated from field: string os_version = 2;
   */
  osVersion = "";

  /**
   * @generated from field: string app_version = 3;
   */
  appVersion = "";

  /**
   * @generated from field: string device_model = 4;
   */
  deviceModel = "";

  /**
   * @generated from field: string device_language = 5;
   */
  deviceLanguage = "";

  constructor(data?: PartialMessage<DeviceMetadata>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.DeviceMetadata";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "platform", kind: "scalar", T: 9 /* ScalarType.STRING */ },
    { no: 2, name: "os_version", kind: "scalar", T: 9 /* ScalarType.STRING */ },
    { no: 3, name: "app_version", kind: "scalar", T: 9 /* ScalarType.STRING */ },
    { no: 4, name: "device_model", kind: "scalar", T: 9 /* ScalarType.STRING */ },
    { no: 5, name: "device_language", kind: "scalar", T: 9 /* ScalarType.STRING */ },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): DeviceMetadata {
    return new DeviceMetadata().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): DeviceMetadata {
    return new DeviceMetadata().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): DeviceMetadata {
    return new DeviceMetadata().fromJsonString(jsonString, options);
  }

  static equals(a: DeviceMetadata | PlainMessage<DeviceMetadata> | undefined, b: DeviceMetadata | PlainMessage<DeviceMetadata> | undefined): boolean {
    return proto3.util.equals(DeviceMetadata, a, b);
  }
}

/**
 * Google Maps Timeline Export data
 *
 * @generated from message chronotopia.GoogleMapsTimelineExport
 */
export class GoogleMapsTimelineExport extends Message<GoogleMapsTimelineExport> {
  /**
   * Raw JSON content of the Google Maps Timeline export
   *
   * @generated from field: string json_content = 1;
   */
  jsonContent = "";

  /**
   * Optional user-provided metadata
   *
   * @generated from field: string export_name = 2;
   */
  exportName = "";

  /**
   * @generated from field: optional chronotopia.DateTime export_date = 3;
   */
  exportDate?: DateTime;

  constructor(data?: PartialMessage<GoogleMapsTimelineExport>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.GoogleMapsTimelineExport";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "json_content", kind: "scalar", T: 9 /* ScalarType.STRING */ },
    { no: 2, name: "export_name", kind: "scalar", T: 9 /* ScalarType.STRING */ },
    { no: 3, name: "export_date", kind: "message", T: DateTime, opt: true },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): GoogleMapsTimelineExport {
    return new GoogleMapsTimelineExport().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): GoogleMapsTimelineExport {
    return new GoogleMapsTimelineExport().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): GoogleMapsTimelineExport {
    return new GoogleMapsTimelineExport().fromJsonString(jsonString, options);
  }

  static equals(a: GoogleMapsTimelineExport | PlainMessage<GoogleMapsTimelineExport> | undefined, b: GoogleMapsTimelineExport | PlainMessage<GoogleMapsTimelineExport> | undefined): boolean {
    return proto3.util.equals(GoogleMapsTimelineExport, a, b);
  }
}

/**
 * Response to data uploads
 *
 * @generated from message chronotopia.IngestResponse
 */
export class IngestResponse extends Message<IngestResponse> {
  /**
   * @generated from field: bool success = 1;
   */
  success = false;

  /**
   * @generated from field: string alert_message = 2;
   */
  alertMessage = "";

  /**
   * Optional fields for server instructions
   *
   * @generated from field: bool pause_tracking = 3;
   */
  pauseTracking = false;

  /**
   * in seconds
   *
   * @generated from field: int32 recommended_upload_interval = 4;
   */
  recommendedUploadInterval = 0;

  /**
   * Processing statistics for timeline exports
   *
   * @generated from field: optional int32 processed_locations = 5;
   */
  processedLocations?: number;

  /**
   * @generated from field: optional int32 processed_activities = 6;
   */
  processedActivities?: number;

  /**
   * @generated from field: optional int32 processed_visits = 7;
   */
  processedVisits?: number;

  constructor(data?: PartialMessage<IngestResponse>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.IngestResponse";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "success", kind: "scalar", T: 8 /* ScalarType.BOOL */ },
    { no: 2, name: "alert_message", kind: "scalar", T: 9 /* ScalarType.STRING */ },
    { no: 3, name: "pause_tracking", kind: "scalar", T: 8 /* ScalarType.BOOL */ },
    { no: 4, name: "recommended_upload_interval", kind: "scalar", T: 5 /* ScalarType.INT32 */ },
    { no: 5, name: "processed_locations", kind: "scalar", T: 5 /* ScalarType.INT32 */, opt: true },
    { no: 6, name: "processed_activities", kind: "scalar", T: 5 /* ScalarType.INT32 */, opt: true },
    { no: 7, name: "processed_visits", kind: "scalar", T: 5 /* ScalarType.INT32 */, opt: true },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): IngestResponse {
    return new IngestResponse().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): IngestResponse {
    return new IngestResponse().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): IngestResponse {
    return new IngestResponse().fromJsonString(jsonString, options);
  }

  static equals(a: IngestResponse | PlainMessage<IngestResponse> | undefined, b: IngestResponse | PlainMessage<IngestResponse> | undefined): boolean {
    return proto3.util.equals(IngestResponse, a, b);
  }
}

