// @generated by protoc-gen-es v1.10.0 with parameter "target=ts"
// @generated from file common.proto (package chronotopia, syntax proto3)
/* eslint-disable */
// @ts-nocheck

import type { BinaryReadOptions, FieldList, JsonReadOptions, JsonValue, PartialMessage, PlainMessage } from "@bufbuild/protobuf";
import { Duration, Message, proto3, protoInt64 } from "@bufbuild/protobuf";

/**
 * @generated from message chronotopia.RoadSegment
 */
export class RoadSegment extends Message<RoadSegment> {
  /**
   * @generated from field: uint64 id = 1;
   */
  id = protoInt64.zero;

  /**
   * @generated from field: uint64 osm_way_id = 2;
   */
  osmWayId = protoInt64.zero;

  /**
   * @generated from field: repeated chronotopia.LatLon coordinates = 3;
   */
  coordinates: LatLon[] = [];

  /**
   * @generated from field: bool is_oneway = 4;
   */
  isOneway = false;

  /**
   * @generated from field: string highway_type = 5;
   */
  highwayType = "";

  /**
   * @generated from field: repeated uint64 connections = 6;
   */
  connections: bigint[] = [];

  /**
   * @generated from field: optional string name = 7;
   */
  name?: string;

  /**
   * @generated from field: optional uint32 interim_start_idx = 8;
   */
  interimStartIdx?: number;

  /**
   * @generated from field: optional uint32 interim_end_idx = 9;
   */
  interimEndIdx?: number;

  /**
   * @generated from field: repeated chronotopia.LatLon full_coordinates = 10;
   */
  fullCoordinates: LatLon[] = [];

  constructor(data?: PartialMessage<RoadSegment>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.RoadSegment";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "id", kind: "scalar", T: 4 /* ScalarType.UINT64 */ },
    { no: 2, name: "osm_way_id", kind: "scalar", T: 4 /* ScalarType.UINT64 */ },
    { no: 3, name: "coordinates", kind: "message", T: LatLon, repeated: true },
    { no: 4, name: "is_oneway", kind: "scalar", T: 8 /* ScalarType.BOOL */ },
    { no: 5, name: "highway_type", kind: "scalar", T: 9 /* ScalarType.STRING */ },
    { no: 6, name: "connections", kind: "scalar", T: 4 /* ScalarType.UINT64 */, repeated: true },
    { no: 7, name: "name", kind: "scalar", T: 9 /* ScalarType.STRING */, opt: true },
    { no: 8, name: "interim_start_idx", kind: "scalar", T: 13 /* ScalarType.UINT32 */, opt: true },
    { no: 9, name: "interim_end_idx", kind: "scalar", T: 13 /* ScalarType.UINT32 */, opt: true },
    { no: 10, name: "full_coordinates", kind: "message", T: LatLon, repeated: true },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): RoadSegment {
    return new RoadSegment().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): RoadSegment {
    return new RoadSegment().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): RoadSegment {
    return new RoadSegment().fromJsonString(jsonString, options);
  }

  static equals(a: RoadSegment | PlainMessage<RoadSegment> | undefined, b: RoadSegment | PlainMessage<RoadSegment> | undefined): boolean {
    return proto3.util.equals(RoadSegment, a, b);
  }
}

/**
 * UUID type
 *
 * @generated from message chronotopia.UUID
 */
export class UUID extends Message<UUID> {
  /**
   * @generated from field: string value = 1;
   */
  value = "";

  constructor(data?: PartialMessage<UUID>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.UUID";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "value", kind: "scalar", T: 9 /* ScalarType.STRING */ },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): UUID {
    return new UUID().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): UUID {
    return new UUID().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): UUID {
    return new UUID().fromJsonString(jsonString, options);
  }

  static equals(a: UUID | PlainMessage<UUID> | undefined, b: UUID | PlainMessage<UUID> | undefined): boolean {
    return proto3.util.equals(UUID, a, b);
  }
}

/**
 * @generated from message chronotopia.LatLon
 */
export class LatLon extends Message<LatLon> {
  /**
   * @generated from field: double lat = 1;
   */
  lat = 0;

  /**
   * @generated from field: double lon = 2;
   */
  lon = 0;

  constructor(data?: PartialMessage<LatLon>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.LatLon";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "lat", kind: "scalar", T: 1 /* ScalarType.DOUBLE */ },
    { no: 2, name: "lon", kind: "scalar", T: 1 /* ScalarType.DOUBLE */ },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): LatLon {
    return new LatLon().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): LatLon {
    return new LatLon().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): LatLon {
    return new LatLon().fromJsonString(jsonString, options);
  }

  static equals(a: LatLon | PlainMessage<LatLon> | undefined, b: LatLon | PlainMessage<LatLon> | undefined): boolean {
    return proto3.util.equals(LatLon, a, b);
  }
}

/**
 * @generated from message chronotopia.Point
 */
export class Point extends Message<Point> {
  /**
   * @generated from field: chronotopia.LatLon latlon = 1;
   */
  latlon?: LatLon;

  /**
   * @generated from field: chronotopia.DateTime dateTime = 3;
   */
  dateTime?: DateTime;

  /**
   * @generated from field: optional string label = 4;
   */
  label?: string;

  /**
   * @generated from field: optional string note = 5;
   */
  note?: string;

  /**
   * @generated from field: optional float elevation = 6;
   */
  elevation?: number;

  constructor(data?: PartialMessage<Point>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.Point";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "latlon", kind: "message", T: LatLon },
    { no: 3, name: "dateTime", kind: "message", T: DateTime },
    { no: 4, name: "label", kind: "scalar", T: 9 /* ScalarType.STRING */, opt: true },
    { no: 5, name: "note", kind: "scalar", T: 9 /* ScalarType.STRING */, opt: true },
    { no: 6, name: "elevation", kind: "scalar", T: 2 /* ScalarType.FLOAT */, opt: true },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): Point {
    return new Point().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): Point {
    return new Point().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): Point {
    return new Point().fromJsonString(jsonString, options);
  }

  static equals(a: Point | PlainMessage<Point> | undefined, b: Point | PlainMessage<Point> | undefined): boolean {
    return proto3.util.equals(Point, a, b);
  }
}

/**
 * @generated from message chronotopia.Points
 */
export class Points extends Message<Points> {
  /**
   * @generated from field: repeated chronotopia.Point points = 1;
   */
  points: Point[] = [];

  constructor(data?: PartialMessage<Points>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.Points";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "points", kind: "message", T: Point, repeated: true },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): Points {
    return new Points().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): Points {
    return new Points().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): Points {
    return new Points().fromJsonString(jsonString, options);
  }

  static equals(a: Points | PlainMessage<Points> | undefined, b: Points | PlainMessage<Points> | undefined): boolean {
    return proto3.util.equals(Points, a, b);
  }
}

/**
 * @generated from message chronotopia.PlanRouteRequest
 */
export class PlanRouteRequest extends Message<PlanRouteRequest> {
  /**
   * @generated from field: chronotopia.LatLon start_point = 1;
   */
  startPoint?: LatLon;

  /**
   * @generated from field: chronotopia.LatLon end_point = 2;
   */
  endPoint?: LatLon;

  /**
   * @generated from field: repeated chronotopia.LatLon via_points = 3;
   */
  viaPoints: LatLon[] = [];

  constructor(data?: PartialMessage<PlanRouteRequest>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.PlanRouteRequest";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "start_point", kind: "message", T: LatLon },
    { no: 2, name: "end_point", kind: "message", T: LatLon },
    { no: 3, name: "via_points", kind: "message", T: LatLon, repeated: true },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): PlanRouteRequest {
    return new PlanRouteRequest().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): PlanRouteRequest {
    return new PlanRouteRequest().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): PlanRouteRequest {
    return new PlanRouteRequest().fromJsonString(jsonString, options);
  }

  static equals(a: PlanRouteRequest | PlainMessage<PlanRouteRequest> | undefined, b: PlanRouteRequest | PlainMessage<PlanRouteRequest> | undefined): boolean {
    return proto3.util.equals(PlanRouteRequest, a, b);
  }
}

/**
 * @generated from message chronotopia.PlanRouteResponse
 */
export class PlanRouteResponse extends Message<PlanRouteResponse> {
  /**
   * @generated from field: string geojson = 1;
   */
  geojson = "";

  /**
   * @generated from field: repeated chronotopia.RoadSegment segments = 2;
   */
  segments: RoadSegment[] = [];

  constructor(data?: PartialMessage<PlanRouteResponse>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.PlanRouteResponse";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "geojson", kind: "scalar", T: 9 /* ScalarType.STRING */ },
    { no: 2, name: "segments", kind: "message", T: RoadSegment, repeated: true },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): PlanRouteResponse {
    return new PlanRouteResponse().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): PlanRouteResponse {
    return new PlanRouteResponse().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): PlanRouteResponse {
    return new PlanRouteResponse().fromJsonString(jsonString, options);
  }

  static equals(a: PlanRouteResponse | PlainMessage<PlanRouteResponse> | undefined, b: PlanRouteResponse | PlainMessage<PlanRouteResponse> | undefined): boolean {
    return proto3.util.equals(PlanRouteResponse, a, b);
  }
}

/**
 * Represents civil time (or occasionally physical time).
 *
 * This type can represent a civil time in one of a few possible ways:
 *
 *  * When utc_offset is set and time_zone is unset: a civil time on a calendar
 *    day with a particular offset from UTC.
 *  * When time_zone is set and utc_offset is unset: a civil time on a calendar
 *    day in a particular time zone.
 *  * When neither time_zone nor utc_offset is set: a civil time on a calendar
 *    day in local time.
 *
 * The date is relative to the Proleptic Gregorian Calendar.
 *
 * If year is 0, the DateTime is considered not to have a specific year. month
 * and day must have valid, non-zero values.
 *
 * This type may also be used to represent a physical time if all the date and
 * time fields are set and either case of the `time_offset` oneof is set.
 * Consider using `Timestamp` message for physical time instead. If your use
 * case also would like to store the user's timezone, that can be done in
 * another field.
 *
 * This type is more flexible than some applications may want. Make sure to
 * document and validate your application's limitations.
 *
 * @generated from message chronotopia.DateTime
 */
export class DateTime extends Message<DateTime> {
  /**
   * Optional. Year of date. Must be from 1 to 9999, or 0 if specifying a
   * datetime without a year.
   *
   * @generated from field: uint32 year = 1;
   */
  year = 0;

  /**
   * Required. Month of year. Must be from 1 to 12.
   *
   * @generated from field: uint32 month = 2;
   */
  month = 0;

  /**
   * Required. Day of month. Must be from 1 to 31 and valid for the year and
   * month.
   *
   * @generated from field: uint32 day = 3;
   */
  day = 0;

  /**
   * Required. Hours of day in 24 hour format. Should be from 0 to 23. An API
   * may choose to allow the value "24:00:00" for scenarios like business
   * closing time.
   *
   * @generated from field: uint32 hours = 4;
   */
  hours = 0;

  /**
   * Required. Minutes of hour of day. Must be from 0 to 59.
   *
   * @generated from field: uint32 minutes = 5;
   */
  minutes = 0;

  /**
   * Required. Seconds of minutes of the time. Must normally be from 0 to 59. An
   * API may allow the value 60 if it allows leap-seconds.
   *
   * @generated from field: uint32 seconds = 6;
   */
  seconds = 0;

  /**
   * Required. Fractions of seconds in nanoseconds. Must be from 0 to
   * 999,999,999.
   *
   * @generated from field: uint32 nanos = 7;
   */
  nanos = 0;

  /**
   * Optional. Specifies either the UTC offset or the time zone of the DateTime.
   * Choose carefully between them, considering that time zone data may change
   * in the future (for example, a country modifies their DST start/end dates,
   * and future DateTimes in the affected range had already been stored).
   * If omitted, the DateTime is considered to be in local time.
   *
   * @generated from oneof chronotopia.DateTime.time_offset
   */
  timeOffset: {
    /**
     * UTC offset. Must be whole seconds, between -18 hours and +18 hours.
     * For example, a UTC offset of -4:00 would be represented as
     * { seconds: -14400 }.
     *
     * @generated from field: google.protobuf.Duration utc_offset = 8;
     */
    value: Duration;
    case: "utcOffset";
  } | {
    /**
     * Time zone.
     *
     * @generated from field: chronotopia.TimeZone time_zone = 9;
     */
    value: TimeZone;
    case: "timeZone";
  } | { case: undefined; value?: undefined } = { case: undefined };

  constructor(data?: PartialMessage<DateTime>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.DateTime";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "year", kind: "scalar", T: 13 /* ScalarType.UINT32 */ },
    { no: 2, name: "month", kind: "scalar", T: 13 /* ScalarType.UINT32 */ },
    { no: 3, name: "day", kind: "scalar", T: 13 /* ScalarType.UINT32 */ },
    { no: 4, name: "hours", kind: "scalar", T: 13 /* ScalarType.UINT32 */ },
    { no: 5, name: "minutes", kind: "scalar", T: 13 /* ScalarType.UINT32 */ },
    { no: 6, name: "seconds", kind: "scalar", T: 13 /* ScalarType.UINT32 */ },
    { no: 7, name: "nanos", kind: "scalar", T: 13 /* ScalarType.UINT32 */ },
    { no: 8, name: "utc_offset", kind: "message", T: Duration, oneof: "time_offset" },
    { no: 9, name: "time_zone", kind: "message", T: TimeZone, oneof: "time_offset" },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): DateTime {
    return new DateTime().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): DateTime {
    return new DateTime().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): DateTime {
    return new DateTime().fromJsonString(jsonString, options);
  }

  static equals(a: DateTime | PlainMessage<DateTime> | undefined, b: DateTime | PlainMessage<DateTime> | undefined): boolean {
    return proto3.util.equals(DateTime, a, b);
  }
}

/**
 * Represents a time zone from the
 * [IANA Time Zone Database](https://www.iana.org/time-zones).
 *
 * @generated from message chronotopia.TimeZone
 */
export class TimeZone extends Message<TimeZone> {
  /**
   * IANA Time Zone Database time zone, e.g. "America/New_York".
   *
   * @generated from field: string id = 1;
   */
  id = "";

  /**
   * Optional. IANA Time Zone Database version number, e.g. "2019a".
   *
   * @generated from field: string version = 2;
   */
  version = "";

  constructor(data?: PartialMessage<TimeZone>) {
    super();
    proto3.util.initPartial(data, this);
  }

  static readonly runtime: typeof proto3 = proto3;
  static readonly typeName = "chronotopia.TimeZone";
  static readonly fields: FieldList = proto3.util.newFieldList(() => [
    { no: 1, name: "id", kind: "scalar", T: 9 /* ScalarType.STRING */ },
    { no: 2, name: "version", kind: "scalar", T: 9 /* ScalarType.STRING */ },
  ]);

  static fromBinary(bytes: Uint8Array, options?: Partial<BinaryReadOptions>): TimeZone {
    return new TimeZone().fromBinary(bytes, options);
  }

  static fromJson(jsonValue: JsonValue, options?: Partial<JsonReadOptions>): TimeZone {
    return new TimeZone().fromJson(jsonValue, options);
  }

  static fromJsonString(jsonString: string, options?: Partial<JsonReadOptions>): TimeZone {
    return new TimeZone().fromJsonString(jsonString, options);
  }

  static equals(a: TimeZone | PlainMessage<TimeZone> | undefined, b: TimeZone | PlainMessage<TimeZone> | undefined): boolean {
    return proto3.util.equals(TimeZone, a, b);
  }
}

