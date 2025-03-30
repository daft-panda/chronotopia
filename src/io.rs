use chrono::{DateTime, FixedOffset, TimeZone, Utc};
use prost_types::Timestamp;

pub(crate) mod google_maps_local_timeline {
    use chrono::{DateTime, FixedOffset, TimeDelta};
    use serde::{Deserialize, Serialize};

    use crate::proto::{LatLon, Point, Trip};

    use super::chrono_to_prost;

    #[derive(Debug, Serialize, Deserialize, Clone)]
    #[serde(untagged)]
    pub enum LocationHistoryEntry {
        Visit(VisitEntry),
        Activity(ActivityEntry),
        Timeline(TimelineEntry),
        TimelineMemory(TimelineMemoryEntry),
    }

    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct VisitEntry {
        #[serde(rename = "endTime")]
        pub end_time: DateTime<FixedOffset>,

        #[serde(rename = "startTime")]
        pub start_time: DateTime<FixedOffset>,

        pub visit: Visit,
    }

    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct ActivityEntry {
        #[serde(rename = "endTime")]
        pub end_time: DateTime<FixedOffset>,

        #[serde(rename = "startTime")]
        pub start_time: DateTime<FixedOffset>,

        pub activity: Activity,
    }

    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct TimelineEntry {
        #[serde(rename = "endTime")]
        pub end_time: DateTime<FixedOffset>,

        #[serde(rename = "startTime")]
        pub start_time: DateTime<FixedOffset>,

        #[serde(rename = "timelinePath")]
        pub timeline_path: Vec<TimelinePoint>,
    }

    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct TimelineMemoryEntry {
        #[serde(rename = "endTime")]
        pub end_time: DateTime<FixedOffset>,

        #[serde(rename = "startTime")]
        pub start_time: DateTime<FixedOffset>,

        #[serde(rename = "timelineMemory")]
        pub timeline_memory: TimelineMemory,
    }

    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct TimelineMemory {
        pub destinations: Vec<Destination>,

        #[serde(rename = "distanceFromOriginKms")]
        pub distance_from_origin_kms: String,
    }

    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct Destination {
        pub identifier: String,
    }

    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct TimelinePoint {
        pub point: String,

        #[serde(rename = "durationMinutesOffsetFromStartTime")]
        pub duration_minutes_offset: String,
    }

    impl TimelinePoint {
        pub fn into_point_with_entry(self, entry: &TimelineEntry) -> Option<Point> {
            if !self.point.starts_with("geo") {
                None
            } else {
                let parts = self.point.replace("geo:", "");
                let mut parts = parts.split(",");
                let lat: f64 = match parts.next().unwrap().parse() {
                    Ok(v) => v,
                    Err(_) => return None,
                };
                let lon: f64 = match parts.next().unwrap().parse() {
                    Ok(v) => v,
                    Err(_) => return None,
                };
                let offset_minutes: i64 = match self.duration_minutes_offset.parse() {
                    Ok(v) => v,
                    Err(_) => return None,
                };
                Some(Point {
                    latlon: Some(LatLon { lat, lon }),
                    timestamp: Some(chrono_to_prost(
                        &entry
                            .start_time
                            .checked_add_signed(TimeDelta::minutes(offset_minutes))
                            .unwrap(),
                    )),
                    ..Default::default()
                })
            }
        }
    }

    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct Visit {
        #[serde(rename = "hierarchyLevel")]
        pub hierarchy_level: String,

        #[serde(rename = "topCandidate")]
        pub top_candidate: TopCandidate,

        pub probability: String,
    }

    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct Activity {
        pub end: String,

        #[serde(rename = "topCandidate")]
        pub top_candidate: ActivityTopCandidate,

        #[serde(rename = "distanceMeters")]
        pub distance_meters: String,

        pub start: String,
    }

    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct TopCandidate {
        pub probability: String,

        #[serde(rename = "semanticType")]
        pub semantic_type: String,

        #[serde(rename = "placeID")]
        pub place_id: String,

        #[serde(rename = "placeLocation")]
        pub place_location: String,
    }

    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct ActivityTopCandidate {
        #[serde(rename = "type")]
        pub activity_type: String,

        pub probability: String,
    }

    impl From<&TimelineEntry> for Trip {
        fn from(val: &TimelineEntry) -> Self {
            let mut points = vec![];
            for p in &val.timeline_path {
                if let Some(point) = p.clone().into_point_with_entry(val) {
                    points.push(point);
                }
            }
            Trip {
                start: points.first().and_then(|ts| ts.timestamp),
                points,
                ..Default::default()
            }
        }
    }
}

// Convert chrono::DateTime<FixedOffset> to prost_types::Timestamp
pub fn chrono_to_prost(dt: &DateTime<FixedOffset>) -> Timestamp {
    let utc_dt = dt.with_timezone(&Utc);
    Timestamp {
        seconds: utc_dt.timestamp(),
        nanos: utc_dt.timestamp_subsec_nanos() as i32,
    }
}

// Convert prost_types::Timestamp to chrono::DateTime<FixedOffset>
pub fn prost_to_chrono(ts: &Timestamp, offset: FixedOffset) -> DateTime<FixedOffset> {
    let utc_dt = Utc.timestamp_opt(ts.seconds, ts.nanos as u32).unwrap();
    utc_dt.with_timezone(&offset)
}
