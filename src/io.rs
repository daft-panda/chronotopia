use chrono::{DateTime, Datelike, FixedOffset, NaiveDate, Timelike, Utc};

use crate::proto::date_time::TimeOffset;

pub(crate) mod google_maps_local_timeline {
    use chrono::{DateTime, FixedOffset, TimeDelta};
    use serde::{Deserialize, Serialize};

    use crate::proto::{LatLon, Point, Trip};

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
                    date_time: Some(
                        entry
                            .start_time
                            .checked_add_signed(TimeDelta::minutes(offset_minutes))
                            .as_ref()
                            .unwrap()
                            .into(),
                    ),
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
                start: points.first().and_then(|ts| ts.date_time.clone()),
                points,
                ..Default::default()
            }
        }
    }
}

impl From<&DateTime<FixedOffset>> for crate::proto::DateTime {
    fn from(value: &DateTime<FixedOffset>) -> Self {
        Self {
            year: value.year() as u32,
            month: value.month(),
            day: value.day(),
            hours: value.hour(),
            minutes: value.minute(),
            seconds: value.second(),
            nanos: value.nanosecond(),
            time_offset: Some(crate::proto::date_time::TimeOffset::UtcOffset(
                prost_types::Duration {
                    seconds: value.offset().local_minus_utc() as i64,
                    nanos: 0,
                },
            )),
        }
    }
}

impl From<&crate::proto::DateTime> for DateTime<FixedOffset> {
    fn from(value: &crate::proto::DateTime) -> Self {
        let nd = NaiveDate::from_ymd_opt(value.year as i32, value.month, value.day)
            .unwrap()
            .and_hms_nano_opt(value.hours, value.minutes, value.seconds, value.nanos)
            .unwrap();
        match &value.time_offset {
            Some(TimeOffset::TimeZone(_v)) => {
                unimplemented!();
            }
            Some(TimeOffset::UtcOffset(v)) => DateTime::from_naive_utc_and_offset(
                nd,
                FixedOffset::east_opt(v.seconds as i32).unwrap(),
            ),
            None => nd.and_utc().into(),
        }
    }
}

impl From<&crate::proto::DateTime> for DateTime<Utc> {
    fn from(value: &crate::proto::DateTime) -> Self {
        let nd = NaiveDate::from_ymd_opt(value.year as i32, value.month, value.day)
            .unwrap()
            .and_hms_nano_opt(value.hours, value.minutes, value.seconds, value.nanos)
            .unwrap();
        match &value.time_offset {
            Some(TimeOffset::TimeZone(_v)) => {
                unimplemented!();
            }
            Some(TimeOffset::UtcOffset(v)) => DateTime::<FixedOffset>::from_naive_utc_and_offset(
                nd,
                FixedOffset::east_opt(v.seconds as i32).unwrap(),
            )
            .to_utc(),
            None => nd.and_utc(),
        }
    }
}

impl crate::proto::DateTime {
    pub fn timestamp(&self) -> i64 {
        let dt: DateTime<FixedOffset> = self.into();
        dt.timestamp()
    }
}
