use geo::{Closest, ClosestPoint, Distance as _, Haversine, Intersects, LineString, Point};

use crate::{osm_preprocessing::WaySegment, route_matcher::DISTANCE_THRESHOLD_FOR_COST_BIAS_METER};

/// Calculate the traversal cost between two road segments
///
/// Parameters:
/// - from_seg: The segment we're coming from
/// - to_seg: The segment we're going to
/// Calculate the traversal cost between two road segments
pub(crate) fn calculate_static_transition_cost(
    from_segment: &WaySegment,
    to_segment: &WaySegment,
) -> f64 {
    // Early validation
    if from_segment.coordinates.is_empty() || to_segment.coordinates.is_empty() {
        panic!("Segment coordinates invalid");
    }

    // Road type preference factor
    let road_type_factor = get_road_type_factor(&to_segment.highway_type);

    // Continuity bonus for staying on the same road
    let continuity_factor = if from_segment.osm_way_id == to_segment.osm_way_id {
        0.9 // 10% discount
    } else {
        1.0
    };

    // Calculate base cost combines all factors
    (road_type_factor * continuity_factor) + 1.0
}

/// Calculate the traversal cost between two road segments
///
/// Parameters:
/// - from_seg: The segment we're coming from
/// - to_seg: The segment we're going to
/// - from_segment_idx_node: Node index from which the from_segment will be entered
/// Calculate the dynamic traversal cost between two road segments
pub(crate) fn calculate_dynamic_transition_cost(
    from_segment: &WaySegment,
    to_segment: &WaySegment,
    from_segment_entry_idx: usize,
    from_segment_exit_idx: usize,
    gps_points: Option<&[Point<f64>]>,
) -> (f64, f64) {
    // Early validation
    if from_segment.coordinates.is_empty() || to_segment.coordinates.is_empty() {
        panic!("Segment coordinates invalid");
    }

    let static_cost = calculate_static_transition_cost(from_segment, to_segment);

    if from_segment.is_oneway && from_segment_entry_idx > from_segment_exit_idx {
        // We are entering past the connection which is invalid
        panic!("Entering past the connection point");
    }

    // Determine the entry point for the from_segment

    // In A* path finding, we're typically coming from some previous segment
    // For the start of a path, we might enter at any node
    // For intermediate segments, we typically enter at one node and exit at another

    // We need to handle different cases:
    // 1. Direct endpoint-to-endpoint connection
    // 2. Traversal through a segment from one node to another

    // Calculate distance along from_segment between entry and exit points
    let distance = if from_segment_exit_idx == from_segment_entry_idx {
        // No traversal (enter and exit at same point)
        0. // Minimal cost
    } else {
        // We need to sum up distances between consecutive coordinates
        let mut distance = 0.0;

        // Make sure indices are properly ordered for iteration
        let (start_idx, end_idx) = if from_segment_entry_idx <= from_segment_exit_idx {
            (from_segment_entry_idx, from_segment_exit_idx)
        } else {
            (from_segment_exit_idx, from_segment_entry_idx)
        };

        // Walk along the segment coordinates and sum distances
        for i in start_idx..end_idx {
            if i + 1 < from_segment.coordinates.len() {
                let p1 =
                    geo::Point::new(from_segment.coordinates[i].x, from_segment.coordinates[i].y);
                let p2 = geo::Point::new(
                    from_segment.coordinates[i + 1].x,
                    from_segment.coordinates[i + 1].y,
                );
                distance += Haversine.distance(p1, p2);
            }
        }

        distance
    };

    // Normalize distance to a reasonable cost value
    let distance_cost = (distance / 500.0).max(0.1) + 1.0;

    // Calculate base cost combines all factors
    let mut base_cost = distance_cost * static_cost;

    // Check GPS proximity if points are provided
    if let Some(points) = gps_points {
        // Check proximity to the destination segment (to_seg)
        for point in points {
            // Project the point to the segment to get the closest point
            let line = LineString::from(to_segment.coordinates.clone());
            let closest = line.closest_point(point);

            let projection = match closest {
                Closest::SinglePoint(p) => p,
                _ => continue, // Skip if no clear projection
            };

            let projection_distance = Haversine.distance(*point, projection);

            // If segment is close to a GPS point, apply strong negative cost
            if projection_distance <= DISTANCE_THRESHOLD_FOR_COST_BIAS_METER {
                // Apply a significant cost discount if the transition would bring us near a
                // strong segment candidate
                base_cost *= 0.1;
            }
        }
    }

    // Return base cost if no GPS proximity bonus applied
    (base_cost, distance)
}

/// Get factor for road type preference
pub(crate) fn get_road_type_factor(highway_type: &str) -> f64 {
    match highway_type {
        "motorway" => 0.5,
        "motorway_link" => 0.6,
        "trunk" => 0.7,
        "trunk_link" => 0.8,
        "primary" => 0.9,
        "primary_link" => 1.0,
        "secondary" => 1.2,
        "secondary_link" => 1.3,
        "tertiary" => 1.5,
        "tertiary_link" => 1.6,
        "residential" => 2.0,
        "unclassified" => 2.5,
        "service" => 3.0,
        _ => 3.5,
    }
}

// Check if two segments should be connected based on their properties and spatial relationship
/// Returns (is_compatible, should_connect, reason)
/// - is_compatible: True if segments have compatible properties (layers, road types)
/// - should_connect: True if segments should be connected in the network
/// - reason: Detailed explanation of the result
pub(crate) fn check_segment_connectivity(
    segment1: &WaySegment,
    segment2: &WaySegment,
) -> (bool, bool, String) {
    // 1. Check for layer differences
    let layer1 = segment1
        .metadata
        .as_ref()
        .and_then(|m| m.get("layer").map(|l| l.parse::<i8>().unwrap_or(0)))
        .unwrap_or(0);

    let layer2 = segment2
        .metadata
        .as_ref()
        .and_then(|m| m.get("layer").map(|l| l.parse::<i8>().unwrap_or(0)))
        .unwrap_or(0);

    if layer1 != layer2 {
        return (
            false,
            false,
            format!("Segments are on different layers: {} vs {}", layer1, layer2),
        );
    }

    // 2. Check bridge/tunnel status
    let is_bridge1 = segment1
        .metadata
        .as_ref()
        .and_then(|m| m.get("bridge").map(|v| v == "yes"))
        .unwrap_or(false);

    let is_bridge2 = segment2
        .metadata
        .as_ref()
        .and_then(|m| m.get("bridge").map(|v| v == "yes"))
        .unwrap_or(false);

    if is_bridge1 != is_bridge2 {
        return (
            false,
            false,
            format!(
                "Bridge mismatch: {} is {}a bridge, {} is {}a bridge",
                segment1.id,
                if is_bridge1 { "" } else { "not " },
                segment2.id,
                if is_bridge2 { "" } else { "not " }
            ),
        );
    }

    let is_tunnel1 = segment1
        .metadata
        .as_ref()
        .and_then(|m| m.get("tunnel").map(|v| v == "yes"))
        .unwrap_or(false);

    let is_tunnel2 = segment2
        .metadata
        .as_ref()
        .and_then(|m| m.get("tunnel").map(|v| v == "yes"))
        .unwrap_or(false);

    if is_tunnel1 != is_tunnel2 {
        return (
            false,
            false,
            format!(
                "Tunnel mismatch: {} is {}a tunnel, {} is {}a tunnel",
                segment1.id,
                if is_tunnel1 { "" } else { "not " },
                segment2.id,
                if is_tunnel2 { "" } else { "not " }
            ),
        );
    }

    // 3. Check road type compatibility
    let compatible_types =
        are_road_types_compatible(&segment1.highway_type, &segment2.highway_type);

    if !compatible_types {
        return (
            false,
            false,
            format!(
                "Incompatible road types: {} ({}) and {} ({})",
                segment1.highway_type, segment1.id, segment2.highway_type, segment2.id
            ),
        );
    }

    // 4. Now check if they should actually connect (spatial relationship)

    // First check for shared nodes - this is the most reliable indicator
    let mut shared_nodes = Vec::new();
    for node_id in &segment1.nodes {
        if segment2.nodes.contains(node_id) {
            shared_nodes.push(*node_id);
        }
    }

    if !shared_nodes.is_empty() {
        // Also check if these shared nodes are at endpoints
        let is_endpoint1 = shared_nodes.iter().any(|&node_id| {
            *segment1.nodes.first().unwrap() == node_id
                || *segment1.nodes.last().unwrap() == node_id
        });

        let is_endpoint2 = shared_nodes.iter().any(|&node_id| {
            *segment2.nodes.first().unwrap() == node_id
                || *segment2.nodes.last().unwrap() == node_id
        });

        let endpoint_info = if is_endpoint1 && is_endpoint2 {
            " (shared endpoint nodes)"
        } else {
            " (shared intermediate nodes)"
        };

        return (
            true,
            true,
            format!(
                "They share {} node(s): {:?}{}",
                shared_nodes.len(),
                shared_nodes,
                endpoint_info
            ),
        );
    }

    // Check for points that are very close to each other
    const CLOSE_DISTANCE_THRESHOLD: f64 = 1.0; // meters

    let seg1_start = segment1.coordinates.first().unwrap();
    let seg1_end = segment1.coordinates.last().unwrap();
    let seg2_start = segment2.coordinates.first().unwrap();
    let seg2_end = segment2.coordinates.last().unwrap();

    // Check all endpoint combinations for close proximity
    let start_start_point = Point::new(seg1_start.x, seg1_start.y);
    let start_start_dist =
        Haversine.distance(start_start_point, Point::new(seg2_start.x, seg2_start.y));
    if start_start_dist < CLOSE_DISTANCE_THRESHOLD {
        return (
            true,
            true,
            format!("Start points are very close ({:.2}m)", start_start_dist),
        );
    }

    let start_end_dist = Haversine.distance(start_start_point, Point::new(seg2_end.x, seg2_end.y));
    if start_end_dist < CLOSE_DISTANCE_THRESHOLD {
        return (
            true,
            true,
            format!(
                "Start point of #1 is close to end point of #2 ({:.2}m)",
                start_end_dist
            ),
        );
    }

    let end_start_point = Point::new(seg1_end.x, seg1_end.y);
    let end_start_dist =
        Haversine.distance(end_start_point, Point::new(seg2_start.x, seg2_start.y));
    if end_start_dist < CLOSE_DISTANCE_THRESHOLD {
        return (
            true,
            true,
            format!(
                "End point of #1 is close to start point of #2 ({:.2}m)",
                end_start_dist
            ),
        );
    }

    let end_end_dist = Haversine.distance(end_start_point, Point::new(seg2_end.x, seg2_end.y));
    if end_end_dist < CLOSE_DISTANCE_THRESHOLD {
        return (
            true,
            true,
            format!("End points are very close ({:.2}m)", end_end_dist),
        );
    }

    // Create LineStrings for intersection check
    let line1 = LineString::from(segment1.coordinates.clone());
    let line2 = LineString::from(segment2.coordinates.clone());

    // Check if lines intersect
    if line1.intersects(&line2) {
        return (true, true, "Lines geometrically intersect".to_string());
    }

    // At this point, segments are compatible but don't appear to connect
    (true, false, "No connection criteria met".to_string())
}

pub(crate) fn are_road_types_compatible(type1: &str, type2: &str) -> bool {
    // Same types are obviously compatible
    if type1 == type2 {
        return true;
    }

    // Define road class groups
    let highway_classes: Vec<Vec<&str>> = vec![
        vec!["motorway", "motorway_link"],
        vec!["trunk", "trunk_link"],
        vec!["primary", "primary_link"],
        vec!["secondary", "secondary_link"],
        vec!["tertiary", "tertiary_link"],
        vec!["residential", "unclassified", "service"],
        vec!["track", "path", "footway", "cycleway"],
    ];

    // Find which class each type belongs to
    let class1 = highway_classes
        .iter()
        .position(|class| class.contains(&type1));
    let class2 = highway_classes
        .iter()
        .position(|class| class.contains(&type2));

    match (class1, class2) {
        (Some(c1), Some(c2)) => {
            // Same class or adjacent classes are compatible
            (c1 as i32 - c2 as i32).abs() <= 1
        }
        _ => {
            // Unknown type, be conservative
            false
        }
    }
}
