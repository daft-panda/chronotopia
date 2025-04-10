import 'package:chronotopia_app/models/activity_data.dart';
import 'package:chronotopia_app/models/location_data.dart';

class TripPoint {
  final LocationData location;
  final ActivityData? activity;

  TripPoint({required this.location, this.activity});

  Map<String, dynamic> toJson() => {
    'location': location.toJson(),
    'activity': activity?.toJson(),
  };
}

class TripSegment {
  final String id;
  final DateTime startTime;
  final DateTime? endTime;
  final ActivityType primaryActivityType;
  final List<TripPoint> points;
  final String? startPlaceName;
  final String? endPlaceName;
  final double? distance;
  final double? duration; // in seconds

  TripSegment({
    required this.id,
    required this.startTime,
    this.endTime,
    required this.primaryActivityType,
    required this.points,
    this.startPlaceName,
    this.endPlaceName,
    this.distance,
    this.duration,
  });

  Map<String, dynamic> toJson() => {
    'id': id,
    'startTime': startTime.toIso8601String(),
    'endTime': endTime?.toIso8601String(),
    'primaryActivityType': primaryActivityType.toString(),
    'points': points.map((p) => p.toJson()).toList(),
    'startPlaceName': startPlaceName,
    'endPlaceName': endPlaceName,
    'distance': distance,
    'duration': duration,
  };
}
