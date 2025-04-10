enum ActivityType {
  inVehicle,
  onBicycle,
  onFoot,
  running,
  still,
  tilting,
  walking,
  unknown,
}

class ActivityData {
  final ActivityType type;
  final int confidence; // 0-100
  final DateTime timestamp;
  final DateTime? startTime;
  final DateTime? endTime;
  final double? stepCount;
  final double? calories;
  final double? distance;

  ActivityData({
    required this.type,
    required this.confidence,
    required this.timestamp,
    this.startTime,
    this.endTime,
    this.stepCount,
    this.calories,
    this.distance,
  });

  Map<String, dynamic> toJson() => {
    'type': type.toString(),
    'confidence': confidence,
    'timestamp': timestamp.toIso8601String(),
    'startTime': startTime?.toIso8601String(),
    'endTime': endTime?.toIso8601String(),
    'stepCount': stepCount,
    'calories': calories,
    'distance': distance,
  };
}
