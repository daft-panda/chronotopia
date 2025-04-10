import 'dart:async';
import 'package:chronotopia_app/models/activity_data.dart';
import 'package:flutter/services.dart';

class ActivityService {
  static const MethodChannel _channel = MethodChannel(
    'io.chronotopia.app/activity',
  );
  static const EventChannel _activityStream = EventChannel(
    'io.chronotopia.app/activity_stream',
  );

  StreamSubscription? _activitySubscription;
  final StreamController<ActivityData> _activityController =
      StreamController.broadcast();

  Stream<ActivityData> get activityUpdates => _activityController.stream;

  Future<void> initialize() async {
    try {
      await _channel.invokeMethod('initialize');
    } on PlatformException catch (e) {
      throw Exception('Failed to initialize activity service: ${e.message}');
    }
  }

  Future<void> startTracking() async {
    try {
      await _channel.invokeMethod('startTracking');
      _activitySubscription = _activityStream.receiveBroadcastStream().listen(
        _onActivityUpdate,
      );
    } on PlatformException catch (e) {
      throw Exception('Failed to start activity tracking: ${e.message}');
    }
  }

  Future<void> stopTracking() async {
    try {
      await _channel.invokeMethod('stopTracking');
      await _activitySubscription?.cancel();
      _activitySubscription = null;
    } on PlatformException catch (e) {
      throw Exception('Failed to stop activity tracking: ${e.message}');
    }
  }

  void _onActivityUpdate(dynamic event) {
    if (event is! Map) return;

    try {
      final activityTypeString = event['type'] as String;
      final activityType = ActivityType.values.firstWhere(
        (type) => type.toString() == activityTypeString,
        orElse: () => ActivityType.unknown,
      );

      final activityData = ActivityData(
        type: activityType,
        confidence: event['confidence'] ?? 0,
        timestamp: DateTime.parse(event['timestamp']),
        startTime:
            event['startTime'] != null
                ? DateTime.parse(event['startTime'])
                : null,
        endTime:
            event['endTime'] != null ? DateTime.parse(event['endTime']) : null,
        stepCount: event['stepCount'],
        calories: event['calories'],
        distance: event['distance'],
      );

      _activityController.add(activityData);
    } catch (e) {
      print('Error parsing activity data: $e');
    }
  }
}
