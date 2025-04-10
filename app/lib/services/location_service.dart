import 'dart:async';
import 'package:chronotopia_app/models/location_data.dart';
import 'package:flutter/services.dart';

class LocationService {
  static const MethodChannel _channel = MethodChannel(
    'io.chronotopia.app/location',
  );
  static const EventChannel _locationStream = EventChannel(
    'io.chronotopia.app/location_stream',
  );

  StreamSubscription? _locationSubscription;
  final StreamController<LocationData> _locationController =
      StreamController.broadcast();

  Stream<LocationData> get locationUpdates => _locationController.stream;

  Future<void> initialize() async {
    try {
      await _channel.invokeMethod('initialize');
    } on PlatformException catch (e) {
      throw Exception('Failed to initialize location service: ${e.message}');
    }
  }

  Future<void> startTracking() async {
    try {
      await _channel.invokeMethod('startTracking');
      _locationSubscription = _locationStream.receiveBroadcastStream().listen(
        _onLocationUpdate,
      );
    } on PlatformException catch (e) {
      throw Exception('Failed to start location tracking: ${e.message}');
    }
  }

  Future<void> stopTracking() async {
    try {
      await _channel.invokeMethod('stopTracking');
      await _locationSubscription?.cancel();
      _locationSubscription = null;
    } on PlatformException catch (e) {
      throw Exception('Failed to stop location tracking: ${e.message}');
    }
  }

  void _onLocationUpdate(dynamic event) {
    if (event is! Map) return;

    try {
      final locationData = LocationData(
        latitude: event['latitude'],
        longitude: event['longitude'],
        altitude: event['altitude'],
        accuracy: event['accuracy'],
        heading: event['heading'],
        speed: event['speed'],
        timestamp: DateTime.parse(event['timestamp']),
        placeId: event['placeId'],
        placeName: event['placeName'],
        address: event['address'],
        administrativeArea: event['administrativeArea'],
        country: event['country'],
        batteryLevel: event['batteryLevel'] ?? 0,
        networkType: event['networkType'],
      );

      _locationController.add(locationData);
    } catch (e) {
      print('Error parsing location data: $e');
    }
  }
}
