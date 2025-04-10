import 'dart:async';

import 'package:chronotopia_app/models/activity_data.dart';
import 'package:chronotopia_app/models/location_data.dart';
import 'package:chronotopia_app/models/trip_data.dart';

class DataSyncService {
  Timer? _syncTimer;
  final List<TripPoint> _pendingPoints = [];

  Future<void> initialize() async {
    // Load any cached data that wasn't sent previously
  }

  Future<void> startPeriodicSync() async {
    // Sync every 15 minutes
    _syncTimer = Timer.periodic(Duration(minutes: 15), (_) => _syncData());
  }

  Future<void> stopPeriodicSync() async {
    _syncTimer?.cancel();
    _syncTimer = null;

    // Force one last sync
    await _syncData();
  }

  Future<void> addPoint(LocationData location, ActivityData? activity) async {
    final point = TripPoint(location: location, activity: activity);
    _pendingPoints.add(point);

    // If we have too many points, trigger a sync
    if (_pendingPoints.length > 100) {
      await _syncData();
    }
  }

  Future<void> _syncData() async {
    if (_pendingPoints.isEmpty) return;

    try {
      // Here you would call your gRPC client to send data to server
      // final response = await _grpcClient.sendTripData(_pendingPoints);

      // For now, we'll just clear the pending points
      _pendingPoints.clear();
    } catch (e) {
      print('Failed to sync data: $e');
      // Keep the pending points to try again later
    }
  }
}
