// services/enhanced_background_service.dart
import 'dart:async';
import 'dart:io';
import 'package:chronotopia_app/services/activity_service.dart';
import 'package:chronotopia_app/services/data_sync_service.dart';
import 'package:chronotopia_app/services/location_service.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:workmanager/workmanager.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

// Background tasks constants
const String locationTrackingTask = "io.chronotopia.app.locationTrackingTask";
const String dataUploadTask = "io.chronotopia.app.dataUploadTask";
const String serviceRestartTask = "io.chronotopia.app.serviceRestartTask";

@pragma('vm:entry-point')
void callbackDispatcher() {
  Workmanager().executeTask((task, inputData) async {
    switch (task) {
      case locationTrackingTask:
        // This is for when the service gets killed but Workmanager stays alive
        await _handleLocationTrackingTask();
        break;
      case dataUploadTask:
        await _handleDataUploadTask();
        break;
      case serviceRestartTask:
        await _handleServiceRestartTask();
        break;
    }
    return true;
  });
}

Future<void> _handleLocationTrackingTask() async {
  // Check if tracking is enabled
  final prefs = await SharedPreferences.getInstance();
  final isTrackingEnabled = prefs.getBool('tracking_enabled') ?? false;

  if (!isTrackingEnabled) {
    return;
  }

  // Start native services if they're not running
  final methodChannel = MethodChannel('io.chronotopia.app/service');
  final isRunning =
      await methodChannel.invokeMethod<bool>('isServiceRunning') ?? false;

  if (!isRunning) {
    await methodChannel.invokeMethod('startServices');
  }
}

Future<void> _handleDataUploadTask() async {
  // Check if tracking is enabled
  final prefs = await SharedPreferences.getInstance();
  final isTrackingEnabled = prefs.getBool('tracking_enabled') ?? false;
  final syncOnWifiOnly = prefs.getBool('sync_on_wifi_only') ?? false;

  if (!isTrackingEnabled) {
    return;
  }

  // Check connectivity if Wi-Fi only is enabled
  if (syncOnWifiOnly) {
    final connectivity = await Connectivity().checkConnectivity();
    if (connectivity != ConnectivityResult.wifi) {
      // Skip upload because we're not on Wi-Fi
      return;
    }
  }

  // Perform data upload
  final dataSyncService = DataSyncService();
  await dataSyncService.initialize();
}

Future<void> _handleServiceRestartTask() async {
  // Check if tracking is enabled
  final prefs = await SharedPreferences.getInstance();
  final isTrackingEnabled = prefs.getBool('tracking_enabled') ?? false;

  if (!isTrackingEnabled) {
    return;
  }

  // Force restart the native services
  final methodChannel = MethodChannel('io.chronotopia.app/service');
  await methodChannel.invokeMethod('restartServices');
}

class BackgroundService extends ChangeNotifier {
  final LocationService _locationService = LocationService();
  final ActivityService _activityService = ActivityService();
  final DataSyncService _dataSyncService = DataSyncService();

  final FlutterLocalNotificationsPlugin _notificationsPlugin =
      FlutterLocalNotificationsPlugin();

  // Platform channels for native communication
  final MethodChannel _serviceChannel = const MethodChannel(
    'io.chronotopia.app/service',
  );
  final MethodChannel _batteryChannel = const MethodChannel(
    'io.chronotopia.app/battery',
  );

  bool _isRunning = false;
  String _manufacturer = 'unknown';
  Timer? _heartbeatTimer;
  Timer? _backupLocationTimer;

  bool get isRunning => _isRunning;

  BackgroundService() {
    _initializeDeviceInfo();
  }

  Future<void> _initializeDeviceInfo() async {
    final deviceInfo = DeviceInfoPlugin();

    if (Platform.isAndroid) {
      final androidInfo = await deviceInfo.androidInfo;
      _manufacturer = androidInfo.manufacturer.toLowerCase();
    }
  }

  Future<void> initialize() async {
    await _initializeNotifications();
    await _initializeWorkManager();
    // await _initializeFirebaseMessaging();

    await _locationService.initialize();
    await _activityService.initialize();
    await _dataSyncService.initialize();

    // Check if service was running before
    final prefs = await SharedPreferences.getInstance();
    final wasRunning = prefs.getBool('tracking_enabled') ?? false;

    if (wasRunning) {
      await startTracking();
    }
  }

  Future<void> _initializeNotifications() async {
    const AndroidInitializationSettings initializationSettingsAndroid =
        AndroidInitializationSettings('ic_notification');

    final DarwinInitializationSettings initializationSettingsIOS =
        DarwinInitializationSettings(
          requestAlertPermission: false,
          requestBadgePermission: false,
          requestSoundPermission: false,
        );

    final InitializationSettings initializationSettings =
        InitializationSettings(
          android: initializationSettingsAndroid,
          iOS: initializationSettingsIOS,
        );

    await _notificationsPlugin.initialize(initializationSettings);
  }

  Future<void> _initializeWorkManager() async {
    await Workmanager().initialize(callbackDispatcher, isInDebugMode: false);
  }

  Future<void> startTracking() async {
    if (_isRunning) return;

    // Save tracking state
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('tracking_enabled', true);

    // Request ignoring battery optimizations on Android
    if (Platform.isAndroid) {
      await _requestBatteryOptimizationExemption();
    }

    // Start native services
    await _locationService.startTracking();
    await _activityService.startTracking();
    await _dataSyncService.startPeriodicSync();

    // Show notification
    await _showTrackingNotification();

    // Register work manager tasks for background processing
    await _registerWorkManagerTasks();

    // Start heartbeat timer to check service health
    _startHeartbeatTimer();

    // Start backup timer that uses Flutter's timer to track location
    // if all else fails (works when app is in foreground)
    _startBackupLocationTimer();

    _isRunning = true;
    notifyListeners();
  }

  Future<void> stopTracking() async {
    if (!_isRunning) return;

    // Save tracking state
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('tracking_enabled', false);

    // Stop native services
    await _locationService.stopTracking();
    await _activityService.stopTracking();
    await _dataSyncService.stopPeriodicSync();

    // Cancel all work manager tasks
    await Workmanager().cancelAll();

    // Remove notification
    await _notificationsPlugin.cancel(1);

    // Stop timers
    _heartbeatTimer?.cancel();
    _backupLocationTimer?.cancel();

    _isRunning = false;
    notifyListeners();
  }

  Future<void> applySettings() async {
    final prefs = await SharedPreferences.getInstance();
    final trackInLowBattery = prefs.getBool('track_in_low_battery') ?? true;
    final locationAccuracy = prefs.getInt('location_accuracy') ?? 1;
    final uploadFrequency = prefs.getInt('upload_frequency') ?? 1;
    final syncOnWifiOnly = prefs.getBool('sync_on_wifi_only') ?? false;
    final showNotification = prefs.getBool('show_notification') ?? true;
    final serverUrl = prefs.getString('server_url') ?? "ingest.chronotopia.io";

    // Apply settings to services
    await _serviceChannel.invokeMethod('applySettings', {
      'trackInLowBattery': trackInLowBattery,
      'locationAccuracy': locationAccuracy,
      'uploadFrequency': uploadFrequency,
      'syncOnWifiOnly': syncOnWifiOnly,
      'serverUrl': serverUrl,
    });

    // Update notification visibility
    if (showNotification && _isRunning) {
      await _showTrackingNotification();
    } else {
      await _notificationsPlugin.cancel(1);
    }

    // Re-register work manager tasks with new frequencies
    if (_isRunning) {
      await _registerWorkManagerTasks();
    }
  }

  Future<void> _requestBatteryOptimizationExemption() async {
    try {
      bool isExempt =
          await _batteryChannel.invokeMethod(
            'isIgnoringBatteryOptimizations',
          ) ??
          false;

      if (!isExempt) {
        // Request exemption
        await _batteryChannel.invokeMethod(
          'requestBatteryOptimizationExemption',
        );

        // Also apply manufacturer-specific optimizations
        await _applyManufacturerSpecificOptimizations();
      }
    } catch (e) {
      print('Failed to request battery optimization exemption: $e');
    }
  }

  Future<void> _applyManufacturerSpecificOptimizations() async {
    try {
      // Different manufacturers have different ways of handling background services
      if (_manufacturer.contains('xiaomi') || _manufacturer.contains('redmi')) {
        await _serviceChannel.invokeMethod('applyXiaomiOptimizations');
      } else if (_manufacturer.contains('samsung')) {
        await _serviceChannel.invokeMethod('applySamsungOptimizations');
      } else if (_manufacturer.contains('huawei')) {
        await _serviceChannel.invokeMethod('applyHuaweiOptimizations');
      } else if (_manufacturer.contains('oppo') ||
          _manufacturer.contains('oneplus') ||
          _manufacturer.contains('realme')) {
        await _serviceChannel.invokeMethod('applyOppoOptimizations');
      }
    } catch (e) {
      print('Failed to apply manufacturer optimizations: $e');
    }
  }

  Future<void> _showTrackingNotification() async {
    final prefs = await SharedPreferences.getInstance();
    final showNotification = prefs.getBool('show_notification') ?? true;

    if (!showNotification) {
      return;
    }

    const AndroidNotificationDetails androidPlatformChannelSpecifics =
        AndroidNotificationDetails(
          'tracking_channel',
          'Location Tracking',
          channelDescription: 'Used for location tracking service',
          importance: Importance.low,
          priority: Priority.low,
          ongoing: true,
          showWhen: false,
        );

    const DarwinNotificationDetails iOSPlatformChannelSpecifics =
        DarwinNotificationDetails(
          presentAlert: false,
          presentBadge: false,
          presentSound: false,
        );

    const NotificationDetails platformChannelSpecifics = NotificationDetails(
      android: androidPlatformChannelSpecifics,
      iOS: iOSPlatformChannelSpecifics,
    );

    await _notificationsPlugin.show(
      1,
      'Trip Tracker Active',
      'Tracking your trips and visited places',
      platformChannelSpecifics,
    );
  }

  Future<void> _registerWorkManagerTasks() async {
    // Get the settings
    final prefs = await SharedPreferences.getInstance();
    final uploadFrequency = prefs.getInt('upload_frequency') ?? 1;

    // Convert upload frequency to actual intervals
    int uploadIntervalMinutes;
    switch (uploadFrequency) {
      case 0: // Low
        uploadIntervalMinutes = 120; // 2 hours
        break;
      case 1: // Balanced
        uploadIntervalMinutes = 30; // 30 minutes
        break;
      case 2: // High
        uploadIntervalMinutes = 15; // 15 minutes
        break;
      default:
        uploadIntervalMinutes = 30;
    }

    // Cancel existing tasks
    await Workmanager().cancelAll();

    // Register location tracking task (runs more frequently, every 15 minutes)
    await Workmanager().registerPeriodicTask(
      locationTrackingTask,
      locationTrackingTask,
      frequency: Duration(minutes: 15),
      constraints: Constraints(
        networkType: NetworkType.not_required,
        requiresBatteryNotLow: false,
        requiresCharging: false,
        requiresDeviceIdle: false,
      ),
      initialDelay: Duration(minutes: 1),
      backoffPolicy: BackoffPolicy.linear,
      existingWorkPolicy: ExistingWorkPolicy.replace,
    );

    // Register data upload task (runs based on frequency setting)
    await Workmanager().registerPeriodicTask(
      dataUploadTask,
      dataUploadTask,
      frequency: Duration(minutes: uploadIntervalMinutes),
      constraints: Constraints(
        networkType: NetworkType.connected,
        requiresBatteryNotLow: false,
        requiresCharging: false,
        requiresDeviceIdle: false,
      ),
      initialDelay: Duration(minutes: 5),
      backoffPolicy: BackoffPolicy.linear,
      existingWorkPolicy: ExistingWorkPolicy.replace,
    );

    // Register service restart task (runs every 6 hours)
    // This is a last resort measure to restart services if they get killed
    await Workmanager().registerPeriodicTask(
      serviceRestartTask,
      serviceRestartTask,
      frequency: Duration(hours: 6),
      constraints: Constraints(
        networkType: NetworkType.not_required,
        requiresBatteryNotLow: false,
        requiresCharging: false,
        requiresDeviceIdle: false,
      ),
      initialDelay: Duration(hours: 1),
      backoffPolicy: BackoffPolicy.linear,
      existingWorkPolicy: ExistingWorkPolicy.replace,
    );

    // Schedule one-time task that runs soon to ensure services are running
    await Workmanager().registerOneOffTask(
      'immediate_service_check',
      locationTrackingTask,
      initialDelay: Duration(minutes: 1),
      constraints: Constraints(
        networkType: NetworkType.not_required,
        requiresBatteryNotLow: false,
        requiresCharging: false,
        requiresDeviceIdle: false,
      ),
    );
  }

  void _startHeartbeatTimer() {
    // Cancel existing timer if any
    _heartbeatTimer?.cancel();

    // Run every 5 minutes
    _heartbeatTimer = Timer.periodic(const Duration(minutes: 5), (timer) async {
      await _checkServiceHealth();
    });
  }

  Future<void> _checkServiceHealth() async {
    try {
      // Check if native services are still running
      final isServiceRunning =
          await _serviceChannel.invokeMethod<bool>('isServiceRunning') ?? false;

      if (!isServiceRunning && _isRunning) {
        // Services have been killed, try to restart them
        print('Services not running, attempting to restart...');
        await _serviceChannel.invokeMethod('startServices');

        // Show notification to user
        const AndroidNotificationDetails androidPlatformChannelSpecifics =
            AndroidNotificationDetails(
              'service_channel',
              'Service Status',
              channelDescription: 'Used for service status notifications',
              importance: Importance.high,
              priority: Priority.high,
            );

        const NotificationDetails platformChannelSpecifics =
            NotificationDetails(android: androidPlatformChannelSpecifics);

        await _notificationsPlugin.show(
          2,
          'Tracking Restarted',
          'Trip Tracker had to restart tracking services',
          platformChannelSpecifics,
        );
      }

      // Check battery level and adjust tracking frequency if needed
      await _checkBatteryAndAdjust();
    } catch (e) {
      print('Error checking service health: $e');
    }
  }

  Future<void> _checkBatteryAndAdjust() async {
    try {
      final batteryLevel =
          await _batteryChannel.invokeMethod<int>('getBatteryLevel') ?? 100;
      final isCharging =
          await _batteryChannel.invokeMethod<bool>('isCharging') ?? false;
      final prefs = await SharedPreferences.getInstance();
      final trackInLowBattery = prefs.getBool('track_in_low_battery') ?? true;

      // If battery is low and not charging, and user doesn't want to track in low battery
      if (batteryLevel < 20 && !isCharging && !trackInLowBattery) {
        // Reduce tracking frequency
        await _serviceChannel.invokeMethod('setLowPowerMode', true);
      } else {
        // Normal tracking frequency
        await _serviceChannel.invokeMethod('setLowPowerMode', false);
      }
    } catch (e) {
      print('Error checking battery status: $e');
    }
  }

  void _startBackupLocationTimer() {
    // Cancel existing timer if any
    _backupLocationTimer?.cancel();

    // This is a backup mechanism that only works when the app is in the foreground
    // But it might help in some edge cases where the service gets killed
    // Run every 2 minutes
    _backupLocationTimer = Timer.periodic(const Duration(minutes: 2), (
      timer,
    ) async {
      if (!_isRunning) {
        timer.cancel();
        return;
      }

      try {
        // Request a single location update and store it
        await _serviceChannel.invokeMethod('requestSingleLocationUpdate');
      } catch (e) {
        print('Error requesting backup location: $e');
      }
    });
  }
}
