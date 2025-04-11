import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:uuid/uuid.dart';

class BackgroundService extends ChangeNotifier {
  // Method channels for communication with native code
  static const MethodChannel _serviceChannel = MethodChannel(
    'io.chronotopia.app/service',
  );
  static const MethodChannel _batteryChannel = MethodChannel(
    'io.chronotopia.app/battery',
  );
  static const MethodChannel _settingsChannel = MethodChannel(
    'io.chronotopia.app/settings',
  );

  // Event channels for streaming data
  static const EventChannel _locationEventChannel = EventChannel(
    'io.chronotopia.app/location_stream',
  );
  static const EventChannel _activityEventChannel = EventChannel(
    'io.chronotopia.app/activity_stream',
  );

  // Service state
  bool _isRunning = false;
  bool _isInitialized = false;
  String? _authToken;
  String _serverUrl = 'ingest.chronotopia.io';
  bool _useTLS = true;

  // Upload status cache
  Map<String, dynamic> _uploadStatus = {};
  DateTime? _lastStatusUpdate;

  // Service state getters
  bool get isRunning => _isRunning;
  bool get isInitialized => _isInitialized;
  String? get authToken => _authToken;
  String get serverUrl => _serverUrl;
  bool get useTLS => _useTLS;

  // Initialize the service
  Future<void> initialize() async {
    try {
      // Check if service is already running
      final isRunning = await _serviceChannel.invokeMethod('isServiceRunning');
      _isRunning = isRunning ?? false;

      // Load or generate auth token
      await _loadOrGenerateAuthToken();

      // Load server settings
      await _loadServerSettings();

      // Set up event listeners
      _setupEventListeners();

      _isInitialized = true;
      notifyListeners();
    } catch (e) {
      debugPrint('Error initializing background service: $e');
    }
  }

  // Load existing auth token or generate a new one
  Future<void> _loadOrGenerateAuthToken() async {
    final prefs = await SharedPreferences.getInstance();
    _authToken = prefs.getString('auth_token');

    if (_authToken == null) {
      // Generate a new UUID v4 token
      final newToken = const Uuid().v4();
      await setAuthToken(newToken);
    }
  }

  // Load server settings from preferences
  Future<void> _loadServerSettings() async {
    final prefs = await SharedPreferences.getInstance();
    _serverUrl = prefs.getString('server_url') ?? 'ingest.chronotopia.io';
    _useTLS = prefs.getBool('use_tls') ?? true;
  }

  // Set a new authorization token
  Future<void> setAuthToken(String token) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('auth_token', token);
    _authToken = token;

    // Pass it to the native code
    if (Platform.isIOS) {
      try {
        await _serviceChannel.invokeMethod('setAuthToken', {'token': token});
      } catch (e) {
        debugPrint('Error setting auth token in iOS: $e');
      }
    }

    // For Android, we pass it via settings method
    if (Platform.isAndroid) {
      try {
        await _serviceChannel.invokeMethod('setAuthToken', {'token': token});
      } catch (e) {
        debugPrint('Error setting auth token in Android: $e');
        // If the method isn't implemented in Android yet, apply settings
        await applySettings();
      }
    }

    notifyListeners();
  }

  // Set server settings
  Future<void> setServerSettings(String serverUrl, bool useTLS) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('server_url', serverUrl);
    await prefs.setBool('use_tls', useTLS);

    _serverUrl = serverUrl;
    _useTLS = useTLS;

    // Pass the settings to native code
    if (Platform.isIOS) {
      try {
        await _serviceChannel.invokeMethod('setServerSettings', {
          'serverUrl': serverUrl,
          'useTLS': useTLS,
        });
      } catch (e) {
        debugPrint('Error setting server settings in iOS: $e');
      }
    }

    // For Android
    if (Platform.isAndroid) {
      try {
        await _serviceChannel.invokeMethod('setServerSettings', {
          'serverUrl': serverUrl,
          'useTLS': useTLS,
        });
      } catch (e) {
        debugPrint('Error setting server settings in Android: $e');
        // If the method isn't implemented in Android yet, apply settings
        await applySettings();
      }
    }

    notifyListeners();
  }

  // Get upload status information
  Future<Map<String, dynamic>> getUploadStatus() async {
    try {
      // Don't request status too frequently (max once per second)
      final now = DateTime.now();
      if (_lastStatusUpdate != null &&
          now.difference(_lastStatusUpdate!).inSeconds < 1) {
        return _uploadStatus;
      }

      final status = await _serviceChannel.invokeMethod('getUploadStatus');
      if (status is Map) {
        _uploadStatus = Map<String, dynamic>.from(status);
        _lastStatusUpdate = now;
      }
      return _uploadStatus;
    } catch (e) {
      debugPrint('Error getting upload status: $e');
      return {'error': e.toString()};
    }
  }

  // Force an immediate sync
  Future<void> forceSyncNow() async {
    try {
      await _serviceChannel.invokeMethod('forceSyncNow');
    } catch (e) {
      debugPrint('Error forcing sync: $e');
    }
  }

  // Start tracking
  Future<void> startTracking() async {
    try {
      await _serviceChannel.invokeMethod('startServices');
      _isRunning = true;
      notifyListeners();
    } catch (e) {
      debugPrint('Error starting tracking: $e');
    }
  }

  // Stop tracking
  Future<void> stopTracking() async {
    try {
      await _serviceChannel.invokeMethod('stopServices');
      _isRunning = false;
      notifyListeners();
    } catch (e) {
      debugPrint('Error stopping tracking: $e');
    }
  }

  // Restart tracking services
  Future<void> restartTracking() async {
    try {
      await _serviceChannel.invokeMethod('restartServices');
      _isRunning = true;
      notifyListeners();
    } catch (e) {
      debugPrint('Error restarting tracking: $e');
    }
  }

  // Apply settings from shared preferences
  Future<void> applySettings() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final trackInLowBattery = prefs.getBool('track_in_low_battery') ?? true;
      final locationAccuracy = prefs.getInt('location_accuracy') ?? 1;
      final uploadFrequency = prefs.getInt('upload_frequency') ?? 1;
      final syncOnWifiOnly = prefs.getBool('sync_on_wifi_only') ?? false;
      final serverUrl =
          prefs.getString('server_url') ?? 'ingest.chronotopia.io';
      final useTLS = prefs.getBool('use_tls') ?? true;
      final authToken = prefs.getString('auth_token') ?? const Uuid().v4();

      await _serviceChannel.invokeMethod('applySettings', {
        'trackInLowBattery': trackInLowBattery,
        'locationAccuracy': locationAccuracy,
        'uploadFrequency': uploadFrequency,
        'syncOnWifiOnly': syncOnWifiOnly,
        'serverUrl': serverUrl,
        'useTLS': useTLS,
        'authToken': authToken,
      });
    } catch (e) {
      debugPrint('Error applying settings: $e');
    }
  }

  // Set up event listeners for location and activity events
  void _setupEventListeners() {
    // Location events
    _locationEventChannel.receiveBroadcastStream().listen(
      (event) {
        // Parse location data
        debugPrint('Location update: $event');
        // Here you could process and store the location data
      },
      onError: (error) {
        debugPrint('Error in location stream: $error');
      },
    );

    // Activity events
    _activityEventChannel.receiveBroadcastStream().listen(
      (event) {
        // Parse activity data
        debugPrint('Activity update: $event');
        // Here you could process and store the activity data
      },
      onError: (error) {
        debugPrint('Error in activity stream: $error');
      },
    );
  }

  // Get battery level
  Future<int> getBatteryLevel() async {
    try {
      final level = await _batteryChannel.invokeMethod('getBatteryLevel');
      return level ?? -1;
    } catch (e) {
      debugPrint('Error getting battery level: $e');
      return -1;
    }
  }

  // Check if device is charging
  Future<bool> isCharging() async {
    try {
      final charging = await _batteryChannel.invokeMethod('isCharging');
      return charging ?? false;
    } catch (e) {
      debugPrint('Error checking if device is charging: $e');
      return false;
    }
  }

  // Request battery optimization exemption (Android only)
  Future<bool> requestBatteryOptimizationExemption() async {
    if (!Platform.isAndroid) return true;

    try {
      final result = await _batteryChannel.invokeMethod(
        'requestBatteryOptimizationExemption',
      );
      return result ?? false;
    } catch (e) {
      debugPrint('Error requesting battery optimization exemption: $e');
      return false;
    }
  }

  // Open relevant settings
  Future<void> openBatterySettings() async {
    try {
      await _settingsChannel.invokeMethod('openBatteryOptimizationSettings');
    } catch (e) {
      debugPrint('Error opening battery settings: $e');
    }
  }

  Future<void> openLocationSettings() async {
    try {
      await _settingsChannel.invokeMethod('openLocationSettings');
    } catch (e) {
      debugPrint('Error opening location settings: $e');
    }
  }

  Future<void> openAppSettings() async {
    try {
      await _settingsChannel.invokeMethod('openAppSettings');
    } catch (e) {
      debugPrint('Error opening app settings: $e');
    }
  }

  // Generate a random auth token (for testing)
  Future<void> generateRandomAuthToken() async {
    final newToken = const Uuid().v4();
    await setAuthToken(newToken);
  }

  @override
  void dispose() {
    // Clean up resources
    super.dispose();
  }
}
