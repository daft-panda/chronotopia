import 'package:permission_handler/permission_handler.dart';
import 'dart:io';

enum PermissionStatus {
  granted,
  denied,
  permanentlyDenied,
  restricted, // iOS only
  limited, // iOS only
}

class PermissionHandler {
  // Check if all required permissions are granted
  Future<bool> areAllPermissionsGranted() async {
    final requiredPermissions = await _getRequiredPermissions();

    for (final permission in requiredPermissions) {
      final status = await permission.status;
      if (!status.isGranted) {
        return false;
      }
    }

    return true;
  }

  // Get list of required permissions based on platform
  Future<List<Permission>> _getRequiredPermissions() async {
    final permissions = <Permission>[];

    // iOS specific permissions
    if (Platform.isIOS) {
      permissions.add(
        Permission.locationWhenInUse,
      ); // iOS requires this before requesting locationAlways
      permissions.add(Permission.sensors);
      // iOS 13+ requires motion permissions for activity recognition
      permissions.add(Permission.sensors);
    }

    // Core permissions needed on both platforms
    permissions.add(Permission.locationAlways);
    permissions.add(Permission.activityRecognition);

    return permissions;
  }

  // Request all required permissions in sequence
  Future<Map<Permission, PermissionStatus>> requestRequiredPermissions() async {
    final requiredPermissions = await _getRequiredPermissions();
    final results = <Permission, PermissionStatus>{};

    // Request each permission individually for better UX
    for (final permission in requiredPermissions) {
      // First check if already granted
      final currentStatus = await permission.status;

      if (currentStatus.isGranted) {
        results[permission] = PermissionStatus.granted;
        continue;
      }

      // Request the permission
      final requestStatus = await permission.request();

      if (requestStatus.isGranted) {
        results[permission] = PermissionStatus.granted;
      } else if (requestStatus.isPermanentlyDenied) {
        results[permission] = PermissionStatus.permanentlyDenied;
      } else if (requestStatus.isDenied) {
        results[permission] = PermissionStatus.denied;
      } else if (requestStatus.isRestricted) {
        results[permission] = PermissionStatus.restricted;
      } else if (requestStatus.isLimited) {
        results[permission] = PermissionStatus.limited;
      }
    }

    return results;
  }

  // Check if any permission is permanently denied
  Future<bool> hasAnyPermanentlyDeniedPermission() async {
    final requiredPermissions = await _getRequiredPermissions();

    for (final permission in requiredPermissions) {
      final status = await permission.status;
      if (status.isPermanentlyDenied) {
        return true;
      }
    }

    return false;
  }

  // Get list of permissions that are permanently denied
  Future<List<Permission>> getPermanentlyDeniedPermissions() async {
    final requiredPermissions = await _getRequiredPermissions();
    final deniedPermissions = <Permission>[];

    for (final permission in requiredPermissions) {
      final status = await permission.status;
      if (status.isPermanentlyDenied) {
        deniedPermissions.add(permission);
      }
    }

    return deniedPermissions;
  }

  // Open app settings
  Future<bool> openAppSettings() async {
    return await openAppSettings();
  }

  // Get human-readable permission name
  String getPermissionName(Permission permission) {
    switch (permission) {
      case Permission.locationAlways:
        return 'Background Location';
      case Permission.locationWhenInUse:
        return 'Location';
      case Permission.activityRecognition:
        return 'Activity Recognition';
      case Permission.sensors:
        return 'Motion Sensors';
      default:
        return permission.toString().split('.').last;
    }
  }

  // Get reason why this permission is needed
  String getPermissionReason(Permission permission) {
    switch (permission) {
      case Permission.locationAlways:
        return 'Chronotopia needs to track your location even when the app is not in use to build your timeline accurately.';
      case Permission.locationWhenInUse:
        return 'Chronotopia needs to access your location to track your movements and build your timeline.';
      case Permission.activityRecognition:
        return 'This allows the app to detect your transportation mode (walking, cycling, driving, etc.) for more accurate timeline data.';
      case Permission.sensors:
        return 'Motion sensors help detect your activities and improve tracking accuracy.';
      default:
        return 'This permission is required for the app to function properly.';
    }
  }
}
